"""Retrieval-quality metrics for the font similarity embedder.

These are pure functions over ``(N, D)`` embedding matrices plus per-item
labels.  They do no I/O and know nothing about the model or the renderer, so
both the evaluation script (``scripts/similarity_eval.py``) and, later, the
training validation loop can share one definition of "good".

Two conventions are baked in because the shipped index (``build_font_index``)
uses them:

* embeddings are row-L2-normalized, and
* they are mean-centered first, because the raw embeddings are strongly
  mean-shifted and uncentered cosine similarities collapse into a narrow band.

Every function takes ``center`` so the effect can be measured, but defaults to
``True``.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


def _agreement_matrix(labels: Sequence[Hashable]) -> torch.Tensor:
    arr = np.asarray(labels)
    return torch.from_numpy(arr[:, None] == arr[None, :])


def prepare(embeddings: torch.Tensor, *, center: bool = True) -> torch.Tensor:
    """Return row-L2-normalized embeddings, optionally mean-centered first."""
    x = embeddings.float()
    if center:
        x = x - x.mean(dim=0, keepdim=True)
    return F.normalize(x, p=2, dim=-1, eps=1e-8)


def _ranks_from_sim(
    sim: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank of the highest-scoring target per row.

    Returns ``(rank, has_target)`` where ``rank`` is 1-based (0 where a row has
    no target at all).
    """
    order = sim.argsort(dim=-1, descending=True)
    hit = torch.gather(target.float(), 1, order)
    rank = hit.argmax(dim=-1) + 1
    has_target = target.any(dim=1)
    return torch.where(has_target, rank, torch.zeros_like(rank)), has_target


@dataclass(frozen=True)
class RetrievalScore:
    """Recall@k and MRR over the queries that had at least one target."""

    recall: dict[int, float]
    mrr: float
    n_queries: int
    n_pool: int

    def as_dict(self) -> dict:
        return {
            **{f"recall@{k}": v for k, v in self.recall.items()},
            "mrr": self.mrr,
            "n_queries": self.n_queries,
            "n_pool": self.n_pool,
        }


def sibling_retrieval(
    embeddings: torch.Tensor,
    groups: Sequence[Hashable],
    *,
    ks: Sequence[int] = (1, 3, 5),
    center: bool = True,
) -> RetrievalScore:
    """Retrieve another font in the same group (e.g. same family).

    Rows with no same-group sibling are excluded from the score but counted in
    ``n_pool``.
    """
    x = prepare(embeddings, center=center)
    n = x.shape[0]
    target = _agreement_matrix(groups)
    target.fill_diagonal_(False)
    if n < 2 or not bool(target.any()):
        return RetrievalScore({k: 0.0 for k in ks}, 0.0, 0, n)

    sim = x @ x.T
    sim.fill_diagonal_(float("-inf"))
    rank, has_target = _ranks_from_sim(sim, target)
    ranks = rank[has_target].float()
    recall = {k: float((ranks <= k).float().mean()) for k in ks}
    return RetrievalScore(recall, float((1.0 / ranks).mean()), int(has_target.sum()), n)


def label_agreement_at_k(
    embeddings: torch.Tensor,
    labels: Sequence[Hashable],
    *,
    k: int = 5,
    center: bool = True,
) -> dict:
    """Fraction of each item's top-k neighbours that share its label.

    A coarse semantic sanity check (e.g. do neighbours share a category?).
    """
    x = prepare(embeddings, center=center)
    n = x.shape[0]
    k = min(k, n - 1)
    if k <= 0:
        return {"overall": 0.0, "per_label": {}}

    sim = x @ x.T
    sim.fill_diagonal_(float("-inf"))
    topk = sim.topk(k, dim=-1).indices  # (N, k)
    hit = _agreement_matrix(labels)[torch.arange(n).unsqueeze(1), topk]  # (N, k)

    per_label: dict[str, float] = {}
    label_arr = np.asarray(labels)
    for label in sorted(set(labels), key=str):
        rows = torch.from_numpy(np.nonzero(label_arr == label)[0])
        per_label[str(label)] = float(hit[rows].float().mean())
    return {"overall": float(hit.float().mean()), "per_label": per_label}


def cross_set_purity(
    query_embeddings: torch.Tensor,
    query_labels: Sequence[Hashable],
    index_embeddings: torch.Tensor,
    index_labels: Sequence[Hashable],
    *,
    k: int = 5,
) -> dict:
    """Label agreement when the neighbours come from a *different* set.

    Measures generalisation: does an unseen font's nearest library neighbours
    share its label?  Both sets are centered by the index mean so the two share
    an origin.
    """
    mu = index_embeddings.float().mean(dim=0, keepdim=True)
    q = F.normalize(query_embeddings.float() - mu, p=2, dim=-1, eps=1e-8)
    ix = F.normalize(index_embeddings.float() - mu, p=2, dim=-1, eps=1e-8)
    k = min(k, ix.shape[0])
    if k <= 0 or q.shape[0] == 0:
        return {"overall": 0.0, "per_label": {}}

    topk = (q @ ix.T).topk(k, dim=-1).indices  # (Q, k)
    match = torch.from_numpy(
        np.asarray(query_labels)[:, None] == np.asarray(index_labels)[None, :]
    )
    hit = match[torch.arange(q.shape[0]).unsqueeze(1), topk]

    per_label: dict[str, float] = {}
    q_arr = np.asarray(query_labels)
    for label in sorted(set(query_labels), key=str):
        rows = torch.from_numpy(np.nonzero(q_arr == label)[0])
        per_label[str(label)] = float(hit[rows].float().mean())
    return {"overall": float(hit.float().mean()), "per_label": per_label}


def style_tag_agreement(
    embeddings: torch.Tensor,
    tag_vectors: torch.Tensor,
    *,
    ks: Sequence[int] = (1, 3, 5),
    center: bool = True,
) -> dict:
    """Fine-grained cross-family style similarity via style-tag labels.

    For each query font, ranks the pool by embedding cosine and reports the mean
    cosine between the query's style-tag vector and those of its top-k
    neighbours.  ``tag_cosine_all`` is the random-neighbour baseline, so a value
    above it means the embedding is grouping fonts that share fine-grained style
    tags rather than only coarse category/family identity.
    """
    x = prepare(embeddings, center=center)
    n = x.shape[0]
    if n < 2 or tag_vectors.shape[0] != n:
        return {f"tag_cosine@{k}": 0.0 for k in ks} | {"tag_cosine_all": 0.0}

    sim = x @ x.T
    sim.fill_diagonal_(float("-inf"))

    tags = F.normalize(tag_vectors.float().to(x.device), p=2, dim=-1, eps=1e-8)
    tag_sim = tags @ tags.T  # (N, N)

    out: dict = {}
    for k in ks:
        kk = min(k, n - 1)
        topk = sim.topk(kk, dim=-1).indices  # (N, kk)
        gathered = tag_sim[torch.arange(n, device=x.device).unsqueeze(1), topk]
        out[f"tag_cosine@{k}"] = float(gathered.mean())

    off = ~torch.eye(n, dtype=torch.bool, device=x.device)
    out["tag_cosine_all"] = float(tag_sim[off].mean())
    return out


def masked_pairwise_cosine(
    tag_vectors: torch.Tensor, tag_masks: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairwise cosine of sparse tag vectors over assessed dimensions only.

    Entry ``(i, j)`` is the cosine of ``tag_vectors[i]`` and ``tag_vectors[j]``
    restricted to dimensions where *both* masks are ``True`` (i.e. both fonts
    were assessed).  Absent theme tags are therefore treated as "unknown" rather
    than as a real zero — the correct semantics for sparse tags.

    Returns ``(sim, comparable)``: ``sim`` is ``(N, N)`` (0.0 on the diagonal
    and for incomparable/zero-norm pairs), ``comparable`` is the ``(N, N)``
    boolean mask of pairs with a non-empty assessed overlap.
    """
    n = tag_vectors.shape[0]
    V = tag_vectors.float()
    M = tag_masks.to(V.dtype)  # (N, T), 1.0 = assessed

    # Masked dot products and per-pair norms, vectorized over all pairs.
    Vm = V * M  # values masked by each row's own assessment
    num = Vm @ Vm.T  # (N, N): sum_t V_i V_j over assessed overlap
    sq = (V * V) * M
    norm2_i = sq @ M.T  # (N, N): ||V_i||^2 over overlap(i, j)
    norm2_j = norm2_i.T  # (N, N): ||V_j||^2 over overlap(i, j)
    denom = torch.sqrt(norm2_i * norm2_j).clamp(min=1e-12)
    sim = num / denom  # (N, N)

    comparable = (M @ M.T) > 0  # non-empty assessed overlap
    eye = torch.eye(n, device=tag_vectors.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, 0.0)
    comparable = comparable & ~eye

    return sim, comparable


def tag_space_agreement(
    embeddings: torch.Tensor,
    tag_vectors: torch.Tensor,
    tag_masks: torch.Tensor,
    *,
    center: bool = True,
) -> dict:
    """Agreement between tag-space and embedding-space organisation.

    Compares the pairwise-similarity structure implied by the *assessed* style
    tags against the model's embedding space.  ``pearson``/``spearman`` are the
    correlation of the two off-diagonal similarity vectors over the pairs with a
    non-empty assessed tag overlap; a high value means "fonts the tags say are
    similar are also close in the embedding".  ``comparable_fraction`` is the
    share of off-diagonal pairs with any assessed overlap — a coverage/sparsity
    diagnostic for the tag data.
    """
    n = embeddings.shape[0]
    if n < 3:
        return {
            "pearson": 0.0,
            "spearman": 0.0,
            "comparable_fraction": 0.0,
            "n": n,
        }

    tag_sim, comparable = masked_pairwise_cosine(tag_vectors, tag_masks)
    x = prepare(embeddings, center=center)
    emb_sim = x @ x.T

    tri = torch.triu_indices(n, n, offset=1, device=emb_sim.device)
    t = tag_sim[tri[0], tri[1]]
    e = emb_sim[tri[0], tri[1]]
    comp = comparable[tri[0], tri[1]]

    comparable_fraction = float(comp.float().mean()) if comp.numel() else 0.0

    t = t[comp]
    e = e[comp]
    if t.numel() < 2:
        pearson = spearman = 0.0
    else:
        pearson = _pearson(e, t)
        spearman = _pearson(_ranks(e), _ranks(t))

    return {
        "pearson": pearson,
        "spearman": spearman,
        "comparable_fraction": comparable_fraction,
        "n": n,
    }


def geometry_stats(embeddings: torch.Tensor) -> dict:
    """Diagnostics for the shape of the embedding space.

    Reports the pairwise-cosine distribution (raw and centered) and the
    participation ratio — an effective-dimensionality measure that drops toward
    1 when the space has collapsed onto a line.
    """
    x = prepare(embeddings, center=False)
    n = x.shape[0]
    off = ~torch.eye(n, dtype=torch.bool)
    raw = (x @ x.T)[off]
    centered = (prepare(embeddings, center=True) @ prepare(embeddings, center=True).T)[
        off
    ]

    c = embeddings.float() - embeddings.float().mean(dim=0, keepdim=True)
    s2 = torch.linalg.svdvals(c) ** 2
    participation = float(s2.sum() ** 2 / (s2**2).sum()) if s2.sum() > 0 else 0.0

    return {
        "n": n,
        "dim": int(embeddings.shape[1]),
        "mean_cosine_raw": float(raw.mean()),
        "std_cosine_raw": float(raw.std()),
        "mean_cosine_centered": float(centered.mean()),
        "std_cosine_centered": float(centered.std()),
        "participation_ratio": participation,
    }


def pairing_scores(
    base_embeddings: torch.Tensor,
    anchor_idx: Sequence[int],
    perturbed_embeddings: torch.Tensor,
    *,
    base_groups: Sequence[Hashable] | None = None,
    ks: Sequence[int] = (1, 3, 5),
    center: bool = True,
) -> dict:
    """Rank perturbed renders against the base pool.

    Each ``perturbed_embeddings[j]`` is a *different rendering of the font* at
    ``anchor_idx[j]`` (e.g. another axis position).  The correct target is the
    anchor row itself by default; when ``base_groups`` is given the target is
    any base row sharing the anchor's group, which is the fairer test when a
    family also ships static files (a synthetic max-weight render may genuinely
    sit closer to ``Family-Bold.ttf`` than to the variable default).

    ``discrimination_gap`` is the mean cosine to target rows minus the mean
    cosine to unrelated rows — invariance without collapse onto the anchor.
    """
    p = perturbed_embeddings.float()
    anchor = torch.as_tensor(list(anchor_idx), dtype=torch.long)
    rows = torch.arange(p.shape[0])
    mu = (
        base_embeddings.float().mean(dim=0, keepdim=True)
        if center
        else torch.zeros(1, base_embeddings.shape[1])
    )
    bn = F.normalize(base_embeddings.float() - mu, p=2, dim=-1, eps=1e-8)
    pn = F.normalize(p - mu, p=2, dim=-1, eps=1e-8)

    sim = pn @ bn.T  # (P, N)
    if base_groups is None:
        target = torch.zeros_like(sim, dtype=torch.bool)
        target[rows, anchor] = True
    else:
        target = _agreement_matrix(base_groups)[anchor]  # (P, N)

    rank, has_target = _ranks_from_sim(sim, target)
    ranks = rank[has_target].float()
    recall = (
        {k: float((ranks <= k).float().mean()) for k in ks}
        if ranks.numel()
        else {k: 0.0 for k in ks}
    )
    mrr = float((1.0 / ranks).mean()) if ranks.numel() else 0.0
    cos_unrelated = float(sim[~target].mean())
    cos_same = float(sim[target].mean())
    return {
        "recall": recall,
        "mrr": mrr,
        "n_pairs": int(sim.shape[0]),
        "cos_same_pair": cos_same,
        "cos_unrelated": cos_unrelated,
        "discrimination_gap": cos_same - cos_unrelated,
    }


def nearest_neighbours(
    embeddings: torch.Tensor,
    labels: Sequence[Hashable],
    query_indices: Sequence[int],
    *,
    k: int = 5,
    center: bool = True,
) -> list[list[tuple[int, str, float]]]:
    """Top-k ``(index, label, cosine)`` neighbours for each query row."""
    x = prepare(embeddings, center=center)
    sim = x @ x.T
    sim.fill_diagonal_(float("-inf"))
    k = min(k, x.shape[0] - 1)
    out: list[list[tuple[int, str, float]]] = []
    for qi in query_indices:
        vals, idx = sim[qi].topk(k)
        out.append([(int(j), str(labels[int(j)]), float(v)) for v, j in zip(vals, idx)])
    return out


def similarity_vector(embeddings: torch.Tensor, *, center: bool = True) -> torch.Tensor:
    """Flattened upper-triangle pairwise cosine similarities, one per pair."""
    x = prepare(embeddings, center=center)
    idx = torch.triu_indices(x.shape[0], x.shape[0], offset=1)
    return (x @ x.T)[idx[0], idx[1]]


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float() - a.float().mean()
    b = b.float() - b.float().mean()
    denom = (a.norm() * b.norm()).clamp(min=1e-12)
    return float((a @ b) / denom)


def _ranks(x: torch.Tensor) -> torch.Tensor:
    order = x.argsort()
    ranks = torch.empty(x.numel(), dtype=torch.float32)
    ranks[order] = torch.arange(x.numel(), dtype=torch.float32)
    return ranks


def signal_attribution(
    embedding: torch.Tensor,
    descriptors: dict[str, torch.Tensor],
    *,
    center: bool = True,
) -> dict:
    """Explain embedding similarity by independent reference descriptors.

    Correlates the embedding's pairwise-similarity vector with each
    descriptor's similarity vector, then fits a joint least-squares model so we
    can read off how much of the embedding's structure each reference accounts
    for.  This answers "is the embedding reading proportion or shape?" directly.
    """
    import numpy as np

    target = similarity_vector(embedding, center=center)
    refs = {
        name: similarity_vector(desc, center=center)
        for name, desc in descriptors.items()
    }
    out: dict = {}
    for name, ref in refs.items():
        out[name] = {
            "pearson": _pearson(target, ref),
            "spearman": _pearson(_ranks(target), _ranks(ref)),
        }

    names = list(refs)
    columns = [refs[n].numpy() for n in names]
    columns.append(np.ones(target.numel(), dtype=np.float32))
    X = np.stack(columns, axis=1)
    y = target.numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = float(((y - X @ beta) ** 2).sum())
    total = float(((y - y.mean()) ** 2).sum())
    out["joint"] = {
        "r2": 1.0 - residual / total if total > 0 else 0.0,
        **{f"beta_{n}": float(b) for n, b in zip(names, beta[:-1])},
    }
    return out


def membership_ranking(
    glyph_embeddings: torch.Tensor,
    font_embeddings: torch.Tensor,
    own_font: Sequence[int],
    *,
    ks: Sequence[int] = (1, 3, 5),
) -> dict:
    """Rank each glyph's own font among all candidate fonts.

    ``glyph_embeddings`` are per-glyph vectors, ``font_embeddings`` are the
    reference vectors of every candidate font, and ``own_font[i]`` is the index
    of ``glyph_embeddings[i]``'s font.  Both sides are centered by the font
    mean so they share an origin.
    """
    n = glyph_embeddings.shape[0]
    if n == 0 or font_embeddings.shape[0] < 2:
        return {"recall": {k: 0.0 for k in ks}, "mrr": 0.0, "auc": 0.0, "n": int(n)}

    mu = font_embeddings.float().mean(dim=0, keepdim=True)
    g = F.normalize(glyph_embeddings.float() - mu, p=2, dim=-1, eps=1e-8)
    f = F.normalize(font_embeddings.float() - mu, p=2, dim=-1, eps=1e-8)
    sim = g @ f.T  # (N, P)
    own = torch.as_tensor(list(own_font), dtype=torch.long)
    rows = torch.arange(n)
    correct = sim[rows, own]
    rank = (sim > correct.unsqueeze(1)).sum(dim=1) + 1
    others = sim.clone()
    others[rows, own] = float("-inf")
    return {
        "recall": {k: float((rank <= k).float().mean()) for k in ks},
        "mrr": float((1.0 / rank.float()).mean()),
        "auc": float((correct.unsqueeze(1) > others).float().mean()),
        "n": int(n),
    }
