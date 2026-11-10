"""Font-identity embedding feasibility probe.

Two modes:

* ``--mode scalar`` — embed the *scalar* combiner similarity (hand features) with
  classical MDS.  This tests "does the combiner's single-number similarity fold
  into a low-dim Euclidean space?".
* ``--mode features`` — embed the *raw per-font feature vectors* directly (the
  axis magnitudes, no scalar collapse): standardized numeric columns + one-hot
  categorical codes + boolean flags + tag centiles, per-column z-scored, then
  PCA.  This tests "how many dimensions do the raw axis magnitudes need?".

The eigenvalue spectrum is the read for both: the number of significant
dimensions, and (for the scalar kernel distance) how non-Euclidean it is.

Usage::

    python -m hrothgar.similarity.embedding --mode features
    python -m hrothgar.similarity.embedding --mode scalar --combiner softmax
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from hrothgar.cedars import load_cedars
from hrothgar.similarity.supervision import (
    JUDGEMENTS_PATH,
    SIMILARITIES_PATH,
    build_supervision,
    text_half_supervision,
)
from hrothgar.utils import device
from hrothgar.similarity.axis_combiner import (
    _BOOLEAN_AXES,
    _CATEGORICAL_AXES,
    _NUMERIC_AXES,
    _TAG_AXES,
    _categorical_codes,
    _numeric_raw,
    _pair_weight_matrices,
    _score_matrix,
    _tag_mats,
    _train_combiner,
    build_similarity_tensor,
)


def _pairwise_dist(X: np.ndarray) -> np.ndarray:
    sq = (X[:, None, :] - X[None, :, :]) ** 2
    return np.sqrt(sq.sum(axis=-1))


def nn_overlap(sim: np.ndarray, emb_dist: np.ndarray, k: int = 10) -> float:
    """Mean fraction of top-``k`` teacher neighbours preserved in the embedding."""
    n = sim.shape[0]
    order_sim = np.argsort(-sim, axis=1)  # highest similarity first
    order_emb = np.argsort(emb_dist, axis=1)  # smallest distance first
    fracs = []
    for i in range(n):
        s = set(order_sim[i, 1 : k + 1].tolist())
        e = set(order_emb[i, 1 : k + 1].tolist())
        fracs.append(len(s & e) / k)
    return float(np.mean(fracs))


def classical_mds(D2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Classical MDS (PCoA): double-center squared distances, eigen-decompose."""
    n = D2.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    B = (B + B.T) / 2
    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1]
    return evals[order], evecs[:, order]


def build_raw_features(sup, proportion_pcs: int | None = None) -> tuple[np.ndarray, dict[str, int]]:
    """Build a per-font raw-feature matrix ``(N, D)`` from the combiner's inputs.

    Concatenates: standardized numeric columns, one-hot categorical codes
    (missing = all zeros), boolean flags (true=1, false/missing=0), and tag
    centiles.  No scalar collapse — the axis magnitudes are kept.  If
    ``proportion_pcs`` is set, the two 26-dim proportion vectors are replaced by
    their top-``proportion_pcs`` principal components.
    """
    cedars = load_cedars()
    numeric = _numeric_raw(sup)
    cat_codes = _categorical_codes(sup, cedars)
    tag_mats = _tag_mats(sup)

    blocks: list[np.ndarray] = []
    block_sizes: dict[str, int] = {}

    n_numeric = 0
    prop_lower = prop_upper = None
    for name in sorted(_NUMERIC_AXES) + ["proportions"]:
        if name == "proportion-lower":
            prop_lower = numeric[name][0].numpy()
            continue
        if name == "proportion-upper":
            prop_upper = numeric[name][0].numpy()
            continue
        V, _ = numeric[name]
        blocks.append(V.numpy())
        n_numeric += V.shape[1]
    if proportion_pcs is not None and proportion_pcs > 0:
        blocks.append(compress_proportions(prop_lower, prop_upper, proportion_pcs))
        n_numeric += proportion_pcs
    else:
        blocks.append(prop_lower)
        blocks.append(prop_upper)
        n_numeric += prop_lower.shape[1] + prop_upper.shape[1]
    block_sizes["numeric"] = n_numeric

    n_cat = 0
    for name, _src, _col, vocab in _CATEGORICAL_AXES:
        codes = cat_codes[name].numpy()
        onehot = np.zeros((len(codes), len(vocab)), dtype=np.float64)
        mask = codes >= 0
        onehot[np.arange(len(codes))[mask], codes[mask]] = 1.0
        blocks.append(onehot)
        n_cat += len(vocab)
    block_sizes["categorical"] = n_cat

    n_bool = 0
    for name, _col in _BOOLEAN_AXES:
        codes = cat_codes[name].numpy()
        blocks.append((codes == 1).astype(np.float64).reshape(-1, 1))
        n_bool += 1
    block_sizes["boolean"] = n_bool

    n_tag = 0
    for name in _TAG_AXES:
        V, _ = tag_mats[name]
        blocks.append(V.numpy())
        n_tag += V.shape[1]
    block_sizes["tag"] = n_tag

    return np.concatenate(blocks, axis=1), block_sizes


def build_numeric_features(sup) -> tuple[np.ndarray, dict[str, int]]:
    """Build the numeric-only feature block ``(N, D)`` with per-axis column counts."""
    numeric = _numeric_raw(sup)
    blocks: list[np.ndarray] = []
    sizes: dict[str, int] = {}
    for name in sorted(_NUMERIC_AXES) + ["proportions"]:
        V, _ = numeric[name]
        blocks.append(V.numpy())
        sizes[name] = V.shape[1]
    return np.concatenate(blocks, axis=1), sizes


def build_proportion_features(sup) -> tuple[np.ndarray, np.ndarray]:
    """Build the two proportion-vector blocks (lowercase, uppercase)."""
    numeric = _numeric_raw(sup)
    lower, _ = numeric["proportion-lower"]
    upper, _ = numeric["proportion-upper"]
    return lower.numpy(), upper.numpy()


def compress_proportions(lower: np.ndarray, upper: np.ndarray, k: int) -> np.ndarray:
    """Compress the two proportion vectors to their top-``k`` principal components."""
    X = np.concatenate([lower, upper], axis=1).astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    X = X / std
    U, s, _Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :k] * s[:k]


def axis_blocks(sup) -> list[tuple[str, np.ndarray]]:
    """Return the per-axis blocks ``(name, (N, d) array)`` in combiner order."""
    cedars = load_cedars()
    numeric = _numeric_raw(sup)
    cat_codes = _categorical_codes(sup, cedars)
    tag_mats = _tag_mats(sup)

    blocks: list[tuple[str, np.ndarray]] = []
    for name in sorted(_NUMERIC_AXES) + ["proportions"]:
        V, _ = numeric[name]
        blocks.append((name, V.numpy()))
    for name, _src, _col, vocab in _CATEGORICAL_AXES:
        codes = cat_codes[name].numpy()
        onehot = np.zeros((len(codes), len(vocab)), dtype=np.float64)
        mask = codes >= 0
        onehot[np.arange(len(codes))[mask], codes[mask]] = 1.0
        blocks.append((name, onehot))
    for name, _col in _BOOLEAN_AXES:
        codes = cat_codes[name].numpy()
        blocks.append((name, (codes == 1).astype(np.float64).reshape(-1, 1)))
    for name in _TAG_AXES:
        V, _ = tag_mats[name]
        blocks.append((name, V.numpy()))
    return blocks


def weighted_features(sup, w: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[tuple[str, int, int]], list[np.ndarray], list[np.ndarray]]:
    """Axis-weighted raw features: z-score each axis block, scale by its weight.

    Returns ``(X, ranges, block_means, block_stds)`` where ``X`` is ``(N, D)``,
    ``ranges`` is ``[(axis_name, col_start, col_end)]`` for interpreting the PCA
    loadings, and ``block_means``/``block_stds`` are the per-block z-score stats
    (one array per axis block) needed to project an unseen font onto the same
    basis.
    """
    blocks = axis_blocks(sup)
    out: list[np.ndarray] = []
    ranges: list[tuple[str, int, int]] = []
    block_means: list[np.ndarray] = []
    block_stds: list[np.ndarray] = []
    offset = 0
    for name, block in blocks:
        b = block.astype(np.float64)
        bmean = b.mean(axis=0)
        b = b - bmean
        bstd = b.std(axis=0)
        bstd[bstd < 1e-8] = 1.0
        b = b / bstd
        block_means.append(bmean)
        block_stds.append(bstd)
        w_k = float(w[names.index(name)])
        out.append(b * w_k)
        ranges.append((name, offset, offset + b.shape[1]))
        offset += b.shape[1]
    return np.concatenate(out, axis=1), ranges, block_means, block_stds


def pca_summary(X: np.ndarray, nn_k: int = 10, scale: bool = True) -> dict:
    """Center (and optionally per-column z-score), then PCA (SVD).

    ``scale=False`` is used when the caller has already applied axis-level
    weighting, so re-z-scoring would undo it.
    """
    X = X.astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    if scale:
        std = X.std(axis=0)
        std[std < 1e-8] = 1.0
        X = X / std
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    evals = s ** 2
    ratio = evals / evals.sum()
    full_dist = _pairwise_dist(X)
    nn = {}
    for k in (5, 10, 20, 30):
        if k <= X.shape[0]:
            coords_k = U[:, :k] * s[:k]
            nn[k] = nn_overlap(-full_dist, _pairwise_dist(coords_k), k=nn_k)
    return {"evals": evals, "ratio": ratio, "nn": nn, "n": X.shape[0], "d": X.shape[1], "Vt": Vt}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("features", "scalar", "numeric", "weights", "proportion", "teacher", "umap"), default="features")
    p.add_argument("--weights-out", default="similarity_axis_weights.txt", help="path for --mode weights")
    p.add_argument("--repo", default=os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts"))
    p.add_argument("--list", default=SIMILARITIES_PATH)
    p.add_argument("--judgements", default=JUDGEMENTS_PATH)
    p.add_argument("--combiner", choices=("linear", "softmax"), default="softmax")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--margin", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=128.0)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--nn-k", type=int, default=10, help="neighbourhood size for the overlap check")
    p.add_argument("--proportion-pcs", type=int, default=0, help="compress proportion vectors to this many PCs (0 = none; --mode features)")
    p.add_argument("--teacher-out", default="models/text_teacher.npz", help="path for the frozen teacher (--mode teacher)")
    args = p.parse_args()

    dev = device(args.device)
    sup = build_supervision(args.repo, args.list, args.judgements, universe="full")
    sup = text_half_supervision(sup)
    print(f"\ntext-half universe: {len(sup.fonts)} fonts  "
          f"pos={len(sup.pos_pairs)} neg={len(sup.neg_pairs)}  mode={args.mode}")

    if args.mode == "features":
        X, block_sizes = build_raw_features(sup, proportion_pcs=args.proportion_pcs or None)
        r = pca_summary(X, nn_k=args.nn_k)
        print(f"raw features: {r['d']} dims  {block_sizes}")
        print(f"\n  k   eigenvalue   cum. explained")
        for k in (1, 2, 3, 5, 10, 15, 20, 30, 50):
            if k <= r["d"]:
                cum = float(r["evals"][:k].sum() / r["evals"].sum())
                print(f"  {k:>2}   {r['evals'][k-1]:10.4f}   {100*cum:6.2f}%")
        print(f"\n  k-dim   top-{args.nn_k} NN overlap (embedding vs full-dim)")
        for k in (5, 10, 20, 30):
            print(f"  {k:>5}   {r['nn'][k]:.3f}")
        return 0

    if args.mode == "numeric":
        X, sizes = build_numeric_features(sup)
        r = pca_summary(X, nn_k=args.nn_k)
        print(f"numeric block: {r['d']} dims")
        print("  per-axis columns:", sizes)
        print(f"\n  k   eigenvalue   cum. explained")
        for k in (1, 2, 3, 5, 10, 15, 20, 30, 50):
            if k <= r["d"]:
                cum = float(r["evals"][:k].sum() / r["evals"].sum())
                print(f"  {k:>2}   {r['evals'][k-1]:10.4f}   {100*cum:6.2f}%")
        print(f"\n  k-dim   top-{args.nn_k} NN overlap (embedding vs full-dim)")
        for k in (5, 10, 20, 30):
            print(f"  {k:>5}   {r['nn'][k]:.3f}")
        return 0

    if args.mode == "weights":
        S, names = build_similarity_tensor(sup, feature_source="hand")
        S = S.to(dev)
        pos, neg = _pair_weight_matrices(sup, dev)
        kw = dict(steps=args.steps, lr=args.lr, margin=args.margin, gamma=args.gamma,
                  seed=args.seed, weight_decay=args.weight_decay)
        w_lin, _ = _train_combiner(S, pos, neg, combiner="linear", **kw)
        w_soft, _ = _train_combiner(S, pos, neg, combiner="softmax", **kw)
        w_lin = w_lin.detach().cpu().numpy()
        w_soft = w_soft.detach().cpu().numpy()
        order = np.argsort(-np.abs(w_soft))
        lines = [
            "# combiner axis weights — text half, hand features",
            f"# universe={len(sup.fonts)} fonts  pos={len(sup.pos_pairs)} neg={len(sup.neg_pairs)}",
            "# sorted by |softmax| weight",
            "",
            f"{'rank':>4}  {'axis':<20} {'linear':>10} {'softmax':>10} {'|softmax|':>10}",
        ]
        for rank, i in enumerate(order):
            lines.append(f"{rank + 1:>4}  {names[i]:<20} {w_lin[i]:>+10.4f} {w_soft[i]:>+10.4f} {abs(w_soft[i]):>10.4f}")
        text = "\n".join(lines) + "\n"
        Path(args.weights_out).write_text(text)
        print(text)
        return 0

    if args.mode == "proportion":
        lower, upper = build_proportion_features(sup)
        for label, X in (
            ("proportion-lower (26)", lower),
            ("proportion-upper (26)", upper),
            ("both (52)", np.concatenate([lower, upper], axis=1)),
        ):
            r = pca_summary(X, nn_k=args.nn_k)
            print(f"\n{label}: {r['d']} dims")
            print("  k   cum. explained   NN@10 overlap")
            for k in (1, 2, 3, 5, 10, 15, 20):
                if k <= r["d"]:
                    cum = float(r["evals"][:k].sum() / r["evals"].sum())
                    nn = r["nn"].get(k)
                    nn_s = f"{nn:.3f}" if nn is not None else "    -"
                    print(f"  {k:>2}   {100*cum:6.2f}%        {nn_s}")
        return 0

    if args.mode == "teacher":
        S, names = build_similarity_tensor(sup, feature_source="hand")
        S = S.to(dev)
        pos, neg = _pair_weight_matrices(sup, dev)
        w, _ = _train_combiner(
            S, pos, neg, combiner=args.combiner, steps=args.steps, lr=args.lr,
            margin=args.margin, gamma=args.gamma, seed=args.seed,
            weight_decay=args.weight_decay,
        )
        w = w.detach().cpu().numpy()
        X, ranges, block_means, block_stds = weighted_features(sup, w, names)
        mean = X.mean(axis=0)
        r = pca_summary(X, nn_k=args.nn_k, scale=False)
        print(f"axis-weighted features: {r['d']} dims, {len(ranges)} axes, combiner={args.combiner}")
        print(f"\n  k   eigenvalue   cum. explained")
        for k in (1, 2, 3, 5, 10, 15, 20, 30):
            if k <= r["d"]:
                cum = float(r["evals"][:k].sum() / r["evals"].sum())
                print(f"  {k:>2}   {r['evals'][k-1]:10.4f}   {100*cum:6.2f}%")
        print(f"\n  k-dim   top-{args.nn_k} NN overlap (embedding vs weighted full-dim)")
        for k in (5, 10, 20, 30):
            print(f"  {k:>5}   {r['nn'][k]:.3f}")
        print("\n  top modes (dominant axes by |loading|):")
        Vt = r["Vt"]
        for j in range(min(5, Vt.shape[0])):
            load = Vt[j]
            axis_loads = [(name, float(np.abs(load[lo:hi]).sum())) for name, lo, hi in ranges]
            top = sorted(axis_loads, key=lambda x: -x[1])[:5]
            print(f"    mode {j+1}: " + ", ".join(f"{n}({v:.2f})" for n, v in top))

        # ── Freeze the teacher: coords + the projection basis ─────────────────
        coords = (X - mean) @ Vt[:30].T
        np.savez(
            args.teacher_out,
            coords=coords.astype(np.float32),
            families=np.array(sup.families, dtype=object),
            components=Vt[:30].astype(np.float32),
            mean=mean.astype(np.float32),
            axis_weights=w.astype(np.float32),
            axis_names=np.array(names, dtype=object),
            block_mean=np.concatenate(block_means).astype(np.float32),
            block_std=np.concatenate(block_stds).astype(np.float32),
        )
        print(f"\nsaved frozen teacher -> {args.teacher_out}  coords={coords.shape}")
        return 0

    if args.mode == "umap":
        import umap as umap_mod

        S, names = build_similarity_tensor(sup, feature_source="hand")
        S = S.to(dev)
        pos, neg = _pair_weight_matrices(sup, dev)
        w, _ = _train_combiner(
            S, pos, neg, combiner=args.combiner, steps=args.steps, lr=args.lr,
            margin=args.margin, gamma=args.gamma, seed=args.seed,
            weight_decay=args.weight_decay,
        )
        w = w.detach().cpu().numpy()
        X, _ranges, _bm, _bs = weighted_features(sup, w, names)
        Xc = X - X.mean(axis=0, keepdims=True)
        full_dist = _pairwise_dist(Xc)
        print(f"UMAP on axis-weighted features ({X.shape[1]} dims), combiner={args.combiner}")
        print(f"  (compare PCA: k=5 {0.301:.3f}, k=10 {0.427:.3f}, k=30 {0.704:.3f})")
        print(f"\n  k-dim   top-{args.nn_k} NN overlap (vs weighted full-dim)")
        for d in (2, 5, 10, 30):
            reducer = umap_mod.UMAP(n_components=d, random_state=args.seed)
            emb = reducer.fit_transform(Xc)
            ov = nn_overlap(-full_dist, _pairwise_dist(emb), k=args.nn_k)
            print(f"  {d:>5}   {ov:.3f}")
        return 0

    # ── scalar mode: embed the combiner's scalar similarity ────────────────────
    S, names = build_similarity_tensor(sup, feature_source="hand")
    S = S.to(dev)
    pos, neg = _pair_weight_matrices(sup, dev)
    w, b = _train_combiner(
        S, pos, neg, combiner=args.combiner, steps=args.steps, lr=args.lr,
        margin=args.margin, gamma=args.gamma, seed=args.seed,
        weight_decay=args.weight_decay,
    )
    sim = _score_matrix(S, w, b).cpu().numpy()
    sim = (sim + sim.T) / 2

    d_self = np.diag(sim).copy()
    D2 = d_self[:, None] + d_self[None, :] - 2.0 * sim
    n_neg_d2 = int((D2 < -1e-9).sum())
    D2 = np.maximum(D2, 0.0)
    evals, evecs = classical_mds(D2)
    tol = 1e-9
    total_pos = float(evals[evals > tol].sum())
    total_neg = float((-evals[evals < -tol]).sum())
    print(f"scalar teacher: negative kernel d² = {n_neg_d2} pairs; "
          f"|neg|/pos = {total_neg/total_pos:.3f}")
    print("\n  k   eigenvalue   cum. explained (of positive)")
    for k in (1, 2, 3, 5, 10, 15, 20, 30):
        if k <= int((evals > tol).sum()):
            cum = float(evals[:k].clip(min=0).sum()) / total_pos
            print(f"  {k:>2}   {evals[k-1]:10.4f}   {100*cum:6.2f}%")
    print(f"\n  k-dim   top-{args.nn_k} NN overlap (embedding vs teacher)")
    for k in (5, 10, 20, 30):
        coords_k = evecs[:, :k] * np.sqrt(np.maximum(evals[:k], 0.0))[None, :]
        ov = nn_overlap(sim, _pairwise_dist(coords_k), k=args.nn_k)
        print(f"  {k:>5}   {ov:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
