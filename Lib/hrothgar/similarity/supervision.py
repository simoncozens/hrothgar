"""Build graded contrastive supervision from similarities.json + judgements.csv.

The agreed scheme is two per-pair channels:

* **positive weight** ``p(i,j)`` — pull together:
  * L1 same-font two views  (handled by the caller at the *view* level)
  * L2 same family, different font  → ``W_FAMILY`` (weak; weight-matching can trump)
  * L3 leaf-tag cosine  → ``W_TAG * cos`` (above a threshold, assessed dims only)
  * L4 human positive (similarities include / game positive) → agreement-mapped
* **negative weight** ``n(i,j)`` — push apart (named negatives only):
  * L4 human negative (similarities exclude / game negative / ``none_alike``)
    → agreement-mapped

Unassigned pairs stay neutral: they are *not* assumed to be negatives.  That is
the deliberate difference from the current similarity objective, which treats
"everyone not sharing a tag" as a negative and thereby re-creates coarse-tag
clustering instead of fine style.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

from hrothgar.googlefonts import GoogleFonts, compute_display_score, regular_font
from hrothgar.similarity.dataset import style_tag_mask, style_tag_vector
from hrothgar.similarity.metrics import masked_pairwise_cosine

#: Conventional locations of the supervision data, relative to the repo root.
SUPERVISION_DIR = "data/similarity_supervision"
SIMILARITIES_PATH = f"{SUPERVISION_DIR}/similarities.json"
JUDGEMENTS_PATH = f"{SUPERVISION_DIR}/judgements.csv"

# ── Supervision weights (policy constants, not flags) ────────────────────────
W_SAME_VIEW = 1.0     # L1: the two views of the same font (handled at view level)
W_FAMILY = 0.1        # L2: same family, different font (reviewer: keep weak)
W_TAG = 0.3           # L3: leaf-tag soft positive, scaled by tag cosine
TAG_THRESHOLD = 0.1   # L3: minimum tag cosine to count as a positive
W_HUMAN_SIM = 0.8     # L4: similarities.json include/exclude (one soft curator)
HUMAN_FLOOR = 0.5     # L4: agreement → weight floor (single expert > 0)


@dataclass
class Supervision:
    """Resolved universe + cross-font positive/negative pair lists."""

    fonts: list
    families: list[str]
    font_ids: list[str]
    tag_vec: torch.Tensor          # (N, T) leaf-tag centiles / 100
    tag_mask: torch.Tensor         # (N, T) bool (True = assessed)
    pos_pairs: list[tuple[int, int, float]]  # (i, j, weight) i != j
    neg_pairs: list[tuple[int, int, float]]  # (i, j, weight) i != j
    clusters: list[dict]           # {"include": [idx], "exclude": [idx]}


def _load_clusters(path: str) -> list[dict]:
    try:
        return json.loads(Path(path).read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"malformed JSON in {path}: {e}") from e


def _agreement_weight(count: float, shown: float) -> float:
    """Map expert agreement (picks/shown) to a soft weight in [floor, 1]."""
    agree = count / shown if shown > 0 else 0.0
    return HUMAN_FLOOR + (1.0 - HUMAN_FLOOR) * agree


def build_supervision(
    repo: str,
    sim_path: str,
    judge_path: str,
    *,
    pos_weight: float = 1.0,
    neg_weight: float = 1.0,
    universe: str = "supervised",
) -> Supervision:
    """Resolve the supervision files into one indexed universe.

    Both files name fonts by family.  ``universe`` controls which families enter
    the universe:

    * ``"supervised"`` (default) — only families that actually participate in
      supervision (similarities include/exclude, judgement anchor, a picked
      positive/negative, or a ``none_alike`` candidate); shown-but-unpicked
      candidates are neutral and left out.  This is the *training-eval* universe.
    * ``"full"`` — every non-skipped family in the library (its regular
      instance), so scoring can rank an anchor against the whole library.  The
      supervised pairs are remapped onto this larger index; families that never
      appear in supervision simply have no pairs and stay neutral.

    Every family is resolved to its regular instance.
    """
    gf = GoogleFonts(repo)
    by_family: dict[str, list] = {}
    for font in gf.fonts:
        by_family.setdefault(font.family.lower(), []).append(font)

    clusters = _load_clusters(sim_path)
    judge_rows = list(csv.DictReader(open(judge_path)))

    def _key(name: str) -> str:
        return name.lower().strip()

    # ── Collect supervised families + name-level agreement counts ─────────────
    supervised: set[str] = set()
    pos_n: dict[tuple[str, str], float] = defaultdict(float)
    neg_n: dict[tuple[str, str], float] = defaultdict(float)
    shown_n: dict[tuple[str, str], float] = defaultdict(float)

    for cluster in clusters:
        for name in cluster["include"] + cluster["exclude"]:
            supervised.add(_key(name))

    for row in judge_rows:
        if row["skipped"] == "true" or row["no_clear_answer"] == "true":
            continue
        ak = _key(json.loads(row["anchor"])["family"])
        cands = [c["family"] for c in json.loads(row["candidates"])]
        for cand in cands:
            shown_n[(ak, _key(cand))] += 1

        if row["none_alike"] == "true":
            supervised.add(ak)
            for cand in cands:
                ck = _key(cand)
                supervised.add(ck)
                neg_n[(ak, ck)] += 1
            continue

        supervised.add(ak)
        for p in json.loads(row["positive"]):
            pk = _key(p)
            supervised.add(pk)
            pos_n[(ak, pk)] += 1
        if row["negative"]:
            nk = _key(row["negative"])
            supervised.add(nk)
            neg_n[(ak, nk)] += 1

    # ── Resolve the universe (supervised subset or the full library) ─────────
    if universe == "full":
        universe_keys = sorted(by_family)
    elif universe == "supervised":
        universe_keys = sorted(supervised)
    else:
        raise ValueError(f"unknown universe {universe!r} (expected 'supervised' or 'full')")

    name_to_font: dict[str, object] = {}
    for key in universe_keys:
        if key in by_family:
            name_to_font[key] = regular_font(by_family[key])

    fonts = [name_to_font[k] for k in sorted(name_to_font)]
    index = {f.family.lower(): i for i, f in enumerate(fonts)}

    families = [f.family for f in fonts]
    font_ids = [str(f.path) for f in fonts]
    tag_vec = torch.stack([style_tag_vector(f) for f in fonts])
    tag_mask = torch.stack([style_tag_mask(f) for f in fonts])

    # ── Similarities clusters → positive/negative pairs + leave-out groups ───
    pos: dict[tuple[int, int], float] = {}
    neg: dict[tuple[int, int], float] = {}
    resolved_clusters: list[dict] = []

    for cluster in clusters:
        inc = [index[n.lower()] for n in cluster["include"] if n.lower() in index]
        exc = [index[n.lower()] for n in cluster["exclude"] if n.lower() in index]
        resolved_clusters.append({"include": inc, "exclude": exc})
        for a in inc:
            for b in inc:
                if a != b:
                    pos[(a, b)] = max(pos.get((a, b), 0.0), W_HUMAN_SIM * pos_weight)
            for b in exc:
                neg[(a, b)] = max(neg.get((a, b), 0.0), W_HUMAN_SIM * neg_weight)

    # ── Judgements → anchor-relative positive/negative pairs (agreement) ──────
    for (ak, pk), c in pos_n.items():
        a, b = index.get(ak), index.get(pk)
        if a is not None and b is not None:
            pos[(a, b)] = max(pos.get((a, b), 0.0), _agreement_weight(c, shown_n[(ak, pk)]) * pos_weight)
    for (ak, nk), c in neg_n.items():
        a, b = index.get(ak), index.get(nk)
        if a is not None and b is not None:
            neg[(a, b)] = max(neg.get((a, b), 0.0), _agreement_weight(c, shown_n[(ak, nk)]) * neg_weight)

    # ── L2: same family, different font (weak intrinsic-style attraction) ────
    # A no-op at family-level resolution (one font per family), kept so that an
    # instance-level universe gets it for free.
    for a in range(len(fonts)):
        for b in range(a + 1, len(fonts)):
            if families[a].lower() == families[b].lower() and font_ids[a] != font_ids[b]:
                pos[(a, b)] = max(pos.get((a, b), 0.0), W_FAMILY)
                pos[(b, a)] = max(pos.get((b, a), 0.0), W_FAMILY)

    # ── Positive wins any conflict (soft-label "any similar → pull together") ─
    for k in list(neg):
        if k in pos or (k[1], k[0]) in pos:
            del neg[k]

    pos_pairs = [(a, b, w) for (a, b), w in pos.items()]
    neg_pairs = [(a, b, w) for (a, b), w in neg.items()]

    return Supervision(
        fonts=fonts,
        families=families,
        font_ids=font_ids,
        tag_vec=tag_vec,
        tag_mask=tag_mask,
        pos_pairs=pos_pairs,
        neg_pairs=neg_pairs,
        clusters=resolved_clusters,
    )


def display_scores(sup: Supervision) -> torch.Tensor:
    """Per-font display score (0-100) from tag composition (the teacher)."""
    return torch.tensor(
        [compute_display_score(f.tags()) for f in sup.fonts], dtype=torch.float32
    )


def filter_supervision(sup: Supervision, keep: list[int]) -> Supervision:
    """Restrict ``sup`` to the given font indices, remapping pairs and clusters.

    Clusters that no longer have at least two includes and one exclude are
    dropped (they cannot contribute a triplet/query to the eval).
    """
    keep = sorted(set(keep))
    remap = {old: new for new, old in enumerate(keep)}
    keep_set = set(keep)
    idx = torch.tensor(keep, dtype=torch.long)

    clusters: list[dict] = []
    for c in sup.clusters:
        inc = [remap[i] for i in c["include"] if i in keep_set]
        exc = [remap[i] for i in c["exclude"] if i in keep_set]
        if len(inc) >= 2 and len(exc) >= 1:
            clusters.append({"include": inc, "exclude": exc})

    return Supervision(
        fonts=[sup.fonts[i] for i in keep],
        families=[sup.families[i] for i in keep],
        font_ids=[sup.font_ids[i] for i in keep],
        tag_vec=sup.tag_vec[idx],
        tag_mask=sup.tag_mask[idx],
        pos_pairs=[(remap[a], remap[b], w) for a, b, w in sup.pos_pairs if a in keep_set and b in keep_set],
        neg_pairs=[(remap[a], remap[b], w) for a, b, w in sup.neg_pairs if a in keep_set and b in keep_set],
        clusters=clusters,
    )


def text_half_supervision(sup: Supervision, frac: float = 0.5) -> Supervision:
    """Keep the bottom ``frac`` of fonts by display score (the text-like arm)."""
    scores = display_scores(sup)
    threshold = scores.quantile(frac)
    keep = (scores <= threshold).nonzero(as_tuple=True)[0].tolist()
    return filter_supervision(sup, keep)


def display_half_supervision(sup: Supervision, frac: float = 0.5) -> Supervision:
    """Keep the top ``frac`` of fonts by display score (the display/vibe arm).

    The complement of :func:`text_half_supervision`: the ``frac``-most display
    fonts.  The median font is in both halves (it is genuinely "semi-display"),
    which is fine — the eventual blend weights the two arms by display score.
    """
    scores = display_scores(sup)
    threshold = scores.quantile(1.0 - frac)
    keep = (scores >= threshold).nonzero(as_tuple=True)[0].tolist()
    return filter_supervision(sup, keep)


def tag_pos_matrix(
    tag_vec: torch.Tensor,
    tag_mask: torch.Tensor,
    idxs: list[int],
    threshold: float = TAG_THRESHOLD,
    weight: float = W_TAG,
) -> torch.Tensor:
    """Return the L3 tag-positive weight matrix ``(k, k)`` over ``idxs``."""
    sub_vec = tag_vec[torch.tensor(idxs, dtype=torch.long)]
    sub_mask = tag_mask[torch.tensor(idxs, dtype=torch.long)]
    sim, comparable = masked_pairwise_cosine(sub_vec, sub_mask)
    keep = (sim > threshold) & comparable
    return (weight * sim).masked_fill(~keep, 0.0)
