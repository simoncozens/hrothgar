"""Evaluation and diagnostics for the content detector.

The question is open-set: for each anchor, can the model separate glyphs that
*are* that codepoint from glyphs that merely look like it (confusables,
diacritic variants), can it stay quiet on glyphs that are none of the known
codepoints, and does either answer depend on the style stratum?

Three things this module is careful about, because they are easy to get wrong:

* **The panel is stratified by style** (sans / serif / fancy / mono).  Taking the
  first N held-out fonts in family-name order samples an arbitrary slice of
  styles, which misrepresents recall.
* **Off-sample glyphs are split into easy and hard.**  A random letter rejects
  trivially and saturates its metric; the informative number is the reject rate
  on *hard* off-sample glyphs.
* **Recall is reported at a target FAR, not at 0.5.**  The learned per-anchor
  bias tracks the (shared) training prevalence rather than acting as a per-anchor
  operating point, so a single 0.5 threshold is a different tradeoff for every
  anchor — recall@0.5 is not comparable across anchors.  ``recall_at_far`` picks
  each anchor's threshold from its own held-out hard negatives, which is.

The score geometry (distribution of ``cos(positive, own anchor)`` and the
anchor-bank pairwise cosines) is reported alongside, to separate a pure
threshold/offset problem from genuine per-class difficulty.

``main()`` runs the whole breakdown against a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass

import torch

from hrothgar.content_detector.config import ContentDetectorConfig
from hrothgar.content_detector.dataset import (
    content_frame,
    hard_negative_pool,
    is_shape_identical,
)
from hrothgar.dataset import style_bucket
from hrothgar.render import render

REJECT_THRESHOLD = 0.5

# Panel composition, per held-out font.
EASY_OFFSAMPLE_PER_FONT = 3
HARD_OFFSAMPLE_PER_FONT = 3

# Target false-accept rate on held-out hard negatives for ``recall_at_far``.
DEFAULT_TARGET_FAR = 0.05


@dataclass
class ContentPanel:
    # ``uint8`` (ink 0, white 255): the panel is held for the whole training run,
    # and at 256px a float32 copy would be ~600MB.  ``batch()`` converts one
    # chunk at a time.
    images: torch.Tensor  # (N, 1, H, W) uint8
    codepoints: list[int]
    families: list[str]
    strata: list[str]  # style bucket of the source font
    buckets: list[str]  # "anchor" | "off_hard" | "off_easy"
    paths: list[str]  # source font path, so a top-scoring negative can be inspected

    def batch(self, start: int, stop: int, device) -> torch.Tensor:
        """Float ``[0, 1]`` images for ``[start, stop)``, on ``device``."""
        return (
            self.images[start:stop]
            .to(device=device, dtype=torch.float32)
            .div_(255.0)
        )


# Chunks of images scored in one forward pass.  The panel is a few thousand
# glyphs; pushing it through the encoder in one go puts the whole activation
# stack on the device at once and OOMs at 256px.  Chunking bounds the peak and
# leaves the result unchanged.
INFERENCE_BATCH = 32


def score_panel(
    model,
    panel: ContentPanel,
    device,
    *,
    batch_size: int = INFERENCE_BATCH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(embeddings, logits)`` for the panel, chunked, on the CPU.

    The outputs stay on the CPU: neither is needed on the device afterwards, and
    keeping them there is what lets a large panel be scored on a small GPU.
    """
    anchors = model.normalized_anchors().detach()
    scale = model.logit_scale.detach()
    bias = model.anchor_bias.detach()
    embeddings: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(panel.images), batch_size):
            z = model.embed(panel.batch(start, start + batch_size, device))
            embeddings.append(z.cpu())
            logits.append((scale * (z @ anchors.T) + bias).cpu())
    if not embeddings:
        return (
            torch.empty((0, model.config.embedding_dim)),
            torch.empty((0, len(model.anchors))),
        )
    return torch.cat(embeddings, dim=0), torch.cat(logits, dim=0)


def off_sample_buckets(config: ContentDetectorConfig) -> tuple[list[int], list[int]]:
    """Split the off-anchor pool into (hard, easy) off-sample codepoints.

    *Hard* off-sample glyphs are the anchors' confusables and diacritic variants
    (minus the anchors themselves) — the ones a broken detector would accept.
    *Easy* off-sample glyphs are everything else in the pool.
    """
    anchors = set(config.anchors)
    hard: set[int] = set()
    for anchor in config.anchors:
        hard |= set(hard_negative_pool(anchor, config.negatives))
    hard -= anchors
    easy = sorted(
        cp for cp in config.negatives if cp not in anchors and cp not in hard
    )
    return sorted(hard), easy


def _stratified_fonts(fonts, max_fonts: int, rng: random.Random) -> list:
    """Take a style-balanced sample of ``max_fonts`` fonts (round-robin)."""
    by_stratum: dict[str, list] = defaultdict(list)
    for font in fonts:
        by_stratum[style_bucket(font)].append(font)
    strata = sorted(by_stratum)
    for stratum in strata:
        rng.shuffle(by_stratum[stratum])

    selected: list = []
    i = 0
    while len(selected) < max_fonts and any(by_stratum[s] for s in strata):
        stratum = strata[i % len(strata)]
        if by_stratum[stratum]:
            selected.append(by_stratum[stratum].pop(0))
        i += 1
    return selected


def build_content_panel(
    fonts,
    config: ContentDetectorConfig,
    *,
    max_fonts: int = 100,
    seed: int = 1234,
    easy_per_font: int = EASY_OFFSAMPLE_PER_FONT,
    hard_per_font: int = HARD_OFFSAMPLE_PER_FONT,
) -> ContentPanel:
    """Render a style-stratified panel of positives, hard negatives and off-sample."""
    rng = random.Random(seed)
    pool = list(config.negatives)
    anchor_set = set(config.anchors)
    hard_set, easy_pool = off_sample_buckets(config)

    images: list[torch.Tensor] = []
    codepoints: list[int] = []
    families: list[str] = []
    strata: list[str] = []
    buckets: list[str] = []
    paths: list[str] = []
    frame = content_frame(config)

    for font in _stratified_fonts(fonts, max_fonts, rng):
        stratum = style_bucket(font)
        available = font.codepoints

        # Render each anchor once: the raster is both the positive entry and the
        # reference for the shape-identity filter below.
        anchor_images = {
            anchor: render(font, codepoint=anchor, size=config.glyph_size, frame=frame)[0]
            for anchor in config.anchors
        }

        def shaped_like_an_anchor(image: torch.Tensor) -> bool:
            """Whether a glyph shares an outline with any anchor."""
            return any(
                is_shape_identical(anchor_images[a], image) for a in config.anchors
            )

        # Positives, plus each anchor's curated hard negatives.
        per_font: list[tuple[int, str]] = []
        for anchor in config.anchors:
            per_font.append((anchor, "anchor"))
            for cp in hard_negative_pool(anchor, pool):
                if cp == anchor or cp not in available:
                    continue
                image, _ = render(font, codepoint=cp, size=config.glyph_size, frame=frame)
                if is_shape_identical(anchor_images[anchor], image):
                    continue  # the same outline under a different codepoint
                per_font.append((cp, "anchor" if cp in anchor_set else "off_hard"))

        # Independent draw from the whole hard / easy off-sample pools.  A glyph
        # that shares an anchor's outline is not off-sample evidence either — it
        # is shaped exactly like a positive, so it can never be rejected.
        hard_avail = [cp for cp in hard_set if cp in available]
        easy_avail = [cp for cp in easy_pool if cp in available]
        draws = [
            (cp, "off_hard")
            for cp in rng.sample(hard_avail, min(hard_per_font, len(hard_avail)))
        ] + [
            (cp, "off_easy")
            for cp in rng.sample(easy_avail, min(easy_per_font, len(easy_avail)))
        ]
        for cp, bucket in draws:
            image, _ = render(font, codepoint=cp, size=config.glyph_size, frame=frame)
            if shaped_like_an_anchor(image):
                continue
            per_font.append((cp, bucket))

        seen: set[int] = set()
        for cp, bucket in per_font:
            if cp in seen:
                continue
            seen.add(cp)
            image = anchor_images.get(cp)
            if image is None:
                image, _ = render(
                    font, codepoint=cp, size=config.glyph_size, frame=frame
                )
            images.append(
                (image * 255.0).round().clamp(0.0, 255.0).to(torch.uint8).unsqueeze(0)
            )
            codepoints.append(cp)
            families.append(font.family)
            strata.append(stratum)
            buckets.append(bucket)
            paths.append(str(font.path))

    return ContentPanel(torch.stack(images), codepoints, families, strata, buckets, paths)


# --------------------------------------------------------------------------- #
# Small statistics helpers
# --------------------------------------------------------------------------- #


def _auc(positive: list[float], negative: list[float]) -> float | None:
    """AUC with tie credit of 0.5; ``None`` when either side is empty."""
    if not positive or not negative:
        return None
    p = torch.tensor(positive, dtype=torch.float64)[:, None]
    n = torch.tensor(negative, dtype=torch.float64)[None, :]
    return float(((p > n).double() + 0.5 * (p == n).double()).mean())


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def letter_similarity(
    embeddings: torch.Tensor, groups: dict[int, list[int]]
) -> tuple[float, float]:
    """(same-letter, cross-letter) mean cosine between glyph embeddings.

    ``groups`` maps a codepoint to the rows of ``embeddings`` that are its
    positives.  Same-letter similarity is the mean cosine within a group (i.e.
    across fonts); cross-letter is the mean cosine between groups.  When the two
    are close, the embedding carries almost no letter identity: every anchor score
    is then a small residue riding on a shared "it's ink" component, which is what
    lets an unseen style flip across an anchor.

    Both means are computed exactly in O(N) from the sum of unit vectors.
    """
    emb = embeddings.detach().double().cpu()
    all_rows: list[int] = []
    within_pairs = 0
    within_total = 0.0
    for rows in groups.values():
        if not rows:
            continue
        all_rows.extend(rows)
        count = len(rows)
        if count < 2:
            continue
        vector_sum = emb[rows].sum(dim=0)
        within_pairs += count * (count - 1)
        within_total += float(vector_sum.dot(vector_sum)) - count

    n_all = len(all_rows)
    if within_pairs == 0 or n_all < 2:
        return 0.0, 0.0
    vector_sum = emb[all_rows].sum(dim=0)
    total_sum = float(vector_sum.dot(vector_sum)) - n_all
    cross_pairs = n_all * (n_all - 1) - within_pairs
    return (
        within_total / within_pairs,
        (total_sum - within_total) / cross_pairs if cross_pairs else 0.0,
    )


def _macro(values: list[float | None]) -> float:
    return _mean([v for v in values if v is not None])


def _reject_rate(values: list[float]) -> float:
    return (
        sum(1 for v in values if v < REJECT_THRESHOLD) / len(values) if values else 0.0
    )


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(torch.quantile(torch.tensor(values, dtype=torch.float64), q))


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(torch.tensor(values, dtype=torch.float64).std())


def _summary(values: list[float]) -> dict:
    """Mean / std / percentiles of a score distribution."""
    if not values:
        return {k: None for k in ("mean", "std", "p10", "p25", "p50", "p75", "p90")}
    q = torch.quantile(
        torch.tensor(values, dtype=torch.float64),
        torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], dtype=torch.float64),
    )
    return {
        "mean": _mean(values),
        "std": _std(values),
        "p10": float(q[0]),
        "p25": float(q[1]),
        "p50": float(q[2]),
        "p75": float(q[3]),
        "p90": float(q[4]),
    }


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


def content_report(
    model,
    panel: ContentPanel,
    config: ContentDetectorConfig,
    device,
    *,
    target_far: float = DEFAULT_TARGET_FAR,
    top_negatives: int = 0,
) -> dict:
    """Score the panel: AUC, recall@FAR, reject rates, and score geometry."""
    embeddings, logits = score_panel(model, panel, device)
    probabilities = torch.sigmoid(logits)
    cosine = (embeddings @ model.normalized_anchors().detach().cpu().T).double()

    index = config.anchor_index()
    hard_sets = {a: set(hard_negative_pool(a, config.negatives)) for a in config.anchors}

    pos_p: dict[int, list[float]] = defaultdict(list)
    hard_p: dict[int, list[float]] = defaultdict(list)
    any_p: dict[int, list[float]] = defaultdict(list)
    pos_cos: dict[int, list[float]] = defaultdict(list)
    hard_cos: dict[int, list[float]] = defaultdict(list)
    any_cos: dict[int, list[float]] = defaultdict(list)
    hits: dict[int, list[bool]] = defaultdict(list)
    pos_p_strat: dict[tuple, list[float]] = defaultdict(list)
    hard_p_strat: dict[tuple, list[float]] = defaultdict(list)
    any_p_strat: dict[tuple, list[float]] = defaultdict(list)
    hits_strat: dict[tuple, list[bool]] = defaultdict(list)
    off_hard: dict[str, list[float]] = defaultdict(list)
    off_easy: dict[str, list[float]] = defaultdict(list)
    # Hard negatives grouped by the negative's *own* codepoint, for the
    # anchor × confusable breakdown (anchor -> negative codepoint -> scores).
    neg_p: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    neg_cos: dict[int, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for n, codepoint in enumerate(panel.codepoints):
        row_p = probabilities[n]
        row_c = cosine[n]
        stratum = panel.strata[n]
        own = index.get(codepoint)

        if own is not None:
            value = float(row_p[own])
            pos_p[codepoint].append(value)
            pos_cos[codepoint].append(float(row_c[own]))
            pos_p_strat[(stratum, codepoint)].append(value)
            hits[codepoint].append(value >= REJECT_THRESHOLD)
            hits_strat[(stratum, codepoint)].append(value >= REJECT_THRESHOLD)
        elif panel.buckets[n] == "off_hard":
            off_hard[stratum].append(float(row_p.max()))
        else:
            off_easy[stratum].append(float(row_p.max()))

        # As a negative for every *other* anchor — including the hard-negative
        # set, which may itself contain anchor codepoints (o is a confusable of
        # a, and both are anchors).
        for anchor, k in index.items():
            if anchor == codepoint:
                continue
            value = float(row_p[k])
            any_p[anchor].append(value)
            any_cos[anchor].append(float(row_c[k]))
            any_p_strat[(stratum, anchor)].append(value)
            if codepoint in hard_sets[anchor]:
                hard_p[anchor].append(value)
                hard_cos[anchor].append(float(row_c[k]))
                hard_p_strat[(stratum, anchor)].append(value)
                neg_p[anchor][codepoint].append(value)
                neg_cos[anchor][codepoint].append(float(row_c[k]))

    # Per-anchor FAR thresholds, from that anchor's own held-out negatives.
    far_threshold: dict[int, float | None] = {}
    for anchor in config.anchors:
        negatives = hard_p[anchor] or any_p[anchor]
        far_threshold[anchor] = _quantile(negatives, 1.0 - target_far)

    def recall_at_far(pos_values: list[float], anchor: int) -> float | None:
        threshold = far_threshold[anchor]
        if threshold is None or not pos_values:
            return None
        return _mean([1.0 if v >= threshold else 0.0 for v in pos_values])

    per_anchor: dict[str, dict] = {}
    for anchor in config.anchors:
        summary = _summary(pos_cos[anchor])
        threshold = far_threshold[anchor]
        # Which confusable is actually doing the damage, ordered worst-first.
        negatives: dict[str, dict] = {}
        for neg_codepoint, values in neg_p[anchor].items():
            negatives[str(neg_codepoint)] = {
                "n": len(values),
                "auc": _auc(pos_p[anchor], values),
                "mean_cosine": _mean(neg_cos[anchor][neg_codepoint]),
                "above_threshold": (
                    _mean([1.0 if v >= threshold else 0.0 for v in values])
                    if threshold is not None
                    else None
                ),
            }
        negatives = dict(
            sorted(negatives.items(), key=lambda kv: -kv[1]["mean_cosine"])
        )
        per_anchor[str(anchor)] = {
            "auc_hard": _auc(pos_p[anchor], hard_p[anchor]),
            "auc_any": _auc(pos_p[anchor], any_p[anchor]),
            "positive_recall": _mean(hits[anchor]),
            "recall_at_far": recall_at_far(pos_p[anchor], anchor),
            "far_threshold": far_threshold[anchor],
            # The FAR threshold expressed in cosine units, so it can be compared
            # directly with the ~0.06 firing level a 0.5 threshold implies.
            "far_threshold_cosine": _quantile(
                (hard_cos[anchor] or any_cos[anchor]), 1.0 - target_far
            ),
            "bias": float(model.anchor_bias[index[anchor]].detach().cpu()),
            "n_pos": len(pos_p[anchor]),
            "n_hard": len(hard_p[anchor]),
            "positive_cosine": summary,
            "hard_cosine_mean": _mean(hard_cos[anchor]),
            "negatives": negatives,
        }

    strata_present = sorted(set(panel.strata))
    per_stratum: dict[str, dict] = {}
    for stratum in strata_present:
        per_stratum[stratum] = {
            "auc_hard": _macro(
                [
                    _auc(pos_p_strat[(stratum, a)], hard_p_strat[(stratum, a)])
                    for a in config.anchors
                ]
            ),
            "auc_any": _macro(
                [
                    _auc(pos_p_strat[(stratum, a)], any_p_strat[(stratum, a)])
                    for a in config.anchors
                ]
            ),
            "positive_recall": _mean(
                [h for a in config.anchors for h in hits_strat[(stratum, a)]]
            ),
            "recall_at_far": _macro(
                [
                    recall_at_far(pos_p_strat[(stratum, a)], a)
                    for a in config.anchors
                ]
            ),
            "offsample_reject_rate_hard": _reject_rate(off_hard[stratum]),
            "offsample_reject_rate_easy": _reject_rate(off_easy[stratum]),
            "n_pos": sum(len(hits_strat[(stratum, a)]) for a in config.anchors),
            "n_off_hard": len(off_hard[stratum]),
            "n_off_easy": len(off_easy[stratum]),
        }

    all_hits = [h for a in config.anchors for h in hits[a]]
    all_off_hard = [v for values in off_hard.values() for v in values]
    all_off_easy = [v for values in off_easy.values() for v in values]

    anchors_matrix = model.normalized_anchors().detach().cpu()
    gram = anchors_matrix @ anchors_matrix.T
    # The quantity the ranking hinge optimises: own-anchor cosine minus the best
    # *other* anchor's, averaged over positives.  Negative means a neighbour wins.
    rank_margins = []
    for n, codepoint in enumerate(panel.codepoints):
        own = index.get(codepoint)
        if own is None:
            continue
        row = cosine[n]
        other = row.clone()
        other[own] = float("-inf")
        rank_margins.append(float(row[own]) - float(other.max()))
    same_cosine, cross_cosine = letter_similarity(
        embeddings,
        {
            anchor: [n for n, cp in enumerate(panel.codepoints) if cp == anchor]
            for anchor in config.anchors
        },
    )
    anchor_neighbours: dict[str, dict] = {}
    for i, anchor in enumerate(config.anchors):
        row = gram[i].clone()
        row[i] = float("-inf")
        j = int(row.argmax())
        anchor_neighbours[str(anchor)] = {
            "nearest": str(config.anchors[j]),
            "cosine": float(row[j]),
        }

    # The individual off-anchor glyphs that score highest for each anchor — the
    # operational false-accepts. Some are genuinely ambiguous styles; some are
    # candidate mis-encodings in the library (the QA target).
    top_dump: dict[str, list] = {}
    if top_negatives > 0:
        codepoint_tensor = torch.tensor(panel.codepoints)
        for k, anchor in enumerate(config.anchors):
            column = probabilities[:, k].clone()
            not_own = codepoint_tensor != anchor
            column[~not_own] = float("-inf")
            count = min(top_negatives, int(not_own.sum()))
            if count <= 0:
                top_dump[str(anchor)] = []
                continue
            values, indices = torch.topk(column, count)
            top_dump[str(anchor)] = [
                {
                    "codepoint": panel.codepoints[int(i)],
                    "family": panel.families[int(i)],
                    "path": panel.paths[int(i)],
                    "probability": float(v),
                }
                for v, i in zip(values, indices)
            ]

    return {
        "auc_hard": _macro([per_anchor[str(a)]["auc_hard"] for a in config.anchors]),
        "auc_any": _macro([per_anchor[str(a)]["auc_any"] for a in config.anchors]),
        # Pooled recall is dominated by the easy anchors; the macro average is
        # the fairer summary.
        "positive_recall": _mean([float(h) for h in all_hits]),
        "positive_recall_macro": _mean(
            [per_anchor[str(a)]["positive_recall"] for a in config.anchors]
        ),
        "recall_at_far": _macro(
            [per_anchor[str(a)]["recall_at_far"] for a in config.anchors]
        ),
        "target_far": target_far,
        "offsample_reject_rate": _reject_rate(all_off_hard + all_off_easy),
        "offsample_reject_rate_hard": _reject_rate(all_off_hard),
        "offsample_reject_rate_easy": _reject_rate(all_off_easy),
        "n_pos": len(all_hits),
        "n_offsample_hard": len(all_off_hard),
        "n_offsample_easy": len(all_off_easy),
        "logit_scale": float(model.logit_scale.detach().cpu()),
        # The separation that matters: how much higher a glyph scores on its own
        # anchor than its hard negatives do, in *cosine* units.
        "anchor_margin_cosine": _mean(
            [
                _mean(pos_cos[a]) - _mean(hard_cos[a])
                for a in config.anchors
                if pos_cos[a] and hard_cos[a]
            ]
        ),
        "anchor_rank_margin_cosine": _mean(rank_margins),
        "embedding_similarity": {
            "same_letter": same_cosine,
            "cross_letter": cross_cosine,
        },
        "anchor_neighbours": anchor_neighbours,
        "top_negatives": top_dump,
        "per_anchor": per_anchor,
        "per_stratum": per_stratum,
    }


def _fmt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:.3f}"


def _print_report(report: dict) -> None:
    print(f"  logit_scale={report['logit_scale']:.3f}")
    similarity = report["embedding_similarity"]
    print(
        f"  embedding cos: same-letter={similarity['same_letter']:.3f} "
        f"cross-letter={similarity['cross_letter']:.3f} "
        f"anchor_margin={report['anchor_margin_cosine']:+.3f} "
        f"rank_margin={report['anchor_rank_margin_cosine']:+.3f}"
    )
    print(
        f"  auc_hard={report['auc_hard']:.3f}  auc_any={report['auc_any']:.3f}  "
        f"recall@0.5={report['positive_recall']:.3f} "
        f"(macro {report['positive_recall_macro']:.3f})  "
        f"recall@{report['target_far']:.0%}FAR={report['recall_at_far']:.3f}"
    )
    print(
        f"  off-sample reject: hard={report['offsample_reject_rate_hard']:.3f} "
        f"(n={report['n_offsample_hard']})  "
        f"easy={report['offsample_reject_rate_easy']:.3f} "
        f"(n={report['n_offsample_easy']})"
    )
    print("  per anchor:")
    for anchor, m in report["per_anchor"].items():
        cos = m["positive_cosine"]
        print(
            f"    {chr(int(anchor))} (U+{int(anchor):04X}): "
            f"auc_hard={_fmt(m['auc_hard'])} "
            f"recall={m['positive_recall']:.3f} "
            f"recall@far={_fmt(m['recall_at_far'])} "
            f"bias={m['bias']:+.2f} "
            f"cos(pos)={_fmt(cos['mean'])}+-{cos['std']:.2f} "
            f"[p10={_fmt(cos['p10'])} p50={_fmt(cos['p50'])} p90={_fmt(cos['p90'])}] "
            f"far_cos={_fmt(m['far_threshold_cosine'])}"
        )
        for neg, nm in m["negatives"].items():
            print(
                f"        vs {chr(int(neg))} (U+{int(neg):04X}): "
                f"n={nm['n']} auc={_fmt(nm['auc'])} "
                f"cos={_fmt(nm['mean_cosine'])} "
                f"above_far={_fmt(nm['above_threshold'])}"
            )
    print("  nearest anchor per anchor:")
    for anchor, neighbour in report["anchor_neighbours"].items():
        print(
            f"    {chr(int(anchor))} -> {chr(int(neighbour['nearest']))} "
            f"(cos={neighbour['cosine']:.3f})"
        )
    print("  per stratum:")
    for stratum, m in report["per_stratum"].items():
        print(
            f"    {stratum:9}: auc_hard={_fmt(m['auc_hard'])} "
            f"auc_any={_fmt(m['auc_any'])} recall={m['positive_recall']:.3f} "
            f"recall@far={_fmt(m['recall_at_far'])} "
            f"reject(hard/easy)={m['offsample_reject_rate_hard']:.3f}/"
            f"{m['offsample_reject_rate_easy']:.3f} n_pos={m['n_pos']}"
        )
    if any(report.get("top_negatives", {}).values()):
        print("  top-scoring negatives per anchor (candidate ambiguities / errors):")
        for anchor, entries in report["top_negatives"].items():
            for entry in entries:
                codepoint = int(entry["codepoint"])
                print(
                    f"    {chr(int(anchor))} <- {chr(codepoint)} "
                    f"(U+{codepoint:04X}) p={entry['probability']:.3f} "
                    f"{entry['family']}  {entry['path']}"
                )


def main() -> int:
    """Run the full diagnostic breakdown against a content-detector checkpoint."""
    from hrothgar.content_detector.dataset import ContentDatasetMaker
    from hrothgar.content_detector.model import ContentDetector
    from hrothgar.utils import pick_device

    parser = argparse.ArgumentParser(
        description="Diagnose a content-detector checkpoint on held-out fonts."
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts"),
    )
    parser.add_argument("--model", required=True, help="path to the .pth checkpoint")
    parser.add_argument("--max-fonts", type=int, default=200)
    parser.add_argument(
        "--far",
        type=float,
        default=DEFAULT_TARGET_FAR,
        help="target false-accept rate for recall@FAR (default 0.05)",
    )
    parser.add_argument(
        "--top-negatives",
        type=int,
        default=5,
        help="dump this many highest-scoring negatives per anchor (0 disables)",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None, help="write metrics as JSON")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    config = ContentDetectorConfig.from_sidecar(args.model)
    model = ContentDetector(config)
    model.load(args.model, device)
    model.to(device)
    model.eval()

    maker = ContentDatasetMaker(args.repo, config)
    panel = build_content_panel(maker.test_fonts, config, max_fonts=args.max_fonts)
    print(
        f"model: {args.model}  panel: {len(panel.codepoints)} glyphs  "
        f"device: {device}"
    )

    report = content_report(
        model,
        panel,
        config,
        device,
        target_far=args.far,
        top_negatives=args.top_negatives,
    )
    _print_report(report)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
