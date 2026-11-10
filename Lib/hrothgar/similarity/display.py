"""Display-arm similarity: top-k "same vibe" fonts for a given font.

The tag-only counterpart to :mod:`hrothgar.similarity.text`.  Unlike the text
arm, this feeds **no** fontquant, CEDARS or recogniser data — every axis is a
masked cosine over a tag family (all 74 tags: style / theme / expressive /
quality / purpose / seasonal / special-use / monospace), and the combiner learns
how much each tag family's agreement should contribute to "fits the vibe".

Because the input is tags only, this is effectively ``--font``-only for now: a
standalone TTF has no tags, so its every axis would be all-missing and it cannot
be ranked.  (Wiring in a tag-prediction model for unseen TTFs is the obvious
follow-up, matching how the text arm extracts fontquant on the fly.)

Supervision is still sparse in the display half, so the learned combiner weights
should not be trusted yet — this front-end exists so the plumbing is ready as
more positive/negative reviewer data lands.

Usage::

    python -m hrothgar.similarity.display --font Creepster --k 10
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from hrothgar.googlefonts import StandaloneFont, families_to_fonts
from hrothgar.similarity.supervision import (
    JUDGEMENTS_PATH,
    SIMILARITIES_PATH,
    build_supervision,
    display_half_supervision,
)
from hrothgar.utils import device
from hrothgar.similarity.axis_combiner import (
    _pair_weight_matrices,
    _score_matrix,
    _train_combiner,
    build_display_tensor,
    pair_breakdown,
    tag_axes_summary,
)


def _resolve_anchor(name: str | None, ttf: str | None, repo: str):
    """Return a ``Font`` for the anchor (a GF regular instance or a standalone TTF)."""
    if ttf:
        return StandaloneFont(ttf)
    fam2font = families_to_fonts(repo)
    return fam2font.get(name.strip().lower())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--font", help="Google Fonts family name to rank against")
    p.add_argument("--ttf", help="local TTF (no tags — will rank degenerate; see docstring)")
    p.add_argument("--k", type=int, default=20, help="number of results to return")
    p.add_argument("--repo", default=os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts"))
    p.add_argument("--list", default=SIMILARITIES_PATH)
    p.add_argument("--judgements", default=JUDGEMENTS_PATH)
    p.add_argument(
        "--combiner",
        choices=("linear", "softmax"),
        default="linear",
        help="how the per-axis similarities are combined (linear = signed weights, "
        "softmax = non-negative convex combination)",
    )
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--margin", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=128.0)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--explain",
        action="store_true",
        help="dump the tag inputs and per-axis contribution for the anchor and the "
        "explained fonts (default: the top-k results)",
    )
    p.add_argument(
        "--explain-fonts",
        help="comma-separated family names to explain instead of the top-k "
        "(e.g. 'Eater,Creepster'); useful for tracing why a particular font ranks",
    )
    args = p.parse_args()

    if not args.font and not args.ttf:
        p.error("one of --font or --ttf is required")

    dev = device(args.device)

    # ── Display-arm library universe (full library, then display-like half) ────
    sup = build_supervision(args.repo, args.list, args.judgements, universe="full")
    sup = display_half_supervision(sup)

    # ── Resolve the anchor and place it in the universe ────────────────────────
    anchor = _resolve_anchor(args.font, args.ttf, args.repo)
    if anchor is None:
        print(f"font {args.font!r} not found in the library", flush=True)
        return 1

    anchor_idx = None
    for i, fam in enumerate(sup.families):
        if fam.lower() == anchor.family.lower():
            anchor_idx = i
            break
    if anchor_idx is None:
        anchor_idx = len(sup.fonts)
        sup.fonts.append(anchor)
        sup.families.append(anchor.family)
        sup.font_ids.append(str(anchor.path))

    # ── Tag-only similarity tensor (anchor included) + train the combiner ──────
    S, names = build_display_tensor(sup)
    S = S.to(dev)
    pos, neg = _pair_weight_matrices(sup, dev)
    w, b = _train_combiner(
        S, pos, neg, combiner=args.combiner, steps=args.steps, lr=args.lr,
        margin=args.margin, gamma=args.gamma, seed=args.seed,
        weight_decay=args.weight_decay,
    )
    sim = _score_matrix(S, w, b)

    # ── Rank the anchor against the library (exclude the anchor itself) ───────
    anchor_scores = sim[anchor_idx]
    anchor_scores = anchor_scores.clone()
    anchor_scores[anchor_idx] = float("-inf")
    order = torch.argsort(anchor_scores, descending=True)

    k = min(args.k, len(sup.fonts) - 1)
    print(f"\nTop {k} display-arm matches for {anchor.family!r} "
          f"(against {len(sup.fonts) - 1} library fonts):")
    for rank in range(k):
        idx = int(order[rank])
        print(f"  {rank + 1:2d}. {sup.families[idx]:<32s} score={anchor_scores[idx].item():.4f}")

    # ── Explain the tag inputs / per-axis contributions (debug) ────────────────
    if args.explain:
        fam_to_idx = {f.lower(): i for i, f in enumerate(sup.families)}
        if args.explain_fonts:
            targets = [t.strip().lower() for t in args.explain_fonts.split(",") if t.strip()]
            explain_idxs, missing = [], []
            for t in targets:
                if t in fam_to_idx:
                    explain_idxs.append(fam_to_idx[t])
                else:
                    missing.append(t)
            if missing:
                print(f"\n(warning: not in display-half universe: {missing}")
        else:
            explain_idxs = [int(order[r]) for r in range(k)]

        print(f"\n===== anchor {anchor.family!r} ({anchor.path.name}) =====")
        for axis, val in tag_axes_summary(anchor).items():
            print(f"  {axis:<18} {val}")

        for idx in explain_idxs:
            print(f"\n===== {sup.families[idx]!r} ({Path(sup.fonts[idx].path).name}) "
                  f"score={sim[anchor_idx, idx].item():.4f} =====")
            for axis, val in tag_axes_summary(sup.fonts[idx]).items():
                print(f"  {axis:<18} {val}")
            print("  --- per-axis contribution vs anchor (sorted by |contrib|) ---")
            for name, s_ij, w_k, c in pair_breakdown(S, names, w, b, anchor_idx, idx):
                print(f"    {name:<18} sim={s_ij:+.3f} weight={w_k:+.3f} contrib={c:+.3f}")
            print(f"    {'<bias>':<18} {'':>10} {'':>8} {b.item():+.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
