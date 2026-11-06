"""Text-arm similarity: top-k similar fonts for a given font.

Given a font — either a Google Fonts family name (``--font``) or a local TTF
(``--ttf``) — returns the top-k most similar fonts from the *text* arm of the
library, using the per-axis combiner (``similarity.axis_combiner``) with
auto-extractable features: fontquant geometry quantifiers plus the small
feature recognisers.

The anchor font's fontquant data is computed on the fly with the ``fontquant``
package's ``quantify`` interface (no ``fontquant.csv`` lookup); the library is
scored against the precomputed CSV.

Usage::

    python -m hrothgar.text --font Inter
    python -m hrothgar.text --ttf path/to/Font-Regular.ttf --k 10
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from hrothgar.googlefonts import StandaloneFont
from hrothgar.outline_tokens.harness import _device
from hrothgar.outline_tokens.oracle_baseline import _load_cedars
from hrothgar.outline_tokens.recognizer_audit import _family_to_font
from hrothgar.outline_tokens.supervision import build_supervision, text_half_supervision
from hrothgar.similarity.axis_combiner import (
    _load_recognisers,
    _load_table,
    _pair_weight_matrices,
    _quantify_row,
    _score_matrix,
    _train_combiner,
    build_similarity_tensor,
    font_feature_summary,
    pair_breakdown,
)


def _resolve_anchor(name: str | None, ttf: str | None, repo: str):
    """Return a ``Font`` for the anchor (a GF regular instance or a standalone TTF)."""
    if ttf:
        return StandaloneFont(ttf)
    fam2font = _family_to_font(repo)
    return fam2font.get(name.strip().lower())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--font", help="Google Fonts family name to rank against")
    p.add_argument("--ttf", help="local TTF to rank against")
    p.add_argument("--k", type=int, default=20, help="number of results to return")
    p.add_argument("--repo", default=os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts"))
    p.add_argument("--list", default="similarities.json")
    p.add_argument("--judgements", default="judgements.csv")
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
        help="dump the raw feature inputs and per-axis contribution for the anchor "
        "and the explained fonts (default: the top-k results)",
    )
    p.add_argument(
        "--explain-fonts",
        help="comma-separated family names to explain instead of the top-k "
        "(e.g. 'Roboto Flex,Vazirmatn,Yantramanav'); useful for tracing why a "
        "particular font does *not* rank",
    )
    args = p.parse_args()

    if not args.font and not args.ttf:
        p.error("one of --font or --ttf is required")

    dev = _device(args.device)

    # ── Text-arm library universe (full library, then text-like half) ─────────
    sup = build_supervision(args.repo, args.list, args.judgements, universe="full")
    sup = text_half_supervision(sup)

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

    # ── Anchor's fontquant data via the quantify interface ─────────────────────
    extra_table = {Path(anchor.path).name.lower(): _quantify_row(anchor.path)}

    # ── Build the similarity tensor (anchor included) and train the combiner ──
    S, names = build_similarity_tensor(
        sup, feature_source="recogniser", extra_table=extra_table
    )
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
    print(f"\nTop {k} text-arm matches for {anchor.family!r} "
          f"(against {len(sup.fonts) - 1} library fonts):")
    for rank in range(k):
        idx = int(order[rank])
        print(f"  {rank + 1:2d}. {sup.families[idx]:<32s} score={anchor_scores[idx].item():.4f}")

    # ── Explain the inputs / per-axis contributions (debug) ────────────────────
    if args.explain:
        table = _load_table(extra_table)
        cedars = _load_cedars()
        recognisers = _load_recognisers()
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
                print(f"\n(warning: not in text-half universe: {missing}")
        else:
            explain_idxs = [int(order[r]) for r in range(k)]

        print(f"\n===== anchor {anchor.family!r} ({anchor.path.name}) =====")
        for axis, val in font_feature_summary(anchor, table, cedars, recognisers).items():
            print(f"  {axis:<18} {val}")

        for idx in explain_idxs:
            print(f"\n===== {sup.families[idx]!r} ({Path(sup.fonts[idx].path).name}) "
                  f"score={sim[anchor_idx, idx].item():.4f} =====")
            for axis, val in font_feature_summary(sup.fonts[idx], table, cedars, recognisers).items():
                print(f"  {axis:<18} {val}")
            print("  --- per-axis contribution vs anchor (sorted by |contrib|) ---")
            for name, s_ij, w_k, c in pair_breakdown(S, names, w, b, anchor_idx, idx):
                print(f"    {name:<18} sim={s_ij:+.3f} weight={w_k:+.3f} contrib={c:+.3f}")
            print(f"    {'<bias>':<18} {'':>10} {'':>8} {b.item():+.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
