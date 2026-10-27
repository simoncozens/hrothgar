#!/usr/bin/env python3
"""Humanist-vs-geometric sans linear probe.

Trains a regularised linear classifier on frozen GlyphEncoder features and
reports leave-one-family-out accuracy/AUC, so the number is a family-level
generalisation measure rather than font-file memorisation.  Also reports a raw
pixel baseline at the same grids to localise any failure to the input vs. the
architecture.

Usage::

    PYTHONPATH=Lib python scripts/similarity/style_probe.py \
        --repo ~/others-repos/fonts
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from hrothgar.similarity.style_probe import run_style_probe


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--repo",
        default=os.environ.get("GOOGLE_FONTS_REPO", "/Users/simon/others-repos/fonts"),
    )
    p.add_argument(
        "--use-tags",
        action="store_true",
        help="auto-select families by /Sans/{Humanist,Geometric} tags instead of the curated list",
    )
    p.add_argument("--tag-threshold", type=float, default=60.0)
    p.add_argument("--n-per-class", type=int, default=None)
    p.add_argument("--glyph-size", type=int, default=128)
    p.add_argument("--downsample", type=int, choices=(2, 4, 8), default=4)
    p.add_argument("--spatial-channels", type=int, default=32)
    p.add_argument("--grids", default="8,16,32")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None, help="write full results here")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    grids = tuple(int(x) for x in args.grids.split(",") if x.strip())

    report = run_style_probe(
        args.repo,
        use_tags=args.use_tags,
        tag_threshold=args.tag_threshold,
        n_per_class=args.n_per_class,
        glyph_size=args.glyph_size,
        downsample=args.downsample,
        spatial_channels=args.spatial_channels,
        grids=grids,
        seed=args.seed,
    )

    print(
        f"[1] selection: {report['selection']}  fonts={report['num_fonts']}  "
        f"glyph_size={report['glyph_size']}  downsample={report['downsample']}"
    )
    print("    families: " + ", ".join(report["families"]))

    print("\n[2] leave-one-family-out linear-probe (chance = class balance)")
    print(f"    {'feature':14} {'dim':>6} {'acc':>6} {'auc':>6} {'perm':>6}")
    for name, r in report["features"].items():
        print(
            f"    {name:14} {r['dim']:>6} {r['accuracy']:6.3f} {r['auc']:6.3f} "
            f"{r['permuted_accuracy']:6.3f}"
        )

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
