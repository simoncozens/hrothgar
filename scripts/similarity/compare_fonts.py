#!/usr/bin/env python3
"""Compare two fonts against the library index.

For a query font A: lists its top-k nearest neighbours.  For a second font B:
reports the (centered) cosine of B with A and where B ranks among A's
neighbours — i.e. "is B actually near A, and is anything else closer?"

Similarities use the same mean-centering + L2-normalization as the eval suite
(``metrics.prepare(center=True)``), so the numbers match the training/eval logs.

Usage::

    PYTHONPATH=Lib python scripts/similarity/compare_fonts.py \
        --repo ~/others-repos/fonts --model models/similarity.pth \
        --font-a 'Lexend[wght].ttf' --font-b 'ZalandoSans-SemiExpanded.ttf'
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from hrothgar.googlefonts import GoogleFonts
from hrothgar.similarity.config import FontStyleEmbedderConfig
from hrothgar.similarity.evaluation import build_eval_panel, embed_panel
from hrothgar.similarity.metrics import prepare
from hrothgar.similarity.model import FontStyleEmbedder
from hrothgar.utils import pick_device


def _load_model(path: str, device) -> FontStyleEmbedder:
    config = FontStyleEmbedderConfig.from_sidecar(path)
    model = FontStyleEmbedder(config)
    model.load(path, device)
    model.to(device)
    model.eval()
    return model


def _find(panel, name: str) -> int | None:
    for i, font in enumerate(panel.fonts):
        if font.path.name == name:
            return i
    return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--repo",
        default=os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts"),
    )
    p.add_argument("--model", default="models/similarity.pth")
    p.add_argument("--device", default=None)
    p.add_argument("--font-a", required=True, help="query font (file name)")
    p.add_argument("--font-b", required=True, help="comparison font (file name)")
    p.add_argument("--limit", type=int, default=None, help="cap fonts indexed (canary)")
    p.add_argument("--top-k", type=int, default=5)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    device = pick_device() if args.device is None else args.device
    model = _load_model(args.model, device)
    config = model.config

    gf = GoogleFonts(args.repo, having=set(config.input_codepoints))
    panel = build_eval_panel(
        gf.fonts, config, max_fonts=args.limit, cache_images=True
    )
    emb = embed_panel(model, panel, device, space="summary")
    x = prepare(emb, center=True)  # mean-centered + L2-normalized

    name_a = Path(args.font_a).name
    name_b = Path(args.font_b).name
    ia = _find(panel, name_a)
    ib = _find(panel, name_b)
    if ia is None:
        raise SystemExit(f"font A not found in panel: {name_a}")
    if ib is None:
        raise SystemExit(f"font B not found in panel: {name_b}")

    sim = x @ x[ia]  # similarity of A to every font (N,)
    sim[ia] = float("-inf")

    print(f"Query A: {panel.fonts[ia].path.name}  ({panel.fonts[ia].family})")
    print(f"top-{args.top_k} neighbours:")
    topk = sim.topk(min(args.top_k, len(sim) - 1))
    for val, idx in zip(topk.values, topk.indices):
        mark = "=" if panel.fonts[idx].family == panel.fonts[ia].family else " "
        print(f"    {mark} {float(val):+.3f}  {panel.fonts[idx].family:40} {panel.fonts[idx].path.name}")

    cos_ab = float(sim[ib])
    rank = int((sim > cos_ab).sum()) + 1
    print(f"\nQuery B: {panel.fonts[ib].path.name}  ({panel.fonts[ib].family})")
    print(f"  cosine(A, B): {cos_ab:+.3f}")
    print(f"  rank of B among A's neighbours: {rank} / {len(sim)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
