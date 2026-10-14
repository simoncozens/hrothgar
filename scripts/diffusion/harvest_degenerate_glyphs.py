#!/usr/bin/env python
"""Harvest raw generated glyphs for every sans/serif instance in a checkpoint's
vocabulary.

Given a trained factorized ``(codepoint, font-instance)`` diffusion checkpoint
and its ``.conf.json`` / ``.codepoints.json`` / ``.instances.json`` sidecars,
this walks every training instance whose family classifies as **sans-serif** or
**serif** (display / handwriting / script / mono / unknown are skipped) and, for
each such instance, samples **every codepoint in the model's vocabulary**.

Unlike :mod:`generate_rupees`, there is no super-resolution, no geometry, and no
de-normalization: each glyph is written as the raw diffusion output
(``[0, 1]``, ink=0 / white=1) at its native resolution to
``<output-dir>/<family>-<codepoint>.png``.  The point is to eyeball the output
of mid-training checkpoints and pick out structurally degenerate glyphs to use
as negatives for the content-coherence evaluator.

Category is read from each family's ``METADATA.pb`` via
:func:`hrothgar.dataset.style_bucket`, so ``--repo`` is required (both to resolve
the repo-relative instance paths and to read the family metadata).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import tqdm
from hrothgar.dataset import style_bucket
from hrothgar.diffusion.config import FontIdDiffusionConfig
from hrothgar.diffusion.fontid import build_fontid_model
from hrothgar.dataset_constants import UCLC_ONLY
from hrothgar.googlefonts import GoogleFont
from hrothgar.utils import pick_device


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve(repo: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo / p


def _sanitize(name: str) -> str:
    """Make a family name safe for a filename (spaces -> underscores)."""
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")
    return cleaned or "family"


def _save_grayscale(image: np.ndarray, path: Path) -> None:
    """Save a ``(H, W)`` [0, 1] (ink=0 / white=1) array as a grayscale PNG."""
    from PIL import Image

    arr = np.clip(image, 0.0, 1.0)
    Image.fromarray((arr * 255.0).astype(np.uint8), mode="L").save(path)


def _chunked(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained diffusion checkpoint (sidecars derived)",
    )
    p.add_argument(
        "--repo",
        type=str,
        default=os.environ.get("GOOGLE_FONTS_REPO"),
        help="Google Fonts repo root, to resolve repo-relative instance paths "
        "and read each family's METADATA.pb",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs/generated",
        help="Directory to write raw generated glyphs into",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Sampling seed (DDIM is deterministic, but initial noise is seeded)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process this many sans/serif instances (for a smoke test)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Codepoints sampled per forward pass",
    )
    args = p.parse_args()

    if not args.repo:
        raise SystemExit("Provide --repo or set GOOGLE_FONTS_REPO")

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"Using device: {device}")

    model_path = Path(args.model_path)
    config = FontIdDiffusionConfig.from_sidecar(model_path)
    codepoints = _load_json(Path(str(model_path) + ".codepoints.json"))
    instances = _load_json(Path(str(model_path) + ".instances.json"))
    cp_to_idx = {cp: i for i, cp in enumerate(codepoints)
        if cp in UCLC_ONLY}
    codepoints = [cp for cp in codepoints if cp in UCLC_ONLY]  # only generate uppercase for now
    repo = Path(args.repo)

    model = build_fontid_model(config).to(device)
    model.load(str(model_path), device=device)
    model.eval()

    # Classify each instance's family once; keep only sans/serif text faces.
    category_cache: dict[str, str] = {}
    skipped: Counter = Counter()
    kept: list[tuple[dict, str]] = []
    for inst in instances:
        family = inst["family"]
        if family not in category_cache:
            try:
                gfont = GoogleFont(_resolve(repo, inst["path"]))
                category_cache[family] = style_bucket(gfont)
            except Exception:  # e.g. missing METADATA.pb on a partial checkout
                category_cache[family] = "other"
        bucket = category_cache[family]
        if bucket not in ("sans", "serif"):
            skipped[bucket] += 1
            continue
        kept.append((inst, bucket))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    n_kept = 0
    progress = tqdm.tqdm(kept, desc="instances")

    for inst, bucket in progress:
        family = inst["family"]
        family_key = _sanitize(family)
        style_suffix = "" if inst["style"] == "normal" else "-italic"
        progress.set_description(f"{bucket}/{family}")

        meta = torch.tensor(
            [[inst["family_id"], inst["style_bucket"]]],
            device=device,
            dtype=torch.float32,
        )
        for chunk in _chunked(codepoints, args.batch_size):
            cp_t = torch.tensor(
                [cp_to_idx[cp] for cp in chunk], device=device, dtype=torch.long
            )
            with torch.no_grad():
                glyphs = model.sample(cp_t, meta.repeat(len(chunk), 1))
            for cp, glyph in zip(chunk, glyphs):
                arr = glyph[0].cpu().numpy()  # (H, W) crop-to-ink, [0, 1]
                _save_grayscale(
                    arr, out_dir / f"{family_key}{style_suffix}-U+{cp:04X}.png"
                )
                generated += 1

        n_kept += 1
        if args.limit is not None and n_kept >= args.limit:
            break

    print(f"Generated {generated} glyphs from {n_kept} sans/serif instances -> {out_dir}")
    if skipped:
        print("Skipped non-sans/serif instances by bucket:", dict(skipped))


if __name__ == "__main__":
    main()
