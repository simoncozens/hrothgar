#!/usr/bin/env python3
"""Build the font style embedding database (raw float32 + JSON labels).

Iterates over every font in a Google Fonts checkout, computes a
``FontStyleEmbedder`` embedding for each, and writes the result directly as a
browser-loadable blob:

- ``<output>.bin``  — a single contiguous little-endian float32 buffer holding
  every embedding, row-major.  It contains ``count * dim`` floats, so in
  JavaScript it can be loaded directly with::

      const floats = new Float32Array(await (await fetch(url)).arrayBuffer());
      // vector i = floats.subarray(i * dim, (i + 1) * dim)

- ``<output>.json`` — ``{"dim": ..., "count": ..., "centered": ...,
  "normalized": ..., "labels": [...]}`` where each label is
  ``{"path": ..., "family": ...}`` and is ordered to match the rows of the
  binary buffer (label[i] corresponds to rows ``[i*dim, (i+1)*dim)``).

By default the embeddings are mean-centered and then L2-normalized before
writing, so that a raw dot product in JavaScript is exactly cosine similarity.

This matters because the raw embeddings are strongly mean-shifted: the dataset
mean has a large norm relative to the vectors themselves, so every embedding
points most of the way toward the same direction.  L2-normalizing alone would
leave every pair with a high cosine similarity (a collapsed dynamic range);
centering first, then normalizing, spreads pairwise similarities back across
the full [-1, 1] range.

Usage::

    python scripts/build_font_index.py \
        --dataset-path ~/others-repos/fonts \
        --model-path outputs/style_embedder/font_style.pth \
        --output-prefix outputs/style_embedder/font_index
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tqdm

from hrothgar.style_embedding.config import FontStyleEmbedderConfig
from hrothgar.style_embedding.model import FontStyleEmbedder
from hrothgar.googlefonts import GoogleFonts
from hrothgar.utils import pick_device


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Return *vectors* with each row scaled to unit L2 norm."""
    v = vectors.astype(np.float64)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return (v / norms).astype("<f4")


def build_index(
    *,
    dataset_path: Path,
    model_path: Path,
    output_prefix: Path,
    device: torch.device,
    limit: Optional[int] = None,
    center: bool = True,
    normalize: bool = True,
) -> None:
    """Compute font embeddings and write them as a raw blob + JSON labels."""
    # ── Load model ─────────────────────────────────────────────────────
    config = FontStyleEmbedderConfig.from_sidecar(model_path)
    model = FontStyleEmbedder(config).to(device)
    model.load(str(model_path), device=device)
    model.eval()
    dim = config.encoder_feature_dim
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params, "
          f"embedding dim = {dim}")

    # ── Collect fonts ──────────────────────────────────────────────────
    gf = GoogleFonts(str(dataset_path))
    needed = set(config.input_codepoints)
    fonts = [
        f for f in gf.fonts
        if needed <= f.codepoints
    ]
    if limit:
        fonts = fonts[:limit]
    print(f"Found {len(fonts)} fonts with the full input glyph set")

    # ── Compute embeddings ─────────────────────────────────────────────
    all_embeddings: list[np.ndarray] = []
    labels: list[dict] = []
    skipped = 0

    for font in tqdm.tqdm(fonts, desc="Embedding fonts"):
        try:
            embedding = model.compute_embedding(font, device=device).numpy()
        except ValueError:
            # ``compute_embedding`` raises when a glyph renders blank.
            skipped += 1
            continue
        all_embeddings.append(embedding)
        labels.append({
            "path": str(font.path),
            "family": font.family,
        })

    if skipped:
        print(f"Skipped {skipped} fonts with blank glyphs")
    if not all_embeddings:
        raise RuntimeError("No embeddings were produced (all fonts skipped)")

    vectors = np.stack(all_embeddings).astype(np.float32)
    count = vectors.shape[0]
    print(f"Embedding matrix: {vectors.shape}")

    # ── Diagnostics ────────────────────────────────────────────────────
    norms = np.linalg.norm(vectors, axis=1)
    mean = vectors.mean(axis=0, dtype=np.float64)
    mean_norm = float(np.linalg.norm(mean))
    print(f"Embedding L2 norms: min/mean/max = {norms.min():.4f} / "
          f"{norms.mean():.4f} / {norms.max():.4f}")
    print(f"Dataset mean vector norm = {mean_norm:.4f} "
          f"({mean_norm / float(norms.mean()):.2f} of average vector norm)")

    # ── Transform ──────────────────────────────────────────────────────
    if center:
        vectors = (vectors.astype(np.float64) - mean).astype("<f4")
        print("Centered: subtracted the dataset mean.")
    else:
        print("Centering skipped.")

    if normalize:
        vectors = _l2_normalize(vectors)
        print("Normalized: L2-scaled each vector to unit norm.")
    else:
        print("Normalization skipped.")

    post_norms = np.linalg.norm(vectors, axis=1)
    print(f"Final embedding L2 norms: min/mean/max = {post_norms.min():.4f} / "
          f"{post_norms.mean():.4f} / {post_norms.max():.4f}")

    # ── Write ──────────────────────────────────────────────────────────
    vectors = np.ascontiguousarray(vectors, dtype="<f4")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    bin_path = output_prefix.with_suffix(".bin")
    vectors.tofile(bin_path)
    print(f"Wrote {vectors.size:,} float32 values ({vectors.nbytes:,} bytes) to "
          f"{bin_path}")

    payload = {
        "dim": int(vectors.shape[1]),
        "count": count,
        "centered": center,
        "normalized": normalize,
        "labels": labels,
    }

    json_path = output_prefix.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {count} labels to {json_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the font style embedding database (raw float32 + JSON)"
    )
    p.add_argument("--dataset-path", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--output-prefix", type=Path, required=True,
                   help="Output stem; writes <prefix>.bin and <prefix>.json")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of fonts (for testing)")
    p.add_argument("--center", dest="center", action="store_true", default=True,
                   help="Subtract the dataset mean before normalizing "
                        "(default: on)")
    p.add_argument("--no-center", dest="center", action="store_false",
                   help="Disable mean centering")
    p.add_argument("--normalize", dest="normalize", action="store_true",
                   default=True,
                   help="L2-normalize each vector to unit norm (default: on)")
    p.add_argument("--no-normalize", dest="normalize", action="store_false",
                   help="Disable L2 normalization")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = pick_device()
    print(f"Using device: {device}")

    build_index(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        output_prefix=args.output_prefix,
        device=device,
        limit=args.limit,
        center=args.center,
        normalize=args.normalize,
    )


if __name__ == "__main__":
    main()
