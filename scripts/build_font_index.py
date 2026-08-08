#!/usr/bin/env python3
"""Build a FAISS index of font style embeddings and query for similar fonts.

Iterates over every font in a Google Fonts checkout, renders the fixed
reference glyph set, computes a ``FontStyleEmbedder`` embedding for each,
and builds a FAISS L2 index.  The index can then be queried to find the
top-N most similar fonts to a given target font.

Usage::

    # Build the index (one-time, takes a few minutes):
    python scripts/build_font_index.py build \\
        --dataset-path ~/others-repos/fonts \\
        --model-path outputs/style_embedder/font_style.pth \\
        --index-path outputs/style_embedder/font_index.faiss

    # Query: find fonts similar to Roboto:
    python scripts/build_font_index.py query \\
        --index-path outputs/style_embedder/font_index.faiss \\
        --font-path ~/others-repos/fonts/ofl/roboto/Roboto-Regular.ttf \\
        --top-k 10 \\
        --dataset-path ~/others-repos/fonts \\
        --model-path outputs/style_embedder/font_style.pth
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
import tqdm

from hrothgar.style_embedding.config import FontStyleEmbedderConfig
from hrothgar.style_embedding.dataset import (
    CONTRASTIVE_PHRASES,
    _render_and_resize,
)
from hrothgar.style_embedding.model import FontStyleEmbedder
from hrothgar.googlefonts import (
    GoogleFonts,
    StandaloneFont,
    find_google_font_by_basename,
)
from hrothgar.utils import pick_device

# Per-font metadata stored alongside the FAISS index.
# Keys are FAISS index positions; values are (family, style, font_path).
_METADATA_FILENAME = "font_index_metadata.pkl"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_index(
    *,
    dataset_path: Path,
    model_path: Path,
    index_path: Path,
    device: torch.device,
    batch_size: int = 32,
    limit: Optional[int] = None,
) -> None:
    """Iterate over all fonts, compute embeddings, build FAISS index."""
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
    needed_chars = set()
    for phrase in CONTRASTIVE_PHRASES:
        needed_chars.update(ord(c) for c in phrase if not c.isspace())
    fonts = [
        f for f in gf.fonts
        if needed_chars <= f.codepoints
    ]
    if limit:
        fonts = fonts[:limit]
    print(f"Found {len(fonts)} fonts with characters for phrase rendering")

    # ── Compute embeddings ─────────────────────────────────────────────
    all_embeddings: list[np.ndarray] = []
    metadata: list[dict] = []

    phrase = CONTRASTIVE_PHRASES[0]  # Use first phrase for all fonts.

    for start in tqdm.tqdm(range(0, len(fonts), batch_size), desc="Embedding fonts"):
        batch_fonts = fonts[start : start + batch_size]
        batch_images = []
        for font in batch_fonts:
            img = _render_and_resize(
                font, phrase,
                font_size=config.phrase_font_size,
                axis_position=None,
                phrase_width=config.phrase_width,
                phrase_height=config.phrase_height,
            )
            batch_images.append(img)

        images = torch.stack(batch_images).to(device)  # (B, 3, H, W)

        with torch.no_grad():
            embeddings = model.encode(images)  # (B, dim)

        all_embeddings.append(embeddings.cpu().numpy())

        for font in batch_fonts:
            metadata.append({
                "family": font.family,
                "style": font.metadata.style if hasattr(font.metadata, "style") else "Regular",
                "path": str(font.path),
            })

    emb_matrix = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    print(f"Embedding matrix: {emb_matrix.shape}")

    # ── Build FAISS index ──────────────────────────────────────────────
    index = faiss.IndexFlatL2(dim)
    index.add(emb_matrix)
    print(f"FAISS index: {index.ntotal} vectors")

    # ── Save ───────────────────────────────────────────────────────────
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    print(f"Saved index to {index_path}")

    meta_path = index_path.parent / _METADATA_FILENAME
    with meta_path.open("wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata to {meta_path} ({len(metadata)} entries)")


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def query_index(
    *,
    index_path: Path,
    font_path: Path,
    top_k: int,
    dataset_path: Optional[Path],
    model_path: Optional[Path],
    device: torch.device,
    style_chars: Optional[str] = None,
) -> None:
    """Query the FAISS index for fonts similar to *font_path*.

    If *font_path* is a path to a font *not* in the index, the model is
    loaded and used to compute the embedding on-the-fly.  Otherwise, the
    font is looked up in the index metadata.
    """
    # ── Load index and metadata ────────────────────────────────────────
    index = faiss.read_index(str(index_path))
    meta_path = index_path.parent / _METADATA_FILENAME
    with meta_path.open("rb") as f:
        metadata = pickle.load(f)
    print(f"Loaded index: {index.ntotal} fonts")

    # ── Resolve query embedding ────────────────────────────────────────
    query_emb: Optional[np.ndarray] = None
    query_label = font_path.stem

    # First, try to find the font in the index metadata.
    font_str = str(font_path.resolve())
    for i, meta in enumerate(metadata):
        if Path(meta["path"]).resolve() == font_path.resolve():
            query_emb = index.reconstruct(i)
            query_label = f"{meta['family']} ({meta['style']})"
            print(f"Found '{query_label}' in index at position {i}")
            break

    if query_emb is None:
        # Font not in index — compute embedding on-the-fly.
        if model_path is None:
            raise ValueError(
                "Font not found in index; --model-path is required to "
                "compute embedding on-the-fly"
            )
        config = FontStyleEmbedderConfig.from_sidecar(model_path)
        model = FontStyleEmbedder(config).to(device)
        model.load(str(model_path), device=device)
        model.eval()

        # Try Google Font first, fall back to standalone for arbitrary files.
        try:
            if dataset_path is not None:
                font = find_google_font_by_basename(dataset_path, font_path)
                query_label = (
                    f"{font.family} "
                    f"({font.metadata.style if hasattr(font.metadata, 'style') else '?'})"
                )
            else:
                raise ValueError("no dataset path")
        except Exception:
            font = StandaloneFont(font_path)
            query_label = font_path.stem
            print(f"Using StandaloneFont for '{query_label}' "
                  f"(not in Google Fonts)")

        # Resolve reference phrase.
        phrase = CONTRASTIVE_PHRASES[0]

        img = _render_and_resize(
            font, phrase,
            font_size=config.phrase_font_size,
            axis_position=None,
            phrase_width=config.phrase_width,
            phrase_height=config.phrase_height,
        )
        images = img.unsqueeze(0).to(device)  # (1, 3, H, W)

        with torch.no_grad():
            query_emb = model.encode(images).cpu().numpy().astype(np.float32)[0]
        print(f"Computed embedding for '{query_label}'")

    # ── Search ─────────────────────────────────────────────────────────
    query_emb = query_emb.reshape(1, -1)
    distances, indices = index.search(query_emb, top_k + 1)  # +1 to exclude self

    print(f"\nTop {top_k} most similar fonts to '{query_label}':\n")
    print(f"{'Rank':<5} {'Distance':<10} {'Family':<30} {'Style':<20} {'Path'}")
    print("-" * 100)

    rank = 0
    for dist, idx in zip(distances[0], indices[0]):
        meta = metadata[idx]
        # Skip self-match.
        if Path(meta["path"]).resolve() == font_path.resolve():
            continue
        rank += 1
        print(f"{rank:<5} {dist:<10.4f} {meta['family']:<30} "
              f"{meta['style']:<20} {meta['path']}")
        if rank >= top_k:
            break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build and query a FAISS index of font style embeddings"
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ---- build ----
    b = sub.add_parser("build", help="Build FAISS index from all fonts")
    b.add_argument("--dataset-path", type=Path, required=True)
    b.add_argument("--model-path", type=Path, required=True)
    b.add_argument("--index-path", type=Path, required=True)
    b.add_argument("--batch-size", type=int, default=32)
    b.add_argument("--limit", type=int, default=None,
                   help="Limit number of fonts (for testing)")

    # ---- query ----
    q = sub.add_parser("query", help="Query FAISS index for similar fonts")
    q.add_argument("--index-path", type=Path, required=True)
    q.add_argument("--font-path", type=Path, required=True,
                   help="Path to font file to query")
    q.add_argument("--top-k", type=int, default=10)
    q.add_argument("--dataset-path", type=Path, default=None,
                   help="Path to Google Fonts checkout (optional; required only "
                        "if querying a Google Font by filename basename)")
    q.add_argument("--model-path", type=Path, default=None,
                   help="Required if font is not in the index")
    q.add_argument("--style-chars", type=str, default=None,
                   help="Override reference glyph string (default: hamburgeHAMBURGE)")

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = pick_device()
    print(f"Using device: {device}")
    #device = torch.device("cpu")

    if args.command == "build":
        build_index(
            dataset_path=args.dataset_path,
            model_path=args.model_path,
            index_path=args.index_path,
            device=device,
            batch_size=args.batch_size,
            limit=args.limit,
        )
    elif args.command == "query":
        query_index(
            index_path=args.index_path,
            font_path=args.font_path,
            top_k=args.top_k,
            dataset_path=args.dataset_path,
            model_path=args.model_path,
            device=device,
            style_chars=getattr(args, "style_chars", None),
        )


if __name__ == "__main__":
    main()
