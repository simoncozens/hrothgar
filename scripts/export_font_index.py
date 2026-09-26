#!/usr/bin/env python3
"""Export a FAISS font index to a raw Float32Array buffer + JSON labels.

Reads an index produced by ``scripts/build_font_index.py`` and writes:

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
export, so that a raw dot product in JavaScript is exactly cosine similarity.

This matters because the raw embeddings are strongly mean-shifted: the dataset
mean has a large norm relative to the vectors themselves, so every embedding
points most of the way toward the same direction.  L2-normalizing alone would
leave every pair with a high cosine similarity (a collapsed dynamic range);
centering first, then normalizing, spreads pairwise similarities back across
the full [-1, 1] range.

Usage::

    python scripts/export_font_index.py \
        --index-path outputs/style_embedder/font_index.faiss \
        --output-prefix outputs/style_embedder/font_index
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import faiss
import numpy as np

# Name of the sidecar metadata file written by ``build_font_index.py``.
_METADATA_FILENAME = "font_index_metadata.pkl"


def _extract_vectors(index: faiss.Index, count: int) -> np.ndarray:
    """Return a ``(count, dim)`` little-endian float32 array of all vectors."""
    try:
        vectors = index.reconstruct_n(0, count)
    except AttributeError:
        # Fallback for older faiss versions without ``reconstruct_n``.
        vectors = np.stack([index.reconstruct(i) for i in range(count)])

    return np.ascontiguousarray(vectors, dtype="<f4")


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Return *vectors* with each row scaled to unit L2 norm."""
    v = vectors.astype(np.float64)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return (v / norms).astype("<f4")


def export_index(
    *,
    index_path: Path,
    metadata_path: Path,
    output_prefix: Path,
    center: bool = True,
    normalize: bool = True,
) -> None:
    """Dump the index vectors to ``<prefix>.bin`` and labels to ``<prefix>.json``."""
    index = faiss.read_index(str(index_path))
    dim = int(index.d)
    count = int(index.ntotal)
    print(f"Loaded index: {type(index).__name__}, {count} vectors, dim = {dim}")

    with metadata_path.open("rb") as f:
        metadata = pickle.load(f)

    if len(metadata) != count:
        raise ValueError(
            f"metadata has {len(metadata)} entries but index has {count} vectors"
        )

    vectors = _extract_vectors(index, count)
    if vectors.shape != (count, dim):
        raise ValueError(
            f"extracted vectors shape {vectors.shape}, expected ({count}, {dim})"
        )

    # ── Diagnostics ────────────────────────────────────────────────────
    norms = np.linalg.norm(vectors, axis=1)
    mean = vectors.mean(axis=0, dtype=np.float64)
    mean_norm = float(np.linalg.norm(mean))
    print(
        f"Embedding L2 norms: min/mean/max = {norms.min():.4f} / "
        f"{norms.mean():.4f} / {norms.max():.4f}"
    )
    print(
        f"Dataset mean vector norm = {mean_norm:.4f} "
        f"({mean_norm / float(norms.mean()):.2f} of average vector norm)"
    )

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
    print(
        f"Final embedding L2 norms: min/mean/max = {post_norms.min():.4f} / "
        f"{post_norms.mean():.4f} / {post_norms.max():.4f}"
    )

    # ── Write ──────────────────────────────────────────────────────────
    bin_path = output_prefix.with_suffix(".bin")
    vectors.tofile(bin_path)
    print(
        f"Wrote {vectors.size:,} float32 values ({vectors.nbytes:,} bytes) to "
        f"{bin_path}"
    )

    labels = [{"path": meta["path"], "family": meta["family"]} for meta in metadata]

    payload = {
        "dim": dim,
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
        description="Export a FAISS font index to Float32Array + JSON labels"
    )
    p.add_argument(
        "--index-path", type=Path, required=True, help="Path to the .faiss index file"
    )
    p.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Path to the pickled metadata (defaults to "
        "font_index_metadata.pkl next to the index)",
    )
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output stem; writes <prefix>.bin and <prefix>.json "
        "(defaults to the index path with its extension removed)",
    )
    p.add_argument(
        "--center",
        dest="center",
        action="store_true",
        default=True,
        help="Subtract the dataset mean before normalizing " "(default: on)",
    )
    p.add_argument(
        "--no-center",
        dest="center",
        action="store_false",
        help="Disable mean centering",
    )
    p.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        default=True,
        help="L2-normalize each vector to unit norm (default: on)",
    )
    p.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Disable L2 normalization",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    index_path = args.index_path
    metadata_path = args.metadata_path or index_path.parent / _METADATA_FILENAME
    output_prefix = args.output_prefix or index_path.with_suffix("")

    export_index(
        index_path=index_path,
        metadata_path=metadata_path,
        output_prefix=output_prefix,
        center=args.center,
        normalize=args.normalize,
    )


if __name__ == "__main__":
    main()
