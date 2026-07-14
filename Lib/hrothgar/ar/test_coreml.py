#!/usr/bin/env python3
"""CLI for testing the Core ML MaskGIT generation pipeline.

Usage (after ``export_coreml.py``)::

    python -m hrothgar.ar.test_coreml \\
        MyFont.ttf --char A --model-dir models/coreml_gen

Requirements: coremltools, freetype-py, matplotlib, numpy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _render_glyph(font_path: str, char: str, size: int) -> np.ndarray:
    import freetype
    face = freetype.Face(font_path)
    face.set_pixel_sizes(size, size)
    face.load_char(char, freetype.FT_LOAD_RENDER)
    bitmap = face.glyph.bitmap
    buf, w, rows = bitmap.buffer, bitmap.width, bitmap.rows
    raw = np.zeros((3, size, size), dtype=np.float32)
    for y in range(rows):
        for x in range(w):
            v = buf[y * w + x] / 255.0
            raw[:, y, x] = v
    return raw


def _render_style_refs(font_path: str, count: int, size: int) -> np.ndarray:
    ref_chars = "ABEGNRSTabdeghknpqy023456789"
    refs = []
    for c in ref_chars:
        try:
            refs.append(_render_glyph(font_path, c, size))
            if len(refs) >= count:
                break
        except Exception:
            continue
    if not refs:
        blank = np.zeros((3, size, size), dtype=np.float32)
        refs = [blank] * count
    while len(refs) < count:
        refs.append(refs[-1])
    return np.stack(refs[:count])


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Test Core ML MaskGIT generator.")
    p.add_argument("font", type=Path, help="Path to a font file.")
    p.add_argument("--char", type=str, required=True, help="Character to generate.")
    p.add_argument("--model-dir", type=Path, default=Path("models/coreml_gen"),
                   help="Directory with exported Core ML models.")
    p.add_argument("--style-ref-count", type=int, default=4,
                   help="Number of style reference glyphs.")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/gen_test"))
    p.add_argument("--no-show", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if not args.font.exists():
        raise FileNotFoundError(f"Font not found: {args.font}")

    from hrothgar.ar.inference_coreml import GeneratorInference

    char = args.char
    if len(char) != 1:
        raise ValueError("--char must be a single Unicode character")

    # Load generator (reads image_size from sidecar).
    gen = GeneratorInference(args.model_dir)
    H = gen.image_size
    K = args.style_ref_count
    print(f"Generator: image_size={H}  steps={gen.num_inference_steps}")

    print(f"Font: {args.font.name}")
    print(f"Generating '{char}' (U+{ord(char):04X}) ...")

    # Render content and style glyphs.
    content = _render_glyph(str(args.font), char, H)
    style = _render_style_refs(str(args.font), K, H)
    print(f"Rendered content + {K} style references at {H}x{H}.")

    # Generate.
    generated = gen.generate(
        content_image=content,
        style_refs=style,
        target_codepoint=ord(char),
    )

    # Save.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.font.stem}_U+{ord(char):04X}"

    def _save(path: Path, arr: np.ndarray) -> None:
        plt.imsave(path, arr.transpose(1, 2, 0).clip(0, 1), vmin=0, vmax=1)

    _save(args.output_dir / f"{stem}_gen_{H}.png", generated)
    print(f"Saved: {args.output_dir / f'{stem}_gen_{H}.png'}")

    # Display.
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(content[0], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Content ({H}x{H})")
    axes[0].axis("off")
    axes[1].imshow(generated[0], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Generated ({H}x{H})")
    axes[1].axis("off")
    fig.suptitle(f"{args.font.name} — {char} (U+{ord(char):04X})")
    fig.tight_layout()
    fig.savefig(args.output_dir / f"{stem}_comparison.png", dpi=150)

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
