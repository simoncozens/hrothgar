#!/usr/bin/env python3
"""CLI for testing the Core ML MaskGIT generation pipeline.

Uses the same rendering and style-sampling as ``generate.py`` so outputs
are directly comparable.

Usage::

    python -m hrothgar.ar.test_coreml \\
        MyFont.ttf --char A --model-dir models/coreml_gen

Requirements: coremltools, matplotlib, numpy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hrothgar.googlefonts import StandaloneFont
from hrothgar.ar.style_sampling import _sample_style_codepoints


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Test Core ML MaskGIT generator.")
    p.add_argument("font", type=Path, help="Path to a font file.")
    p.add_argument("--char", type=str, required=True, help="Character to generate.")
    p.add_argument("--model-dir", type=Path, default=Path("models/coreml_gen"),
                   help="Directory with exported Core ML models.")
    p.add_argument("--reference-font", type=Path, default=None,
                   help="Optional reference font for content image.")
    p.add_argument("--style-chars", type=str, default="ABEGNRSTabdeghknpqy023456789",
                   help="Characters to use as style references.")
    p.add_argument("--style-ref-count", type=int, default=8,
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

    gen = GeneratorInference(args.model_dir)
    H = gen.image_size
    K = args.style_ref_count
    print(f"Generator: image_size={H}  steps={gen.num_inference_steps}")

    # ---- Render inputs (same as generate.py) ----
    target_font = StandaloneFont(args.font)
    ref_font = StandaloneFont(args.reference_font, reference=target_font) if args.reference_font else target_font
    if args.reference_font:
        target_font = StandaloneFont(args.font, reference=ref_font)

    # Content: from reference font if available, else from target.
    content_font = ref_font if args.reference_font else target_font
    if not content_font.has_codepoint(ord(char)):
        content_font = target_font
    content = content_font.render(ord(char), size=H)

    # Style: using the same sampling logic as generate.py.
    common_cps = [ord(c) for c in args.style_chars]
    style_chars = _sample_style_codepoints(
        font=target_font,
        target_char=ord(char),
        style_glyph_count=K,
        common_style_codepoints=common_cps,
    )
    style_rasters = [target_font.render(cp, size=H) for cp in style_chars]
    style = np.stack(style_rasters)

    print(f"Font: {args.font.name}")
    print(f"Rendered content + {K} style refs at {H}x{H}.")
    if args.reference_font:
        print(f"  content from: {args.reference_font}")

    # ---- Generate ----
    generated = gen.generate(
        content_image=content,
        style_refs=style,
        target_codepoint=ord(char),
    )

    # ---- Save ----
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.font.stem}_U+{ord(char):04X}"

    def _save(path, arr):
        if arr.ndim == 4:
            arr = arr[0]
        plt.imsave(path, arr.transpose(1, 2, 0).clip(0, 1), vmin=0, vmax=1)

    _save(args.output_dir / f"{stem}_gen_{H}.png", generated)
    print(f"Saved: {args.output_dir / f'{stem}_gen_{H}.png'}")

    # ---- Display ----
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(content[0], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Content ({H})")
    axes[0].axis("off")
    axes[1].imshow(generated[0], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Generated ({H})")
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
