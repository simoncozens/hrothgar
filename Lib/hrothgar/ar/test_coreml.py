#!/usr/bin/env python3
"""CLI for testing the Core ML MaskGIT generation pipeline.

Uses the same rendering and style-sampling as ``generate.py`` so outputs
are directly comparable.  Style glyphset and target validation come from
the model's config sidecar.

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
from hrothgar.dataset_constants import LATIN_CORE


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Test Core ML MaskGIT generator.")
    p.add_argument("font", type=Path, help="Path to a font file.")
    p.add_argument("--char", type=str, required=True, help="Character to generate.")
    p.add_argument("--model-dir", type=Path, default=Path("models/coreml_gen"),
                   help="Directory with exported Core ML models.")
    p.add_argument("--reference-font", type=Path, default=None,
                   help="Optional reference font for content image.")
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
    cp = ord(char)

    gen = GeneratorInference(args.model_dir)
    H = gen.image_size

    # ---- Validate target character ----
    target_set = gen.target_glyphset
    if target_set is not None and cp not in target_set:
        raise ValueError(
            f"Character '{char}' (U+{cp:04X}) is not in the model's "
            f"target glyphset.  Model was trained on {len(target_set)} "
            f"codepoints.  Use --help to see available options."
        )
    if target_set is None and cp not in LATIN_CORE:
        raise ValueError(
            f"Character '{char}' (U+{cp:04X}) is not in Latin Core."
        )

    # ---- Style glyphset from config ----
    style_cps = gen.style_glyphset
    if style_cps is None:
        # Fallback: use a reasonable default set.
        style_cps = [ord(c) for c in "ABEGNRSTabdeghknpqy023456789"]
    # Use the same count as the style glyphset size (capped at a reasonable max).
    K = min(len(style_cps), 8)

    print(f"Generator: image_size={H}  steps={gen.num_inference_steps}")
    print(f"  target_set: {'full Latin Core' if target_set is None else f'{len(target_set)} codepoints'}")
    print(f"  style_set:  {len(style_cps)} codepoints, using K={K}")

    # ---- Render inputs ----
    target_font = StandaloneFont(args.font)
    ref_font = StandaloneFont(args.reference_font, reference=target_font) if args.reference_font else target_font
    if args.reference_font:
        target_font = StandaloneFont(args.font, reference=ref_font)

    content_font = ref_font if args.reference_font else target_font
    if not content_font.has_codepoint(cp):
        content_font = target_font
    content = content_font.render(cp, size=H)

    style_chars = _sample_style_codepoints(
        font=target_font,
        target_char=cp,
        style_glyph_count=K,
        common_style_codepoints=style_cps,
    )
    style = np.stack([target_font.render(c, size=H) for c in style_chars])

    print(f"Font: {args.font.name}")
    print(f"Rendered content + {K} style refs at {H}x{H}.")
    if args.reference_font:
        print(f"  content from: {args.reference_font}")

    # ---- Generate ----
    generated = gen.generate(
        content_image=content,
        style_refs=style,
        target_codepoint=cp,
    )

    # ---- Save ----
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.font.stem}_U+{cp:04X}"

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
    fig.suptitle(f"{args.font.name} — {char} (U+{cp:04X})")
    fig.tight_layout()
    fig.savefig(args.output_dir / f"{stem}_comparison.png", dpi=150)

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
