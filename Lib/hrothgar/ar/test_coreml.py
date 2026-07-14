#!/usr/bin/env python3
"""CLI for testing the Core ML MaskGIT generation pipeline.

Usage (after ``export_coreml.py``)::

    python -m hrothgar.ar.test_coreml \\
        MyFont.ttf --char A --model-dir models/coreml_gen

Requirements: coremltools, fonttools, matplotlib, numpy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from fontTools.ttLib import TTFont

from hrothgar.render import render_gid


def _char_to_gid(font_path: str, char: str) -> int:
    """Look up the GID for a character in a font."""
    tt = TTFont(font_path)
    cmap = tt.getBestCmap()
    cp = ord(char)
    if cp not in cmap:
        raise ValueError(f"Character '{char}' (U+{cp:04X}) not in font")
    gid = cmap[cp]
    tt.close()
    return gid


def _render_style_refs(font_path: str, count: int, size: int) -> np.ndarray:
    """Render *count* style reference glyphs as (count, 3, size, size)."""
    # These match the style characters used during training.
    ref_chars = "ABEGNRSTabdeghknpqy023456789"
    tt = TTFont(font_path)
    cmap = tt.getBestCmap()
    refs: list[np.ndarray] = []
    for c in ref_chars:
        cp = ord(c)
        if cp in cmap:
            img = render_gid(font_path, cmap[cp], size)
            if not np.allclose(img, 1.0, atol=1e-2):
                refs.append(img)
                if len(refs) >= count:
                    break
    tt.close()

    if not refs:
        blank = np.ones((3, size, size), dtype=np.float32)
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

    gen = GeneratorInference(args.model_dir)
    H = gen.image_size
    K = args.style_ref_count
    print(f"Generator: image_size={H}  steps={gen.num_inference_steps}")

    print(f"Font: {args.font.name}")
    print(f"Generating '{char}' (U+{ord(char):04X}) ...")

    # Render content and style glyphs using the same path as generate.py.
    target_gid = _char_to_gid(str(args.font), char)
    content = render_gid(str(args.font), target_gid, H)
    style = _render_style_refs(str(args.font), K, H)
    print(f"Rendered content (GID {target_gid}) + {K} style refs at {H}x{H}.")

    generated = gen.generate(
        content_image=content,
        style_refs=style,
        target_codepoint=ord(char),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.font.stem}_U+{ord(char):04X}"

    def _save(path: Path, arr: np.ndarray) -> None:
        plt.imsave(path, arr.transpose(1, 2, 0).clip(0, 1), vmin=0, vmax=1)

    _save(args.output_dir / f"{stem}_gen_{H}.png", generated)
    print(f"Saved: {args.output_dir / f'{stem}_gen_{H}.png'}")

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
