#!/usr/bin/env python3
"""CLI for testing the Core ML upscaler pipeline end-to-end.

Usage (after ``export_coreml.py``)::

    python -m hrothgar.upscaler.test_coreml \\
        MyFont.ttf --char A --model-dir models/coreml

Requirements:
    coremltools, freetype-py, matplotlib, numpy
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def _render_glyph(font_path: str, char: str, size: int) -> np.ndarray:
    """Render a glyph as a (3, size, size) float32 CHW array."""
    import freetype

    face = freetype.Face(font_path)
    face.set_pixel_sizes(size, size)
    face.load_char(char, freetype.FT_LOAD_RENDER)

    bitmap = face.glyph.bitmap
    buf = bitmap.buffer
    w, rows = bitmap.width, bitmap.rows

    raw = np.zeros((3, size, size), dtype=np.float32)
    for y in range(rows):
        for x in range(w):
            v = buf[y * w + x] / 255.0
            raw[:, y, x] = v
    return raw


def _render_style_references(
    font_path: str, count: int, size: int
) -> np.ndarray:
    """Render *count* style reference glyphs as (count, 3, size, size)."""
    reference_chars = "ABEGNRSTabdeghknpqy023456789"
    refs: list[np.ndarray] = []
    for c in reference_chars:
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

    return np.stack(refs[:count])  # (count, 3, size, size)


def _find_sidecar(model_dir: Path) -> Path:
    """Find the config sidecar in *model_dir*."""
    for name in model_dir.iterdir():
        if name.suffix == ".conf.json":
            return name
    raise FileNotFoundError(
        f"No .conf.json sidecar found in {model_dir}. "
        "Run export_coreml.py first."
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Test Core ML upscaler.")
    p.add_argument("font", type=Path, help="Path to a font file.")
    p.add_argument("--char", type=str, required=True, help="Character to upscale.")
    p.add_argument(
        "--model-dir", type=Path, default=Path("models/coreml"),
        help="Directory with exported Core ML models and config sidecar.",
    )
    p.add_argument(
        "--style-reference-count", type=int, default=None,
        help="Override the number of reference glyphs (default: from config).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("outputs/coreml_test"),
        help="Output directory for images.",
    )
    p.add_argument("--no-show", action="store_true", help="Skip matplotlib preview.")
    p.add_argument(
        "--disable-style", action="store_true",
        help="Skip style conditioning (uses fallback).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if not args.font.exists():
        raise FileNotFoundError(f"Font not found: {args.font}")

    from hrothgar.upscaler.model import UpscalerConfig
    from hrothgar.upscaler.inference_coreml import UpscalerInference

    # Read config from sidecar.
    sidecar_path = _find_sidecar(args.model_dir)
    config = UpscalerConfig.from_sidecar(sidecar_path)
    low_sz = config.low_res_size
    high_sz = config.high_res_size
    K = args.style_reference_count if args.style_reference_count is not None else config.style_reference_count
    print(f"Upscaler config: {low_sz}->{high_sz}, K={K}")

    print(f"Font: {args.font.name}")
    char = args.char
    if len(char) != 1:
        raise ValueError("--char must be a single Unicode character")
    print(f"Rendering '{char}' (U+{ord(char):04X}) ...")

    low_arr = _render_glyph(str(args.font), char, size=low_sz)
    native_arr = _render_glyph(str(args.font), char, size=high_sz)

    style_refs: Optional[np.ndarray] = None
    if not args.disable_style:
        style_refs = _render_style_references(
            str(args.font), count=K, size=high_sz
        )
        print(f"Rendered {style_refs.shape[0]} style references.")
    else:
        print("Style conditioning DISABLED (using fallback).")

    # Core ML inference.
    print(f"Loading Core ML models from {args.model_dir} ...")
    infer = UpscalerInference(args.model_dir)

    print("Running upscaler ...")
    upscaled_arr = infer.upscale(low_res=low_arr, style_references=style_refs)

    # Save & display.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.font.stem}_U+{ord(char):04X}"

    def _save(path: Path, arr: np.ndarray) -> None:
        plt.imsave(path, arr.transpose(1, 2, 0).clip(0, 1), vmin=0, vmax=1)

    _save(args.output_dir / f"{stem}_{low_sz}.png", low_arr)
    _save(args.output_dir / f"{stem}_{high_sz}_native.png", native_arr)
    _save(args.output_dir / f"{stem}_{high_sz}_coreml.png", upscaled_arr)

    print("Saved:")
    for label, name in [
        (f"{low_sz} input", f"{stem}_{low_sz}.png"),
        (f"{high_sz} ground-truth", f"{stem}_{high_sz}_native.png"),
        (f"{high_sz} Core ML", f"{stem}_{high_sz}_coreml.png"),
    ]:
        print(f"  {label:>20s}: {name}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (img, title) in zip(
        axes,
        [
            (low_arr, f"Native {low_sz}"),
            (upscaled_arr, f"Core ML {high_sz}"),
            (native_arr, f"Native {high_sz}"),
        ],
    ):
        ax.imshow(img[0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(f"{args.font.name} — {char} (U+{ord(char):04X})")
    fig.tight_layout()
    fig_path = args.output_dir / f"{stem}_comparison.png"
    fig.savefig(fig_path, dpi=150)
    print(f"  comparison figure: {fig_path.name}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
