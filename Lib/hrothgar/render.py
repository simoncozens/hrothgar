"""Glyph rendering utilities based on FreeType.

This module renders glyphs by glyph ID (GID) and is intentionally independent
from Torch/data pipeline code so it can be tested in isolation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional, Sequence

import freetype
import numpy as np


def _bitmap_to_array(bitmap: Any) -> np.ndarray:
    """Convert a FreeType bitmap object to a 2D uint8 NumPy array."""
    rows = int(bitmap.rows)
    width = int(bitmap.width)
    if rows <= 0 or width <= 0:
        return np.zeros((0, 0), dtype=np.uint8)

    pitch = int(bitmap.pitch)
    pitch_abs = abs(pitch)
    buffer = bitmap.buffer
    if isinstance(buffer, (bytes, bytearray, memoryview)):
        flat = np.frombuffer(buffer, dtype=np.uint8)
    else:
        flat = np.asarray(buffer, dtype=np.uint8)
    if flat.size < rows * pitch_abs:
        return np.zeros((0, 0), dtype=np.uint8)

    arr = flat[: rows * pitch_abs].reshape(rows, pitch_abs)
    if pitch < 0:
        arr = arr[::-1]
    return arr[:, :width]


def _paste_bitmap_onto_canvas(
    canvas: np.ndarray,
    bitmap_array: np.ndarray,
    bitmap_left: int,
    bitmap_top: int,
    baseline_y: int,
) -> None:
    """Paste a glyph bitmap onto a canvas with baseline alignment.

    The X position uses bitmap_left (left sidebearing in pixels). This means
    negative sidebearings naturally clip at x < 0.
    """
    if bitmap_array.size == 0:
        return

    height, width = bitmap_array.shape
    dst_x0 = int(bitmap_left)
    dst_y0 = int(baseline_y - bitmap_top)
    dst_x1 = dst_x0 + width
    dst_y1 = dst_y0 + height

    src_x0 = max(0, -dst_x0)
    src_y0 = max(0, -dst_y0)
    src_x1 = width - max(0, dst_x1 - canvas.shape[1])
    src_y1 = height - max(0, dst_y1 - canvas.shape[0])

    if src_x0 >= src_x1 or src_y0 >= src_y1:
        return

    dst_x0_clamped = max(0, dst_x0)
    dst_y0_clamped = max(0, dst_y0)
    dst_x1_clamped = dst_x0_clamped + (src_x1 - src_x0)
    dst_y1_clamped = dst_y0_clamped + (src_y1 - src_y0)

    src = bitmap_array[src_y0:src_y1, src_x0:src_x1]
    # FreeType grayscale is coverage alpha. Composite as black ink on white.
    canvas[dst_y0_clamped:dst_y1_clamped, dst_x0_clamped:dst_x1_clamped] = 255 - src


# from matrix_disk_cache import MatrixDiskCache

# Initialize the cache with an optional maxsize
# cache = MatrixDiskCache(cache_dir="image_cache")


# @cache.cache
def render_gid(
    font_path: str | Path,
    gid: int,
    size: int,
    trim_to_rsb: bool = False,
    axis_position: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Render a glyph by GID into a square image.

    Args:
        font_path: Path to the font file.
        gid: Glyph index (GID) to render.
        size: Output image size. Output is (3, size, size).
        trim_to_rsb: If True, trim the output to the right sidebearing instead of the full square. This can be useful for certain applications but may produce variable-width outputs.
        axis_position: Optional in-order list of variable-font user-space
            design coordinates (matching fvar axis order).

    Returns:
        Float32 image in [0, 1], shaped (3, size, size).
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if gid < 0:
        raise ValueError("gid must be non-negative")

    face = freetype.Face(str(font_path))
    if axis_position:
        # freetype-py forwards this to FT_Set_Var_Design_Coordinates.
        face.set_var_design_coords([float(v) for v in axis_position])
    upem = int(face.units_per_EM)
    if upem <= 0:
        raise ValueError(f"Font has invalid units-per-em: {upem}")

    # Match the existing project convention: 1 upem ascent above baseline and
    # 0.5 upem descent below baseline in the output square.
    ppem = size
    face.set_pixel_sizes(0, ppem)
    face.load_glyph(
        gid,
        freetype.FT_LOAD_FLAGS["FT_LOAD_RENDER"]
        | freetype.FT_LOAD_FLAGS["FT_LOAD_NO_HINTING"],
        # Hinting failures can cause segfaults we can't catch
    )

    glyph_slot = face.glyph
    actual_width = glyph_slot.linearHoriAdvance / 65536
    bitmap_array = _bitmap_to_array(glyph_slot.bitmap)
    if trim_to_rsb:
        image = np.full((size, int(np.ceil(actual_width))), 255, dtype=np.uint8)
    else:
        image = np.full((size, size), 255, dtype=np.uint8)
    baseline_y = int(size * 1.0)
    _paste_bitmap_onto_canvas(
        canvas=image,
        bitmap_array=bitmap_array,
        bitmap_left=int(glyph_slot.bitmap_left),
        bitmap_top=int(glyph_slot.bitmap_top),
        baseline_y=baseline_y,
    )

    out = image.astype(np.float32) / 255.0
    return np.stack([out, out, out], axis=0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a font glyph by GID")
    parser.add_argument("font", type=Path, help="Path to font file")
    parser.add_argument("gid", type=int, help="Glyph ID to render")
    parser.add_argument("--size", type=int, default=128, help="Output size")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save rendering to a PNG file next to the font",
    )
    parser.add_argument(
        "--trim",
        action="store_true",
        help="Trim output width to the right sidebearing instead of the full square",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display rendering using matplotlib",
    )
    return parser.parse_args()


def main() -> None:
    """Debug entry point to render a single glyph and display it."""
    args = _parse_args()
    rendering = render_gid(args.font, args.gid, args.size, trim_to_rsb=args.trim)

    non_white_pixels = int((rendering[0] < 1.0).sum())
    print("Rendered non-white pixels:", non_white_pixels)

    if args.show:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(5, 5))
        plt.imshow(rendering[0], cmap="gray", vmin=0.0, vmax=1.0)
        plt.title(f"{args.font.name} GID={args.gid} size={args.size}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    if args.save:
        import matplotlib.pyplot as plt

        output_path = args.font.with_suffix(f".gid{args.gid}.png")
        plt.imsave(output_path, rendering[0], cmap="gray", vmin=0.0, vmax=1.0)
        print(f"Saved rendering to {output_path}")


if __name__ == "__main__":
    main()
