"""Glyph rendering utilities based on FreeType.

This module renders glyphs by glyph ID (GID) and is intentionally independent
from Torch/data pipeline code so it can be tested in isolation.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import freetype
import numpy as np
from PIL import Image

__all__ = [
    "_bitmap_to_array",
    "_paste_bitmap_onto_canvas",
    "_fit_bitmap_to_canvas",
    "_glyph_bounds_in_font_units",
    "_estimate_ppem_for_canvas",
    "render_gid",
    "is_blank_rendering",
]


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


def _fit_bitmap_to_canvas(
    bitmap_array: np.ndarray,
    size: int,
    trim_to_rsb: bool,
    border: int = 1,
    allow_upscale: bool = False,
) -> np.ndarray:
    """Scale a glyph bitmap to fill the canvas while preserving aspect ratio."""
    if bitmap_array.size == 0:
        width = size if not trim_to_rsb else max(1, 2 * border + 1)
        return np.full((size, width), 255, dtype=np.uint8)

    src_height, src_width = bitmap_array.shape
    available_height = max(1, size - 2 * border)
    if trim_to_rsb:
        scale = available_height / src_height
        target_height = available_height
        target_width = max(1, int(round(src_width * scale)))
        canvas_width = target_width + 2 * border
    else:
        available_width = max(1, size - 2 * border)
        scale = min(available_width / src_width, available_height / src_height)
        target_width = max(1, min(available_width, int(round(src_width * scale))))
        target_height = max(1, min(available_height, int(round(src_height * scale))))
        canvas_width = size

    if not allow_upscale and scale > 1.0:
        raise ValueError("bitmap would need upscaling to fit target canvas")

    if target_width == src_width and target_height == src_height:
        resized_bitmap = bitmap_array
    else:
        resized_bitmap = np.asarray(
            Image.fromarray(bitmap_array).resize(
                (target_width, target_height),
                resample=Image.Resampling.LANCZOS,
            ),
            dtype=np.uint8,
        )

    canvas = np.full((size, canvas_width), 255, dtype=np.uint8)
    offset_x = max(border, (canvas_width - target_width) // 2)
    offset_y = max(border, (size - target_height) // 2)
    canvas[offset_y : offset_y + target_height, offset_x : offset_x + target_width] = (
        255 - resized_bitmap
    )
    return canvas


def _glyph_bounds_in_font_units(face: freetype.Face, gid: int) -> tuple[int, int]:
    """Return glyph bounds in font units for ppem estimation."""
    face.load_glyph(
        gid,
        freetype.FT_LOAD_FLAGS["FT_LOAD_NO_SCALE"]
        | freetype.FT_LOAD_FLAGS["FT_LOAD_NO_HINTING"],
    )
    outline = face.glyph.outline
    if outline.n_points > 0:
        bbox = outline.get_bbox()
        width = max(0, int(bbox.xMax) - int(bbox.xMin))
        height = max(0, int(bbox.yMax) - int(bbox.yMin))
    else:
        metrics = face.glyph.metrics
        width = max(0, int(metrics.width))
        height = max(0, int(metrics.height))
    return width, height


def _estimate_ppem_for_canvas(
    face: freetype.Face,
    gid: int,
    size: int,
    trim_to_rsb: bool,
    border: int = 1,
) -> int:
    """Choose a ppem that should produce a bitmap large enough to downsample."""
    width_units, height_units = _glyph_bounds_in_font_units(face, gid)
    if width_units <= 0 or height_units <= 0:
        return max(1, int(round(size / 1.5)))

    upem = int(face.units_per_EM)
    available_height = max(1, size - 2 * border)
    height_ppem = math.ceil(available_height * upem / height_units)
    if trim_to_rsb:
        return max(1, height_ppem)

    available_width = max(1, size - 2 * border)
    width_ppem = math.ceil(available_width * upem / width_units)
    return max(1, height_ppem, width_ppem)


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
        size: Output image height. The glyph is scaled to fit within the
            canvas with a 1-pixel border while preserving aspect ratio.
        trim_to_rsb: If True, return a variable-width output cropped to the
            fitted glyph width plus border padding on each side.
        axis_position: Optional in-order list of variable-font user-space
            design coordinates (matching fvar axis order).

    Returns:
        Float32 image in [0, 1], shaped (3, size, width).
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

    ppem = _estimate_ppem_for_canvas(face, gid, size, trim_to_rsb=trim_to_rsb)
    for ppem_adjustment in range(4):
        face.set_pixel_sizes(0, ppem + ppem_adjustment)
        face.load_glyph(
            gid,
            freetype.FT_LOAD_FLAGS["FT_LOAD_RENDER"]
            | freetype.FT_LOAD_FLAGS["FT_LOAD_NO_HINTING"],
            # Hinting failures can cause segfaults we can't catch
        )

        bitmap_array = _bitmap_to_array(face.glyph.bitmap)
        try:
            image = _fit_bitmap_to_canvas(
                bitmap_array,
                size=size,
                trim_to_rsb=trim_to_rsb,
                allow_upscale=False,
            )
            break
        except ValueError:
            if ppem_adjustment == 3:
                raise RuntimeError("unable to rasterize glyph without upscaling")
    else:
        raise RuntimeError("unreachable ppem selection failure")

    out = image.astype(np.float32) / 255.0
    return np.stack([out, out, out], axis=0)


def is_blank_rendering(rendered) -> bool:
    """Return True when a rendered image is uniformly white or black."""
    max_val = float(rendered.max())
    min_val = float(rendered.min())
    # Real blank glyph rasters are typically all-white (1.0) and occasionally
    # all-black (0.0) in this pipeline.
    return max_val == min_val and (max_val == 1.0 or max_val == 0.0)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the render-module debug entry point."""
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
