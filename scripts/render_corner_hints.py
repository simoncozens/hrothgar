"""Render a glyph (no normalization) and its ground-truth corner hints.

This is the "guide the vectorizer with structure" experiment. It renders a
glyph into a **fixed frame** — no crop-to-ink, no stretch-to-square — so the
raster and the hint share one deterministic coordinate system::

    px = fx * size / upem
    py = baseline_y - fy * size / upem

where ``fx``/``fy`` are TrueType outline font units and ``baseline_y`` is a
fixed fraction of the frame (room for descenders). The corner hint is a
grayscale channel in the same frame: a Gaussian blob at each on-curve point
whose in/out tangents turn by more than ``--smooth-angle-deg`` (the same
~10-degree rule img2bez uses in ``compute_smooth``).

Outputs per glyph::

    <stem>_gray.png      the glyph raster (0 = ink, 1 = white)
    <stem>_corner.png    corner-blob hint channel
    <stem>_overlay.png   raster + corners (red) + smooth (green)
    <stem>_points.json   on-curve points (type + pixel + font-unit coords)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import freetype
import numpy as np

FT_LOAD_NO_SCALE = freetype.FT_LOAD_FLAGS["FT_LOAD_NO_SCALE"]
FT_LOAD_NO_HINTING = freetype.FT_LOAD_FLAGS["FT_LOAD_NO_HINTING"]
FT_CURVE_TAG_ON = 1
FT_CURVE_TAG_CONIC = 0
FT_CURVE_TAG_CUBIC = 2


def _parse_char(value: str) -> int:
    if value.startswith(("U+", "u+")):
        return int(value[2:], 16)
    if len(value) != 1:
        raise ValueError("--char must be a single Unicode character or U+XXXX codepoint")
    return ord(value)


def _parse_axis(value: str | None) -> list[float] | None:
    if not value:
        return None
    return [float(v) for v in value.split(",")]


def _load_face(path: Path, axis_position: list[float] | None) -> freetype.Face:
    face = freetype.Face(str(path))
    if axis_position:
        face.set_var_design_coords(axis_position)
    return face


def _raw_bitmap(
    face: freetype.Face, gid: int, size: int
) -> tuple[np.ndarray, int, int]:
    """Rasterize ``gid`` at ``ppem = size``; return ``(coverage, left, top)``."""
    face.set_pixel_sizes(0, size)
    face.load_glyph(gid, FT_LOAD_NO_HINTING | freetype.FT_LOAD_FLAGS["FT_LOAD_RENDER"])
    slot = face.glyph
    bitmap = slot.bitmap
    rows, width = int(bitmap.rows), int(bitmap.width)
    pitch = int(bitmap.pitch)
    if rows <= 0 or width <= 0:
        return np.zeros((0, 0), dtype=np.uint8), int(slot.bitmap_left), int(slot.bitmap_top)
    buffer = bitmap.buffer
    if isinstance(buffer, (bytes, bytearray, memoryview)):
        flat = np.frombuffer(buffer, dtype=np.uint8)
    else:
        flat = np.asarray(buffer, dtype=np.uint8)
    flat = flat[: rows * abs(pitch)].reshape(rows, abs(pitch))
    if pitch < 0:
        flat = flat[::-1]
    return flat[:, :width].copy(), int(slot.bitmap_left), int(slot.bitmap_top)


def _paste_fixed(
    canvas: np.ndarray,
    coverage: np.ndarray,
    left: int,
    top: int,
    baseline_y: int,
) -> None:
    """Paste a FreeType coverage bitmap into the fixed frame (ink = 255 - cov)."""
    if coverage.size == 0:
        return
    rows, width = coverage.shape
    dst_x0 = left
    dst_y0 = baseline_y - top
    dst_x1 = dst_x0 + width
    dst_y1 = dst_y0 + rows

    src_x0 = max(0, -dst_x0)
    src_y0 = max(0, -dst_y0)
    src_x1 = width - max(0, dst_x1 - canvas.shape[1])
    src_y1 = rows - max(0, dst_y1 - canvas.shape[0])
    if src_x0 >= src_x1 or src_y0 >= src_y1:
        return

    dst_x0c = max(0, dst_x0)
    dst_y0c = max(0, dst_y0)
    dst_x1c = dst_x0c + (src_x1 - src_x0)
    dst_y1c = dst_y0c + (src_y1 - src_y0)
    src = coverage[src_y0:src_y1, src_x0:src_x1]
    canvas[dst_y0c:dst_y1c, dst_x0c:dst_x1c] = 255 - src


def _render_fixed(
    face: freetype.Face, gid: int, size: int, baseline_frac: float
) -> tuple[np.ndarray, int]:
    """Render a glyph into a fixed ``size`` frame; return ``(gray, baseline_y)``.

    ``gray`` is ``(size, size)`` float32 in ``[0, 1]`` (0 = ink, 1 = white).
    The baseline sits at ``baseline_frac * size`` from the top, so descenders
    have room and no ink-bbox dependence enters the transform.
    """
    coverage, left, top = _raw_bitmap(face, gid, size)
    baseline_y = int(baseline_frac * size)
    canvas = np.full((size, size), 255, dtype=np.uint8)
    _paste_fixed(canvas, coverage, left, top, baseline_y)
    return canvas.astype(np.float32) / 255.0, baseline_y


def _outline_contours(face: freetype.Face, gid: int) -> list[list[tuple[float, float, int]]]:
    """Per-contour ``(x, y, tag)`` triples in font units (implied on-curves inserted)."""
    face.load_glyph(gid, FT_LOAD_NO_SCALE | FT_LOAD_NO_HINTING)
    outline = face.glyph.outline
    points = [(float(x), float(y)) for x, y in outline.points]
    tags = [int(t) for t in outline.tags]

    contours: list[list[tuple[float, float, int]]] = []
    start = 0
    for end in outline.contours:
        raw = [(points[i][0], points[i][1], tags[i]) for i in range(start, end + 1)]
        expanded: list[tuple[float, float, int]] = []
        n = len(raw)
        for i in range(n):
            cur = raw[i]
            expanded.append(cur)
            nxt = raw[(i + 1) % n]
            if cur[2] == FT_CURVE_TAG_CONIC and nxt[2] == FT_CURVE_TAG_CONIC:
                expanded.append(
                    ((cur[0] + nxt[0]) / 2.0, (cur[1] + nxt[1]) / 2.0, FT_CURVE_TAG_ON)
                )
        contours.append(expanded)
        start = end + 1
    return contours


def _classify_points(
    contours: list[list[tuple[float, float, int]]],
    smooth_angle_deg: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Split on-curve points into ``(corners, smooth)`` by tangent turn angle.

    Mirrors img2bez ``compute_smooth``: smooth iff ``|cross| < sin(angle)`` and
    the tangents are codirectional (dot > 0).
    """
    sin_tol = math.sin(math.radians(smooth_angle_deg))
    corners: list[tuple[float, float]] = []
    smooth: list[tuple[float, float]] = []
    for contour in contours:
        n = len(contour)
        for i in range(n):
            x, y, tag = contour[i]
            if tag != FT_CURVE_TAG_ON:
                continue
            px, py, _ = contour[(i - 1) % n]
            nx, ny, _ = contour[(i + 1) % n]
            ix, iy = x - px, y - py
            ox, oy = nx - x, ny - y
            il = math.hypot(ix, iy)
            ol = math.hypot(ox, oy)
            if il < 0.01 or ol < 0.01:
                corners.append((x, y))
                continue
            cross = (ix / il) * (oy / ol) - (iy / il) * (ox / ol)
            dot = (ix / il) * (ox / ol) + (iy / il) * (oy / ol)
            if abs(cross) < sin_tol and dot > 0.0:
                smooth.append((x, y))
            else:
                corners.append((x, y))
    return corners, smooth


def _map_fixed(
    points: list[tuple[float, float]], *, size: int, upem: int, baseline_y: int
) -> list[tuple[float, float]]:
    """Map font-unit points into the fixed frame's pixel space."""
    scale = size / upem
    return [(fx * scale, baseline_y - fy * scale) for fx, fy in points]


def _paint_gaussian(canvas: np.ndarray, cx: float, cy: float, radius: float) -> None:
    sigma = radius / 2.0
    k = int(math.ceil(radius * 2.5))
    size = canvas.shape[0]
    gx0 = max(0, int(round(cx)) - k)
    gy0 = max(0, int(round(cy)) - k)
    gx1 = min(size, int(round(cx)) + k + 1)
    gy1 = min(size, int(round(cy)) + k + 1)
    if gx1 <= gx0 or gy1 <= gy0:
        return
    yy, xx = np.mgrid[gy0:gy1, gx0:gx1]
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
    np.maximum(canvas[gy0:gy1, gx0:gx1], g, out=canvas[gy0:gy1, gx0:gx1])


def _paint_hint(
    size: int, corner_px: list[tuple[float, float]], blob_radius: float
) -> np.ndarray:
    hint = np.zeros((size, size), dtype=np.float32)
    for (cx, cy) in corner_px:
        _paint_gaussian(hint, cx, cy, blob_radius)
    return hint


def _overlay(gray: np.ndarray, corners: list[tuple[float, float]], smooth: list[tuple[float, float]]) -> np.ndarray:
    h, w = gray.shape
    rgb = np.stack([gray, gray, gray], axis=-1)
    for (cx, cy) in corners:
        x, y = int(round(cx)), int(round(cy))
        if 0 <= y < h and 0 <= x < w:
            rgb[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3] = [1.0, 0.0, 0.0]
    for (cx, cy) in smooth:
        x, y = int(round(cx)), int(round(cy))
        if 0 <= y < h and 0 <= x < w:
            rgb[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3] = [0.0, 1.0, 0.0]
    return rgb


def _save(path: Path, array: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim == 3:
        img = Image.fromarray((array * 255.0).clip(0, 255).astype(np.uint8), mode="RGB")
    else:
        img = Image.fromarray((array * 255.0).clip(0, 255).astype(np.uint8), mode="L")
    img.save(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a glyph (no normalization) and ground-truth corner hints."
    )
    parser.add_argument("font", type=Path, help="Path to a font file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gid", type=int, help="Glyph ID to render")
    group.add_argument("--char", type=str, help="Unicode character or U+XXXX codepoint")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/corner_hints"))
    parser.add_argument("--size", type=int, default=128, help="Frame size in px (default 128)")
    parser.add_argument("--baseline-frac", type=float, default=0.8, help="Baseline as fraction of frame height")
    parser.add_argument("--axis", type=str, default=None, help="Comma-separated design coords (variable fonts)")
    parser.add_argument("--blob-radius", type=float, default=4.0, help="Corner blob radius px")
    parser.add_argument(
        "--smooth-angle-deg", type=float, default=10.0,
        help="Turn angle below which an on-curve point counts as smooth",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not args.font.exists():
        raise FileNotFoundError(f"Font file not found: {args.font}")

    axis = _parse_axis(args.axis)
    face = _load_face(args.font, axis)
    upem = int(face.units_per_EM)

    if args.gid is not None:
        gid = args.gid
        label = f"gid_{gid}"
    else:
        codepoint = _parse_char(args.char)
        import uharfbuzz as hb

        gid = hb.Font(hb.Face(hb.Blob.from_file_path(str(args.font)))).get_nominal_glyph(
            codepoint
        )
        label = f"cp_{codepoint:04X}"

    gray, baseline_y = _render_fixed(face, gid, args.size, args.baseline_frac)

    contours = _outline_contours(face, gid)
    corners_units, smooth_units = _classify_points(contours, args.smooth_angle_deg)
    corner_px = _map_fixed(corners_units, size=args.size, upem=upem, baseline_y=baseline_y)
    smooth_px = _map_fixed(smooth_units, size=args.size, upem=upem, baseline_y=baseline_y)

    hint = _paint_hint(args.size, corner_px, args.blob_radius)
    overlay = _overlay(gray, corner_px, smooth_px)

    axis_suffix = f"_a{'_'.join(f'{v:g}' for v in axis)}" if axis else ""
    stem = f"{args.font.stem}_{label}_s{args.size}{axis_suffix}"
    out = args.output_dir
    _save(out / f"{stem}_gray.png", gray)
    _save(out / f"{stem}_corner.png", hint)
    _save(out / f"{stem}_overlay.png", overlay)

    points = {
        "font": str(args.font),
        "gid": gid,
        "label": label,
        "upem": upem,
        "size": args.size,
        "baseline_y": baseline_y,
        "smooth_angle_deg": args.smooth_angle_deg,
        "n_corners": len(corner_px),
        "n_smooth": len(smooth_px),
        "corners_px": [[round(x, 2), round(y, 2)] for x, y in corner_px],
        "smooth_px": [[round(x, 2), round(y, 2)] for x, y in smooth_px],
        "corners_font_units": [[round(x, 1), round(y, 1)] for x, y in corners_units],
        "smooth_font_units": [[round(x, 1), round(y, 1)] for x, y in smooth_units],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}_points.json").write_text(json.dumps(points, indent=2), encoding="utf-8")

    print(f"gid={gid} upem={upem} size={args.size} baseline_y={baseline_y} "
          f"corners={len(corner_px)} smooth={len(smooth_px)}")
    for name in ("gray", "corner", "overlay", "points.json"):
        print(f"  {out / f'{stem}_{name}' if name != 'points.json' else out / f'{stem}_points.json'}")


if __name__ == "__main__":
    main()
