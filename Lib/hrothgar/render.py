"""Glyph rendering and normalization.

Single source of truth for everything that turns a font into a glyph image:

* low-level FreeType/Skia rendering (:func:`render_gid`, :func:`render_gid_raw`,
  :func:`render_phrase`)
* the shared crop-to-ink normalization policy (:func:`crop_to_ink`,
  :func:`normalize_bitmap`) and its geometry labels (``GEOMETRY_SPEC`` /
  ``GEOMETRY_NAMES``)
* the model-facing renderers used by the diffusion generator, the upscaler and
  the style embedder (:func:`render_glyph`,
  :func:`render_glyph_with_geometry`, :func:`render_gid_with_geometry`)

Glyphs are rendered as ink-on-white: ink near 0.0 on a white (1.0) background.
The crop-to-ink convention deliberately discards aspect ratio — the generator
predicts the five ``GEOMETRY_NAMES`` labels to place the glyph back onto the
baseline.
"""

from __future__ import annotations

import argparse
import ctypes
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import freetype
import numpy as np
import skia
import torch
import torch.nn.functional as F
import uharfbuzz as hb

if TYPE_CHECKING:
    from hrothgar.googlefonts import Font

# Ink is rendered near 0.0 on a white (1.0) background.
_INK_THRESHOLD = 0.5

# Glyphs are rendered at this multiple of the target size, then downscaled
# during crop-to-ink.  Rendering at the target size and *upscaling* back after
# cropping blurs corners (bilinear spreads the AA ramp); rendering high and
# *downscaling* preserves them.  4x was validated on the DMSerifText R counter.
RENDER_SUPERSAMPLE = 4


def render_size(size: int) -> int:
    """The raster resolution to render at for a given output ``size``."""
    return size * RENDER_SUPERSAMPLE


# Canonical geometry label order and em-unit ranges.  The factorized
# (codepoint, font-ID) diffusion model's geometry regression head predicts these
# five values in this order, using sigmoid (non-negative widths) or tanh
# (signed offsets) scaled by the em range below.  ``left_sidebearing`` and
# ``descender_depth`` are signed; the rest are non-negative.
GEOMETRY_SPEC = (
    # (name, activation, em-scale)
    ("scale_x", "sigmoid", 1.5),
    ("scale_y", "sigmoid", 1.2),
    ("left_sidebearing", "tanh", 1.0),
    # Descender depth = scale_y - baseline_offset: how far the ink hangs below
    # the baseline (overshoot, descender).  Small (mostly ~0), signed, and the
    # quantity that must be *exactly* right for baseline alignment.
    ("descender_depth", "tanh", 0.8),
    ("advance", "sigmoid", 1.5),
)
GEOMETRY_NAMES = tuple(name for name, _, _ in GEOMETRY_SPEC)

# Predicted descender depths within this (em) of zero are snapped to exactly
# zero, so text faces achieve absolute baseline alignment instead of a tiny
# continuous offset that never lands on 0.
DESCENDER_SNAP_EPSILON = 0.001


# ── Low-level FreeType rendering ────────────────────────────────────────────


def _bitmap_to_array(bitmap: Any) -> np.ndarray:
    """Convert a FreeType bitmap object to a 2D uint8 NumPy array."""
    rows = int(bitmap.rows)
    width = int(bitmap.width)
    if rows <= 0 or width <= 0:
        return np.zeros((0, 0), dtype=np.uint8)

    pitch = int(bitmap.pitch)
    pitch_abs = abs(pitch)
    raw_pointer = getattr(bitmap, "_FT_Bitmap", None)
    if raw_pointer is not None and getattr(raw_pointer, "buffer", None):
        # One C-level memcpy.  ``bitmap.buffer`` (the public property) instead
        # builds a Python list with one ctypes dereference *per byte*, which
        # costs ~1.5 ms per 128px glyph and dominates rendering time.
        data = ctypes.string_at(raw_pointer.buffer, rows * pitch_abs)
        flat = np.frombuffer(data, dtype=np.uint8)
    else:
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


@dataclass
class RawGlyph:
    """A rendered glyph's raw bitmap plus its placement metrics.

    All quantities are in *pixels* at the requested ``ppem``.  The conversion to
    font-independent em units (divide by ``ppem``) happens in the caller.
    """

    bitmap: np.ndarray  # (rows, width) uint8 coverage; 0 = no ink, 255 = full ink
    bitmap_left: int  # left sidebearing in px (signed; negative = overhang)
    bitmap_top: int  # top bearing in px, positive above the baseline
    advance_px: float  # advance width in px


@lru_cache(maxsize=128)
def _face_for_path(font_path: str, axis_position: tuple | None) -> freetype.Face:
    """Return a cached FreeType face, configured with the requested axis
    position (``None`` = default instance)."""
    face = freetype.Face(font_path)
    if axis_position:
        # freetype-py forwards this to FT_Set_Var_Design_Coordinates.
        face.set_var_design_coords([float(v) for v in axis_position])
    return face


def render_gid(
    font_path: str | Path,
    gid: int,
    size: int,
    trim_to_rsb: bool = False,
    axis_position: Sequence[float] | None = None,
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

    # Parsing a font file is expensive (especially large variable fonts), and
    # rendering pipelines call this once per glyph, so cache the face.  The
    # axis position is part of the key so a cached face never leaks design
    # coordinates from one call site to another.
    axis_tuple = tuple(axis_position) if axis_position is not None else None
    face = _face_for_path(str(font_path), axis_tuple)
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


def render_gid_raw(
    font_path: str | Path,
    gid: int,
    size: int,
    axis_position: Sequence[float] | None = None,
) -> RawGlyph:
    """Render a glyph to its raw FreeType bitmap without pasting onto a canvas.

    Unlike :func:`render_gid`, this does not crop to a fixed square, clip
    descenders at the baseline, or clip negative left sidebearings at ``x=0``.
    The caller receives the tightly-fit coverage bitmap plus its offsets, from
    which the ink bbox and baseline position can be recovered exactly.

    Returns:
        :class:`RawGlyph` with the coverage bitmap and per-glyph placement in
        pixels.  ``1 em == size`` pixels, so dividing by ``size`` yields em
        units (font-independent).
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if gid < 0:
        raise ValueError("gid must be non-negative")

    axis_tuple = tuple(axis_position) if axis_position is not None else None
    face = _face_for_path(str(font_path), axis_tuple)
    ppem = size
    face.set_pixel_sizes(0, ppem)
    try:
        face.load_glyph(
            gid,
            freetype.FT_LOAD_FLAGS["FT_LOAD_RENDER"]
            | freetype.FT_LOAD_FLAGS["FT_LOAD_NO_HINTING"],
        )
    except freetype.FT_Exception:
        # Very complex outlines can overflow FreeType's rasterizer at render
        # time (e.g. "raster overflow").  The glyph metrics (advance width)
        # are loaded before the render step, so preserve them and fall back to
        # a blank glyph — the caller's ``normalize_bitmap`` already treats an
        # empty bitmap as blank.
        return RawGlyph(
            bitmap=np.zeros((0, 0), dtype=np.uint8),
            bitmap_left=0,
            bitmap_top=0,
            advance_px=face.glyph.linearHoriAdvance / 65536.0,
        )

    glyph_slot = face.glyph
    return RawGlyph(
        bitmap=_bitmap_to_array(glyph_slot.bitmap),
        bitmap_left=int(glyph_slot.bitmap_left),
        bitmap_top=int(glyph_slot.bitmap_top),
        advance_px=glyph_slot.linearHoriAdvance / 65536.0,
    )


# ── Skia phrase rendering ───────────────────────────────────────────────────


def render_phrase(
    font_path: str | Path,
    phrase: str,
    size: int = 48,
    axis_position: Sequence[float] | None = None,
    width: int = 768,
    height: int = 128,
) -> np.ndarray:
    typeface = skia.Typeface.MakeFromFile(str(font_path), 0)
    # Positioning
    left_margin = 10
    baseline = int(height * 2.0 / 3.0)
    surface = skia.Surface(width, height)
    canvas = surface.getCanvas()
    canvas.clear(skia.ColorWHITE)
    if axis_position:
        axes = typeface.getVariationDesignParameters()
        coords = skia.FontArguments.VariationPosition.Coordinates()
        for i in range(min(len(axis_position), len(axes))):
            coords.append(
                skia.FontArguments.VariationPosition.Coordinate(
                    axes[i].tag, axis_position[i]
                )
            )
        varpos = skia.FontArguments.VariationPosition(coords)
        args = skia.FontArguments()
        args.setVariationDesignPosition(varpos)
        typeface = typeface.makeClone(args)

    paint = skia.Paint(AntiAlias=True, Color=skia.ColorBLACK)
    canvas.drawString(phrase, left_margin, baseline, skia.Font(typeface, size), paint)
    image = surface.makeImageSnapshot()
    return image.toarray()


# ── Shared crop-to-ink normalization policy ─────────────────────────────────


def render_glyph(
    font: Font,
    codepoint: int,
    size: int,
    axis_position: list[float] | None = None,
) -> torch.Tensor:
    """Render a glyph as a ``(3, size, size)`` float32 tensor in [0, 1].

    This is the full, uncropped render (the glyph is placed on the fixed square
    canvas).  The crop-to-ink variants (:func:`render_glyph_with_geometry`,
    :func:`crop_to_ink`) are what the models consume.
    """
    arr = font.render(codepoint, size=size, axis_position=axis_position)
    return torch.from_numpy(arr.copy())


def ink_bbox(rendering: torch.Tensor, size: int) -> torch.Tensor:
    """Return the ink bbox as normalized ``(x0, y0, x1, y1)`` in [0, 1].

    ``rendering`` may be ``(1, H, W)`` or ``(3, H, W)``; ink detection uses
    channel 0.  Blank glyphs return an all-zero bbox.
    """
    ink = rendering[0] < _INK_THRESHOLD
    if not ink.any():
        return torch.zeros(4, dtype=torch.float32, device=rendering.device)
    ys, xs = ink.nonzero(as_tuple=True)
    return (
        torch.tensor(
            [
                xs.min().float(),
                ys.min().float(),
                xs.max().float(),
                ys.max().float(),
            ],
            dtype=torch.float32,
            device=rendering.device,
        )
        / size
    )


def bbox_size(rendering: torch.Tensor, size: int) -> torch.Tensor:
    """Return the normalized ink bbox ``(width, height)`` in [0, 1]."""
    x0, y0, x1, y1 = ink_bbox(rendering, size)
    return torch.stack([x1 - x0, y1 - y0])


def crop_to_ink(rendering: torch.Tensor, size: int) -> torch.Tensor:
    """Crop a glyph to its ink bbox and stretch it to ``(C, size, size)``.

    Args:
        rendering: ``(C, H, W)`` float32 glyph image in [0, 1] (C = 1 or 3).
        size: side length of the output square.

    Returns:
        ``(C, size, size)`` float32. Blank glyphs (no ink) return a white
        square of ``size``.
    """
    if rendering.ndim != 3:
        raise ValueError(f"crop_to_ink expects (C, H, W), got {tuple(rendering.shape)}")

    channels = rendering.shape[0]
    ink = rendering[0] < _INK_THRESHOLD
    if not ink.any():
        return torch.ones(
            (channels, size, size), dtype=rendering.dtype, device=rendering.device
        )

    ys, xs = ink.nonzero(as_tuple=True)
    y0, y1 = int(ys.min().item()), int(ys.max().item())
    x0, x1 = int(xs.min().item()), int(xs.max().item())
    crop = rendering[:, y0 : y1 + 1, x0 : x1 + 1]  # (C, h, w)

    return F.interpolate(
        crop.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
    )[0]


def render_normalized(
    font: Font,
    codepoint: int,
    size: int,
    axis_position: list[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render + crop-to-ink a glyph.

    Renders at ``render_size(size)`` (supersampled) and downscales during
    crop-to-ink, so corners survive the AA.

    Returns:
        ``(normalized, bbox)`` where ``normalized`` is ``(3, size, size)`` and
        ``bbox`` is normalized ``(x0, y0, x1, y1)`` in [0, 1].
    """
    rsize = render_size(size)
    rendering = render_glyph(font, codepoint, rsize, axis_position=axis_position)
    return crop_to_ink(rendering, size), ink_bbox(rendering, rsize)


def geometry_tensor(geometry: dict[str, float]) -> torch.Tensor:
    """Pack a geometry dict into a ``(5,)`` float32 tensor in canonical order."""
    return torch.tensor(
        [geometry[name] for name in GEOMETRY_NAMES], dtype=torch.float32
    )


def normalize_bitmap(
    bitmap: np.ndarray,
    bitmap_left: int,
    bitmap_top: int,
    advance_px: float,
    size: int,
    ppem: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Crop a raw FreeType bitmap to its ink and normalize it to a square.

    ``bitmap`` is a ``(rows, width)`` uint8 coverage array (0 = no ink,
    255 = full ink).  ``bitmap_left``/``bitmap_top`` are the FreeType bitmap
    offsets in pixels (left sidebearing and top bearing); ``advance_px`` is the
    glyph advance width in pixels.

    ``size`` is the output square side length.  ``ppem`` is the raster
    resolution the bitmap was *rendered* at (used for the em-unit geometry;
    defaults to ``size``).  They differ when the glyph was rendered
    supersampled (``render_size``) and downscaled here.

    Returns:
        ``(image, geometry)`` where ``image`` is a ``(1, size, size)`` float32
        tensor in [0, 1] (0 = ink, 1 = white) and ``geometry`` is a dict of the
        five labels ``scale_x``, ``scale_y``, ``left_sidebearing``,
        ``descender_depth``, ``advance`` in em units.
    """
    ppem = ppem if ppem is not None else size
    if bitmap.size == 0:
        # Blank glyph (space, etc.): no ink, but the advance is still meaningful.
        image = torch.ones((1, size, size), dtype=torch.float32)
        geometry = {
            "scale_x": 0.0,
            "scale_y": 0.0,
            "left_sidebearing": 0.0,
            "descender_depth": 0.0,
            "advance": advance_px / ppem,
        }
        return image, geometry

    coverage = np.asarray(bitmap, dtype=np.float32) / 255.0  # 0..1, 1 = full ink
    ink = coverage > (1.0 - _INK_THRESHOLD)  # matches crop_to_ink's threshold

    if not ink.any():
        image = torch.ones((1, size, size), dtype=torch.float32)
        geometry = {
            "scale_x": 0.0,
            "scale_y": 0.0,
            "left_sidebearing": float(bitmap_left) / ppem,
            "descender_depth": -float(bitmap_top) / ppem,
            "advance": advance_px / ppem,
        }
        return image, geometry

    ys, xs = ink.nonzero()
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    # Ink as 0.0 on a white 1.0 background (matches the shared render convention).
    crop = 1.0 - coverage[y0 : y1 + 1, x0 : x1 + 1]  # (h, w) float32
    crop_t = (
        torch.from_numpy(np.ascontiguousarray(crop)).float().unsqueeze(0).unsqueeze(0)
    )
    image = F.interpolate(
        crop_t, size=(size, size), mode="bilinear", align_corners=False
    )[
        0
    ]  # (1, size, size)

    scale_y = (y1 - y0 + 1) / ppem
    baseline_offset = (bitmap_top - y0) / ppem
    geometry = {
        "scale_x": (x1 - x0 + 1) / ppem,
        "scale_y": scale_y,
        "left_sidebearing": (bitmap_left + x0) / ppem,
        "descender_depth": scale_y - baseline_offset,
        "advance": advance_px / ppem,
    }
    return image, geometry


def place_glyph(
    image: np.ndarray,
    geometry: Sequence[float],
    *,
    ppm: int = 128,
    ascender_em: float = 1.5,
    descender_em: float = 0.5,
    origin_x_em: float = 0.5,
) -> tuple[np.ndarray, int, int, int]:
    """Place a crop-to-ink glyph back onto a baseline canvas (inverse of
    :func:`normalize_bitmap`).

    Args:
        image: ``(H, W)`` float32 array in [0, 1], ink = 0, white = 1 — a
            crop-to-ink normalized square.
        geometry: the five em-unit labels in canonical order
            ``(scale_x, scale_y, left_sidebearing, descender_depth, advance)``.
        ppm: pixels per em (canvas resolution).
        ascender_em / descender_em: canvas space above / below the baseline.
        origin_x_em: X-origin offset in em from the left edge (room for
            negative left sidebearings).

    Returns:
        ``(canvas, origin_x_px, baseline_y_px, advance_x_px)`` where ``canvas``
        is a ``(H, W)`` float32 array in [0, 1] (ink = 0, white = 1) and the
        three pixel positions locate the X origin, baseline, and advance width.
    """
    scale_x, scale_y, lsb, descender_depth, advance = (float(v) for v in geometry)
    # Recover the absolute baseline position from the alignment-critical residual.
    baseline_offset = scale_y - descender_depth

    # Un-square the normalized glyph back to its true ink size.
    h_px = max(1, round(scale_y * ppm))
    w_px = max(1, round(scale_x * ppm))
    src = (
        torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))
        .unsqueeze(0)
        .unsqueeze(0)
    )  # (1, 1, H, W)
    glyph = F.interpolate(src, size=(h_px, w_px), mode="bilinear", align_corners=False)[
        0, 0
    ].numpy()  # (h_px, w_px)

    canvas_h = round((ascender_em + descender_em) * ppm)
    # 2.5em right of the origin covers the label maxima (LSB 1.0 + scale_x 1.5)
    # and advance (1.5em); 0.5em is reserved left of the origin for negative LSB.
    canvas_w = round((origin_x_em + 2.5) * ppm)
    origin_x = round(origin_x_em * ppm)
    baseline_y = round(ascender_em * ppm)
    advance_x = origin_x + round(advance * ppm)

    canvas = np.ones((canvas_h, canvas_w), dtype=np.float32)
    x0 = origin_x + round(lsb * ppm)
    y0 = baseline_y - round(baseline_offset * ppm)

    # Paste the glyph with clipping (blank or out-of-canvas glyphs are no-ops).
    gy1 = glyph.shape[0]
    gx1 = glyph.shape[1]
    cy0, cy1 = max(0, y0), min(canvas_h, y0 + gy1)
    cx0, cx1 = max(0, x0), min(canvas_w, x0 + gx1)
    if cy1 > cy0 and cx1 > cx0:
        canvas[cy0:cy1, cx0:cx1] = glyph[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]

    return canvas, origin_x, baseline_y, advance_x


# ── Model-facing crop-to-ink renderers ──────────────────────────────────────


def render_gid_with_geometry(
    font,
    gid: int,
    size: int,
    axis_position: Sequence[float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Render a glyph by GID, crop-to-ink, and return its geometry labels.

    Reads FreeType's raw bitmap and offsets directly (via
    :func:`render_gid_raw`), so descenders are not clipped at the baseline and
    negative left sidebearings are not clipped at ``x=0``.  This is the same
    normalize-to-square policy the factorized diffusion model uses, so a glyph
    consumed here occupies the same canonical square as the diffusion model's
    output.  The glyph is rendered supersampled (``render_size``) and downscaled
    during normalization, so corners survive the AA.

    Returns:
        ``(image, geometry)`` where ``image`` is a ``(size, size)`` greyscale
        tensor in [0, 1] (0 = ink, 1 = white) and ``geometry`` is a dict of the
        five em-unit labels ``scale_x``, ``scale_y``, ``left_sidebearing``,
        ``descender_depth``, ``advance``.
    """
    rsize = render_size(size)
    raw = render_gid_raw(font.path, gid, rsize, axis_position=axis_position)
    image, geometry = normalize_bitmap(
        raw.bitmap, raw.bitmap_left, raw.bitmap_top, raw.advance_px, size, ppem=rsize
    )
    return image[0], geometry


def render_glyph_with_geometry(
    font,
    codepoint: int,
    size: int,
    axis_position: Sequence[float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Render + crop-to-ink a glyph by codepoint, returning its geometry labels.

    See :func:`render_gid_with_geometry` for the normalization policy.
    """
    gid = hb.Font(font.hb_face).get_nominal_glyph(codepoint)
    return render_gid_with_geometry(font, gid, size, axis_position=axis_position)


# ── CLI ─────────────────────────────────────────────────────────────────────


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
