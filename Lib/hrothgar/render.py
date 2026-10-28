"""Glyph rendering.

Single source of truth for everything that turns a font into a glyph image.  All
glyph rendering goes through one entry point, :func:`render`, which takes either
a ``codepoint`` or a ``gid`` plus a :class:`Frame` (or the :data:`NORMALIZED`
sentinel) and returns the image together with its five ``GEOMETRY_NAMES``
em-unit labels:

* ``render(font, codepoint=cp, size=size, frame=frame)`` — uniform scale into
  the em-space :class:`Frame` (aspect ratio kept, ink outside the frame clipped).
* ``render(font, codepoint=cp, size=size, frame=NORMALIZED)`` — crop-to-ink,
  stretch to a square (aspect ratio discarded); the framing the diffusion
  generator and the upscaler consume.

Low-level helpers are private: ``_render_gid_raw`` (the FreeType bitmap, no
canvas) and ``_normalize_bitmap`` (crop a raw bitmap to ink).  :func:`render_phrase`
(Skia) renders a string, not a glyph.  The inverse direction — putting a
crop-to-ink glyph back onto the baseline or into a frame — lives in
:func:`place_glyph`, :func:`denormalize_uniform` and :func:`crop_placed_to_frame`.
:func:`frame_for_codepoints` frames a font by its own ink envelope.

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
from typing import Any

import freetype
import numpy as np
import skia
import uharfbuzz as hb

import torch
import torch.nn.functional as F

# Ink is rendered near 0.0 on a white (1.0) background.
_INK_THRESHOLD = 0.5

# Glyphs are rendered at this multiple of the target size, then downscaled
# during crop-to-ink.  Rendering at the target size and *upscaling* back after
# cropping blurs corners (bilinear spreads the AA ramp); rendering high and
# *downscaling* preserves them.  4x was validated on the DMSerifText R counter.
RENDER_SUPERSAMPLE = 4


def _render_size(size: int) -> int:
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
class _RawGlyph:
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


def _render_gid_raw(
    font_path: str | Path,
    gid: int,
    size: int,
    axis_position: Sequence[float] | None = None,
) -> _RawGlyph:
    """Render a glyph to its raw FreeType bitmap without pasting onto a canvas.

    The caller receives the tightly-fit coverage bitmap plus its offsets, from
    which the ink bbox and baseline position can be recovered exactly.  This is
    the primitive every framing in :func:`render` builds on.

    Returns:
        :class:`_RawGlyph` with the coverage bitmap and per-glyph placement in
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
        # a blank glyph — the caller's ``_normalize_bitmap`` already treats an
        # empty bitmap as blank.
        return _RawGlyph(
            bitmap=np.zeros((0, 0), dtype=np.uint8),
            bitmap_left=0,
            bitmap_top=0,
            advance_px=face.glyph.linearHoriAdvance / 65536.0,
        )

    glyph_slot = face.glyph
    return _RawGlyph(
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


def geometry_tensor(geometry: dict[str, float]) -> torch.Tensor:
    """Pack a geometry dict into a ``(5,)`` float32 tensor in canonical order."""
    return torch.tensor(
        [geometry[name] for name in GEOMETRY_NAMES], dtype=torch.float32
    )


def _geometry_from_raw(raw: _RawGlyph, ppem: int) -> dict[str, float]:
    """Compute the five em-unit geometry labels from a raw FreeType bitmap.

    Shared by the crop-to-ink path (:func:`_normalize_bitmap`) and the
    uniform-scale path (:func:`render`), so the labels stay identical no matter
    which framing the image uses.  ``ppem`` is the resolution the glyph was
    rendered at (1 em == ``ppem`` pixels), so dividing pixel offsets by it
    yields em units.
    """
    if raw.bitmap.size == 0:
        # Blank glyph (space, etc.): no ink, but the advance is still meaningful.
        return {
            "scale_x": 0.0,
            "scale_y": 0.0,
            "left_sidebearing": 0.0,
            "descender_depth": 0.0,
            "advance": raw.advance_px / ppem,
        }

    coverage = np.asarray(raw.bitmap, dtype=np.float32) / 255.0  # 0..1, 1 = full ink
    ink = coverage > (1.0 - _INK_THRESHOLD)  # matches the shared ink threshold

    if not ink.any():
        return {
            "scale_x": 0.0,
            "scale_y": 0.0,
            "left_sidebearing": float(raw.bitmap_left) / ppem,
            "descender_depth": -float(raw.bitmap_top) / ppem,
            "advance": raw.advance_px / ppem,
        }

    ys, xs = ink.nonzero()
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    scale_y = (y1 - y0 + 1) / ppem
    baseline_offset = (raw.bitmap_top - y0) / ppem
    return {
        "scale_x": (x1 - x0 + 1) / ppem,
        "scale_y": scale_y,
        "left_sidebearing": (raw.bitmap_left + x0) / ppem,
        "descender_depth": scale_y - baseline_offset,
        "advance": raw.advance_px / ppem,
    }


def _normalize_bitmap(
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
    supersampled (``_render_size``) and downscaled here.

    Returns:
        ``(image, geometry)`` where ``image`` is a ``(1, size, size)`` float32
        tensor in [0, 1] (0 = ink, 1 = white) and ``geometry`` is a dict of the
        five labels ``scale_x``, ``scale_y``, ``left_sidebearing``,
        ``descender_depth``, ``advance`` in em units.
    """
    ppem = ppem if ppem is not None else size
    raw = _RawGlyph(
        bitmap=np.asarray(bitmap, dtype=np.uint8),
        bitmap_left=int(bitmap_left),
        bitmap_top=int(bitmap_top),
        advance_px=float(advance_px),
    )
    geometry = _geometry_from_raw(raw, ppem)

    if raw.bitmap.size == 0:
        # Blank glyph (space, etc.): no ink, but the advance is still meaningful.
        image = torch.ones((1, size, size), dtype=torch.float32)
        return image, geometry

    coverage = np.asarray(raw.bitmap, dtype=np.float32) / 255.0  # 0..1, 1 = full ink
    ink = coverage > (1.0 - _INK_THRESHOLD)  # matches the shared ink threshold

    if not ink.any():
        image = torch.ones((1, size, size), dtype=torch.float32)
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
    """Place a crop-to-ink glyph back onto a baseline canvas (inverse of the
    crop-to-ink normalization).

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


def crop_placed_to_frame(
    placed,
    frame: Frame,
    *,
    size: int = 256,
    ppm: int = 128,
    ascender_em: float = 1.5,
    origin_x_em: float = 0.5,
) -> np.ndarray:
    """Re-frame a :func:`place_glyph` canvas into ``frame``, resized to ``size``.

    A placed canvas is already in em space — baseline ``ascender_em`` em from
    the top, X origin ``origin_x_em`` em from the left, ``1 em == ppm`` pixels —
    so a tighter frame is just a crop of that canvas, no re-render.  The result
    lines up with :func:`render` for the same glyph and ``size``, so
    a generated glyph re-framed here matches GT rendered there (up to the
    geometry predictor's error).  Frames that extend past the placed canvas are
    clipped; for the tight 1.40em frame that never happens (it sits inside the
    2em window).

    Args:
        placed: ``(H, W)`` float32 array in [0, 1] (ink = 0, white = 1), a
            :func:`place_glyph` output.
        frame: the em rectangle to show, as a :class:`Frame`.
        size: output square side length (matches ``render``).
        ppm / ascender_em / origin_x_em: the ``place_glyph`` conventions the
            canvas was produced with.
    """
    span = frame.em_span
    pad_x = (span - frame.width) / 2.0
    pad_y = (span - frame.height) / 2.0
    origin_x = origin_x_em * ppm
    baseline_y = ascender_em * ppm
    x0 = origin_x + (frame.left - pad_x) * ppm
    x1 = origin_x + (frame.right + pad_x) * ppm
    y0 = baseline_y - (frame.top + pad_y) * ppm
    y1 = baseline_y - (frame.bottom - pad_y) * ppm

    arr = np.asarray(placed, dtype=np.float32)
    h, w = arr.shape
    l = max(0, int(round(x0)))
    r = min(w, int(round(x1)))
    t = max(0, int(round(y0)))
    b = min(h, int(round(y1)))
    if r <= l or b <= t:  # degenerate: frame entirely outside the canvas
        return np.ones((size, size), dtype=np.float32)

    crop = arr[t:b, l:r]
    src = torch.from_numpy(np.ascontiguousarray(crop, dtype=np.float32))[None, None]
    out = F.interpolate(src, size=(size, size), mode="bilinear", align_corners=False)
    return out[0, 0].numpy()


# ── Framing ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Frame:
    """The em-space rectangle a uniform render draws into the square canvas.

    The four values are the frame's edges as signed em offsets from the glyph
    origin (x = 0) and the baseline (y = 0, positive up), so the frame spans
    ``x in [left, right]`` and ``y in [bottom, top]``.  The legacy 2em framing
    (:data:`UNIFORM_FRAME`) is ``Frame(-0.5, 1.5, -0.5, 1.5)``: 0.5em of left
    margin, 1.5em of ascender room, 0.5em of descender room, and a right edge
    1.5em to the right of the origin.

    A frame need not be square.  The renderer keeps a single uniform scale (so
    the aspect ratio that content distinctions such as ``o``/``O`` depend on
    survives), fitted to the frame's larger dimension, and centres the frame in
    the canvas.
    """

    left: float = -0.5
    right: float = 1.5
    bottom: float = -0.5
    top: float = 1.5

    def __post_init__(self) -> None:
        if not self.left < self.right or not self.bottom < self.top:
            raise ValueError(
                "Frame needs left < right and bottom < top, got "
                f"left={self.left} right={self.right} "
                f"bottom={self.bottom} top={self.top}"
            )

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.top - self.bottom

    @property
    def em_span(self) -> float:
        """Em covered by one canvas side: the frame's larger dimension."""
        return max(self.width, self.height)

    @classmethod
    def from_extents(
        cls,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        *,
        margin: float = 0.0,
        min_size: float = 0.0,
    ) -> Frame:
        """The frame enclosing an ink box in em, optionally grown and floored.

        ``margin`` grows each dimension by that fraction (so 0.08 gives 8% more
        room); ``min_size`` floors each dimension in em, so a degenerate box —
        a blank glyph, a single inkless codepoint — cannot hand the renderer a
        zero span.
        """
        width = max(x_max - x_min, min_size)
        height = max(y_max - y_min, min_size)
        centre_x = (x_min + x_max) / 2.0
        centre_y = (y_min + y_max) / 2.0
        half_x = width * (1.0 + margin) / 2.0
        half_y = height * (1.0 + margin) / 2.0
        return cls(
            centre_x - half_x,
            centre_x + half_x,
            centre_y - half_y,
            centre_y + half_y,
        )

    def units_to_pixels(
        self, size: float, x: float, y: float, *, upem: float = 1.0
    ) -> tuple[float, float]:
        """Map font coordinates to pixels on a ``size``-square render.

        ``x``/``y`` are em units unless ``upem`` is given: pass the font's
        units-per-em to give them in design units, so 243 at ``upem=1000``
        means 0.243 em.  ``size`` is the side of the image :func:`render`
        returned.

        The canvas shows ``em_span`` em per side and centres the frame in it, so
        the slack on the frame's shorter axis is padding that shifts the origin
        away from the frame edge.  Pixel ``(0, 0)`` is the canvas's top-left
        corner and both axes grow rightwards/downwards, so a positive ``y``
        (above the baseline) maps to a smaller pixel row.

        The mapping is continuous; the raster rounds the origin to the
        supersampled canvas, so expect up to half a supersampled pixel of
        disagreement with the actual ink.
        """
        scale = size / self.em_span
        pad_x = (self.em_span - self.width) / 2.0
        pad_y = (self.em_span - self.height) / 2.0
        px = (pad_x + x / upem - self.left) * scale
        py = (pad_y + self.top - y / upem) * scale
        return px, py


# The canonical 2em framing, and what every existing caller gets.
UNIFORM_FRAME = Frame()


class _Normalized:
    """Sentinel for the crop-to-ink framing (pass :data:`NORMALIZED` to
    :func:`render`).

    Not a :class:`Frame` — it carries no em rectangle — so it cannot be passed
    to the uniform path or its inverse (:func:`denormalize_uniform`).
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "NORMALIZED"


# The crop-to-ink framing: discard aspect ratio and stretch to a square.
NORMALIZED = _Normalized()


# How much to grow a font's ink envelope when framing that font.  The measured
# outlines are the *default instance* (``glyf``/``CFF`` store one), so a render
# at an axis extreme (``wght``/``opsz``/``GRAD``) grows past the envelope.
ENVELOPE_FRAME_MARGIN = 0.08


# Smaller than this (em) and a frame cannot hold a legible glyph anyway; it
# only guards the division in the renderer against degenerate envelopes.
MIN_FRAME_SIZE = 0.2


def frame_for_codepoints(
    font,
    codepoints: Sequence[int],
    *,
    margin: float = ENVELOPE_FRAME_MARGIN,
    min_size: float = MIN_FRAME_SIZE,
) -> Frame:
    """The ink envelope of ``codepoints`` in ``font``, as a :class:`Frame` (em).

    This is the per-font frame that makes a letter occupy comparable pixels in
    every font.  It is measured once from the font's **default instance**, so it
    is a per-font *constant*: every glyph of the font is drawn into the same
    rectangle, which is what preserves the within-font relative sizes that
    content distinctions depend on.  Unlike crop-to-ink it is not per-glyph.

    Fonts whose codepoints render blank fall back to :data:`UNIFORM_FRAME`.
    """
    hb_face = font.hb_face
    upem = hb_face.upem or 1000
    hb_font = hb.Font(hb_face)
    hb_font.scale = (upem, upem)  # extents in font units

    x_min = y_min = float("inf")
    x_max = y_max = float("-inf")
    for codepoint in codepoints:
        gid = hb_font.get_nominal_glyph(codepoint)
        if gid is None:
            continue
        extents = hb_font.get_glyph_extents(gid)
        if extents is None or (extents.width == 0 and extents.height == 0):
            continue  # blank glyph (space, missing ink): nothing to frame
        x = extents.x_bearing / upem
        y = extents.y_bearing / upem
        x_min = min(x_min, x)
        x_max = max(x_max, x + extents.width / upem)
        y_max = max(y_max, y)
        y_min = min(y_min, y + extents.height / upem)  # hb heights are negative

    if x_min > x_max:
        return UNIFORM_FRAME
    return Frame.from_extents(
        x_min, y_min, x_max, y_max, margin=margin, min_size=min_size
    )


# ── The one rendering function ────────────────────────────────────────────


def render(
    font,
    *,
    codepoint: int | None = None,
    gid: int | None = None,
    size: int,
    axis_position: Sequence[float] | None = None,
    frame: Frame | _Normalized = UNIFORM_FRAME,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Render one glyph and return its ``(image, geometry)``.

    Exactly one of ``codepoint`` or ``gid`` must be given; a codepoint is looked
    up through the font's HarfBuzz face.  ``frame`` selects the framing:

    * :class:`Frame` (default :data:`UNIFORM_FRAME`) — uniform scale into the em
      rectangle, aspect ratio kept, ink outside the frame clipped.
    * :data:`NORMALIZED` — crop-to-ink, stretch to a square, aspect ratio
      discarded.  This is the framing the diffusion generator and the upscaler
      consume.

    The glyph is rendered supersampled (:func:`_render_size`) and downscaled so
    corners survive the AA.

    Returns:
        ``(image, geometry)`` where ``image`` is a ``(size, size)`` float32
        tensor in ``[0, 1]`` (0 = ink, 1 = white) and ``geometry`` is a dict of
        the five em-unit labels ``scale_x``, ``scale_y``, ``left_sidebearing``,
        ``descender_depth``, ``advance`` — always in the font's own em units,
        whatever the frame.
    """
    if (gid is None) == (codepoint is None):
        raise ValueError("render() takes exactly one of gid= or codepoint=")
    if gid is None:
        gid = hb.Font(font.hb_face).get_nominal_glyph(codepoint)

    if frame is NORMALIZED:
        return _render_normalized(font, gid, size, axis_position)
    if not isinstance(frame, Frame):
        raise TypeError(f"frame must be a Frame or NORMALIZED, got {frame!r}")
    return _render_uniform(font, gid, size, axis_position, frame)


def _render_normalized(
    font,
    gid: int,
    size: int,
    axis_position: Sequence[float] | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """The crop-to-ink (:data:`NORMALIZED`) path of :func:`render`."""
    rsize = _render_size(size)
    raw = _render_gid_raw(font.path, gid, rsize, axis_position=axis_position)
    image, geometry = _normalize_bitmap(
        raw.bitmap, raw.bitmap_left, raw.bitmap_top, raw.advance_px, size, ppem=rsize
    )
    return image[0], geometry


def _render_uniform(
    font,
    gid: int,
    size: int,
    axis_position: Sequence[float] | None,
    frame: Frame,
) -> tuple[torch.Tensor, dict[str, float]]:
    """The uniform-scale (:class:`Frame`) path of :func:`render`."""
    rsize = _render_size(size)
    span = frame.em_span  # em covered by one canvas side
    ppm = max(1, round(rsize / span))  # 1 em == ppm pixels
    raw = _render_gid_raw(font.path, gid, ppm, axis_position=axis_position)
    geometry = _geometry_from_raw(raw, ppm)

    # Centre the frame in the square canvas (a no-op for a square frame).
    pad_x = (span - frame.width) / 2.0
    pad_y = (span - frame.height) / 2.0
    baseline_y = round((frame.top + pad_y) * ppm)
    origin_x = round((pad_x - frame.left) * ppm)
    canvas = np.full((rsize, rsize), 255, dtype=np.uint8)
    _paste_bitmap_onto_canvas(
        canvas=canvas,
        bitmap_array=raw.bitmap,
        bitmap_left=origin_x + raw.bitmap_left,
        bitmap_top=raw.bitmap_top,
        baseline_y=baseline_y,
    )
    if not torch:
        return canvas, geometry

    # Ink as 0.0 on a white 1.0 background, downscaled from the supersampled canvas.
    out = torch.from_numpy(canvas.astype(np.float32) / 255.0)  # (rsize, rsize)
    image = F.interpolate(
        out[None, None], size=(size, size), mode="bilinear", align_corners=False
    )[
        0, 0
    ]  # (size, size)
    return image, geometry


def denormalize_uniform(
    image: np.ndarray,
    geometry: Sequence[float],
    size: int,
    *,
    frame: Frame = UNIFORM_FRAME,
) -> torch.Tensor:
    """Denormalize a crop-to-ink glyph into the uniform square framing.

    Inverse of the uniform framing of :func:`render`: un-squares a crop-to-ink
    normalized glyph back to its true em extent and places it onto a ``size``-square canvas
    showing exactly the em rectangle ``frame`` (default :data:`UNIFORM_FRAME`:
    baseline 1.5em from the top, X origin 0.5em from the left, ``1 em ==
    size / 2`` pixels).  Used to frame generated glyphs the same way the
    set-membership model's training images are framed; a caller using a
    per-font frame must pass the same one it rendered that font with.

    Args:
        image: ``(H, W)`` float32 array in [0, 1], ink = 0, white = 1 — a
            crop-to-ink normalized square.
        geometry: the five em-unit labels in canonical order
            ``(scale_x, scale_y, left_sidebearing, descender_depth, advance)``.
        size: output square side length.

    Returns:
        ``(size, size)`` float32 tensor in [0, 1] (ink = 0, white = 1).
    """
    ppm = size / frame.em_span  # pixels per em
    scale_x, scale_y, lsb, descender_depth, _advance = (float(v) for v in geometry)
    baseline_offset = scale_y - descender_depth

    h_px = max(1, round(scale_y * ppm))
    w_px = max(1, round(scale_x * ppm))
    src = (
        torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))
        .unsqueeze(0)
        .unsqueeze(0)
    )
    glyph = F.interpolate(src, size=(h_px, w_px), mode="bilinear", align_corners=False)[
        0, 0
    ].numpy()  # (h_px, w_px)

    # Centre the frame in the square canvas (a no-op for a square frame).
    pad_x = (frame.em_span - frame.width) / 2.0
    pad_y = (frame.em_span - frame.height) / 2.0
    origin_x = round((pad_x - frame.left) * ppm)
    baseline_y = round((frame.top + pad_y) * ppm)
    canvas = np.ones((size, size), dtype=np.float32)
    x0 = origin_x + round(lsb * ppm)
    y0 = baseline_y - round(baseline_offset * ppm)

    gy1 = glyph.shape[0]
    gx1 = glyph.shape[1]
    cy0, cy1 = max(0, y0), min(size, y0 + gy1)
    cx0, cx1 = max(0, x0), min(size, x0 + gx1)
    if cy1 > cy0 and cx1 > cx0:
        canvas[cy0:cy1, cx0:cx1] = glyph[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]

    return torch.from_numpy(canvas)


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
        "--show",
        action="store_true",
        help="Display rendering using matplotlib",
    )
    return parser.parse_args()


def main() -> None:
    """Debug entry point to render a single glyph and display it."""
    from hrothgar.googlefonts import StandaloneFont

    args = _parse_args()
    font = StandaloneFont(args.font)
    rendering, _ = render(font, gid=args.gid, size=args.size)

    non_white_pixels = int((rendering < 1.0).sum())
    print("Rendered non-white pixels:", non_white_pixels)

    if args.show:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(5, 5))
        plt.imshow(rendering, cmap="gray", vmin=0.0, vmax=1.0)
        plt.title(f"{args.font.name} GID={args.gid} size={args.size}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    if args.save:
        import matplotlib.pyplot as plt

        output_path = args.font.with_suffix(f".gid{args.gid}.png")
        plt.imsave(output_path, rendering, cmap="gray", vmin=0.0, vmax=1.0)
        print(f"Saved rendering to {output_path}")


if __name__ == "__main__":
    main()
