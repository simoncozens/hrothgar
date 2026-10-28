"""Contour-level glyph signatures: curvature + normal samples at on-curve points.

Refactored from ``pen_probe.py``.  A glyph is drawn (through extrema insertion
and quadratic->cubic conversion) into one closed ``BezPath`` per contour; each
on-curve point (segment start) then yields a fixed-length signature of curvature
and unit-normal samples taken just before and just after the point, along the
two adjacent cubic segments.  The join point itself is deliberately skipped
(curvature is discontinuous there); the samples *around* it are the signal.

Each sample carries its **arc-length offset** from the join point (negative on
the backward arm, positive on the forward arm), normalised by the glyph's ink
height.  This is what keeps rectilinear outlines from collapsing: curvature is
zero along straight edges, so without arc-length the signature reduces to a
sequence of edge *directions* and discards all proportions.  Curvature is also
multiplied by ink height (dimensionless), and normals are unit vectors already.
Orientation is preserved (no rotation invariance) — crossbar tilt and stress
angle are real style signal.
"""

from __future__ import annotations

import math

import uharfbuzz as hb
from kurbopy import BezPathCreatingPen, CubicBez, Line, Vec2

from hrothgar.pens import AddExtremaPen, Qu2CuPen

# Samples per side of each on-curve point (matches pen_probe.py's t grid:
# 0.1 .. 0.6 on the backward arm, 0.0 .. 0.5 on the forward arm).
_SIDE_GRID = [i * 0.1 for i in range(6, 0, -1)]  # 0.6 .. 0.1
_FWD_GRID = [i * 0.1 for i in range(0, 6)]  # 0.0 .. 0.5
_ARCLEN_STEPS = 8  # trapezoid subdivisions for the arc-length estimate


def _normal(tangent: Vec2) -> Vec2:
    length = math.hypot(tangent.x, tangent.y)
    if length < 1e-12:
        return Vec2(0.0, 0.0)
    return Vec2(-tangent.y / length, tangent.x / length)


CURV_CLIP = 100.0  # in ink-height units: a sharp corner, not infinity


def _seg_curvature(seg, t: float) -> float:
    if isinstance(seg, Line):
        return 0.0
    return seg.curvature(t)


def _scaled_curvature(seg, t: float, ink: float) -> float:
    """Curvature scaled by ink height, clamped and NaN-safe.

    Sharp corners have unbounded curvature; clip so a degenerate/cusp segment
    doesn't emit 1e14 and poison the model.
    """
    k = _seg_curvature(seg, t) * ink
    if not math.isfinite(k):
        return 0.0
    return max(-CURV_CLIP, min(CURV_CLIP, k))


def _seg_normal(seg, t: float) -> Vec2:
    if isinstance(seg, Line):
        return _normal(Vec2(seg.end().x - seg.start().x, seg.end().y - seg.start().y))
    return _normal(seg.deriv().eval(t).to_vec2())


def _speed(seg, t: float) -> float:
    d = seg.deriv().eval(t)
    return math.hypot(d.x, d.y)


def _seg_arclen(seg, t: float) -> float:
    """Arc length from the segment start (t=0) to parameter ``t``.

    ``kurbopy``'s ``arclen`` is pathologically slow on real glyph segments, so
    we integrate ``|deriv|`` ourselves with a fixed trapezoid rule — accurate
    enough for a smooth cubic and orders of magnitude faster.
    """
    if isinstance(seg, Line):
        dx = seg.end().x - seg.start().x
        dy = seg.end().y - seg.start().y
        return math.hypot(dx, dy) * t
    total = 0.0
    prev = _speed(seg, 0.0)
    for i in range(1, _ARCLEN_STEPS + 1):
        ti = t * i / _ARCLEN_STEPS
        cur = _speed(seg, ti)
        total += (prev + cur) / 2.0 * (t / _ARCLEN_STEPS)
        prev = cur
    return total


def extract_contours(
    face: hb.Face, gid: int, location: dict[str, float] | None = None
) -> list:
    """Extract the raw, compatible outline of ``gid`` at ``location``.

    Draws the (variation-instanced) glyph into a :class:`BezPathCreatingPen`,
    which accumulates one closed ``BezPath`` per contour.
    """
    font = hb.Font(face)
    if location:
        font.set_variations(location)
    bezpen = BezPathCreatingPen()
    pen = AddExtremaPen(bezpen)
    pen = Qu2CuPen(pen, max_err=5, all_cubic=True)
    font.draw_glyph_with_pen(gid, pen)
    return list(bezpen.paths)


def _ink_height(paths: list) -> float:
    ys: list[float] = []
    for path in paths:
        for seg in path.segments():
            if isinstance(seg, CubicBez):
                ys.extend([seg.start().y, seg.p1.y, seg.p2.y, seg.end().y])
            else:
                ys.extend([seg.start().y, seg.end().y])
    height = max(ys) - min(ys)
    return height if height > 0 else 1.0


def outline_fingerprint(
    face: hb.Face,
    gid: int,
    location: dict[str, float] | None = None,
    *,
    precision: int = 4,
) -> tuple:
    """Return a hashable fingerprint of the *normalized* outline geometry.

    Unlike :func:`glyph_signature` this keeps the actual point coordinates (both
    on-curve and control points), centered on the ink bbox and scaled so the ink
    height is 1, then quantized to a ``10 ** precision`` grid.  It is therefore
    exact for bit-identical outlines (the duplicate-design case) but *not*
    degenerate for rectilinear outlines, whose curvature-only signature loses
    all proportion information.  Used as the dedupe key.
    """
    paths = extract_contours(face, gid, location)

    raw: list[tuple[float, float]] = []
    for path in paths:
        for seg in path.segments():
            if isinstance(seg, CubicBez):
                raw.extend(
                    [
                        (seg.start().x, seg.start().y),
                        (seg.p1.x, seg.p1.y),
                        (seg.p2.x, seg.p2.y),
                        (seg.end().x, seg.end().y),
                    ]
                )
            else:
                raw.extend(
                    [(seg.start().x, seg.start().y), (seg.end().x, seg.end().y)]
                )

    xs = [p[0] for p in raw]
    ys = [p[1] for p in raw]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    height = (max_y - min_y) or 1.0
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    grid = 10 ** precision

    out: list[int] = []
    for x, y in raw:
        out.append(round((x - cx) / height * grid))
        out.append(round((y - cy) / height * grid))
    return tuple(out)


def glyph_signature(
    face: hb.Face,
    gid: int,
    location: dict[str, float] | None = None,
    *,
    precision: int = 4,
) -> tuple:
    """Return a hashable, scale-invariant contour signature.

    The result is a nested tuple ``(contour, point, sample)`` where each sample
    is ``(arc_length_offset / ink_height, curvature * ink_height, normal_x,
    normal_y)`` rounded to ``precision`` decimals.  The arc-length offset is
    signed (negative on the backward arm, positive on the forward arm).
    """
    paths = extract_contours(face, gid, location)
    ink = _ink_height(paths)

    signature: list = []
    for path in paths:
        segments = list(path.segments())
        contour: list = []
        for seg_ix, segment in enumerate(segments):
            prev_segment = segments[seg_ix - 1]
            prev_len = _seg_arclen(prev_segment, 1.0)
            samples: list = []
            for t in _SIDE_GRID:
                tt = 1.0 - t
                s = (_seg_arclen(prev_segment, tt) - prev_len) / ink
                k = _scaled_curvature(prev_segment, tt, ink)
                n = _seg_normal(prev_segment, tt)
                samples.append(
                    (
                        round(s, precision),
                        round(k, precision),
                        round(n.x, precision),
                        round(n.y, precision),
                    )
                )
            for t in _FWD_GRID:
                s = _seg_arclen(segment, t) / ink
                k = _scaled_curvature(segment, t + 0.001, ink)  # t=0 is ill-defined
                n = _seg_normal(segment, t)
                samples.append(
                    (
                        round(s, precision),
                        round(k, precision),
                        round(n.x, precision),
                        round(n.y, precision),
                    )
                )
            contour.append(tuple(samples))
        signature.append(tuple(contour))
    return tuple(signature)


def _contour_winding(path) -> float:
    """Signed winding of a closed contour: +1 outer, -1 inner.

    Uses the shoelace area of the on-curve points; the *sign* is what matters
    (outer and inner contours have opposite orientation), so this is a clean
    scale-invariant "which contour is this" feature.
    """
    pts = [(s.start().x, s.start().y) for s in path.segments()]
    area = 0.0
    for i in range(len(pts)):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % len(pts)]
        area += x0 * y1 - x1 * y0
    return 1.0 if area >= 0 else -1.0


def glyph_tokens(
    face: hb.Face,
    gid: int,
    location: dict[str, float] | None = None,
) -> list[list[float]]:
    """Return one token vector per on-curve point, for the model.

    Each token is 49 floats: 12 samples x (arc-length offset, curvature, nx, ny)
    = 48, plus the contour winding sign.  Scale-invariant and variable-length
    (a *set* of points, not a sequence) — the set-encoder consumes this as a
    bag of local-geometry tokens.
    """
    paths = extract_contours(face, gid, location)
    ink = _ink_height(paths)

    tokens: list[list[float]] = []
    for path in paths:
        winding = _contour_winding(path)
        segments = list(path.segments())
        for seg_ix, segment in enumerate(segments):
            prev_segment = segments[seg_ix - 1]
            prev_len = _seg_arclen(prev_segment, 1.0)
            flat: list[float] = []
            for t in _SIDE_GRID:
                tt = 1.0 - t
                s = (_seg_arclen(prev_segment, tt) - prev_len) / ink
                k = _scaled_curvature(prev_segment, tt, ink)
                n = _seg_normal(prev_segment, tt)
                flat += [s, k, n.x, n.y]
            for t in _FWD_GRID:
                s = _seg_arclen(segment, t) / ink
                k = _scaled_curvature(segment, t + 0.001, ink)  # t=0 is ill-defined
                n = _seg_normal(segment, t)
                flat += [s, k, n.x, n.y]
            tokens.append(flat + [winding])
    return tokens
