"""Raw outline extraction and geometry for vector-space weight adjustment.

This module uses `kurbopy` (the Python port of the `kurbo` curve library) for
all the geometry heavy-lifting: tangents via ``deriv``, curvature via
``curvature``, flattening via ``BezPath.flatten``, and winding via
``BezPath.area``.

We deliberately do **not** route through :mod:`hrothgar.pens`.  That module
converts quadratics to cubics and inserts points at extrema, and the extrema
step is not weight-invariant: a curve that is monotonic in the regular master
can pick up (or lose) an internal extremum in the bold master, so the two
instances come back with *different* point counts.  The raw TrueType outline of
a variable font, by contrast, is interpolation-compatible by construction --
same contours, same on-curve points, same segment types -- which gives us the
regular -> bold node correspondence for free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import kurbopy as k
import uharfbuzz as hb

# Turn angle (degrees) below which a node is considered smooth, matching the
# ~10-degree rule img2bez uses in ``compute_smooth``.
SMOOTH_ANGLE_DEG = 10.0

# Flatten tolerance (font units) used for the medial-axis stroke width.
WIDTH_FLATTEN_TOL = 0.5
# Distance below which a flattened vertex counts as "the node itself" (so its
# incident segments are skipped when growing the inscribed circle).
INCIDENT_EPS = 0.5


@dataclass
class Glyph:
    """A glyph as a list of closed :class:`kurbopy.BezPath` contours."""

    contours: list[k.BezPath] = field(default_factory=list)


@dataclass
class NodeDecomposition:
    """The regular -> bold displacement of one node, in its own frame."""

    contour: int
    index: int
    x: float
    y: float
    dx: float
    dy: float
    nx: float
    ny: float
    tx: float
    ty: float
    offset: float
    tangential: float
    turn_angle: float
    is_corner: bool
    width: float | None


def extract_glyph(
    face: hb.Face, gid: int, location: dict[str, float] | None = None
) -> Glyph:
    """Extract the raw, compatible outline of ``gid`` at ``location``.

    Draws the (variation-instanced) glyph into a :class:`kurbopy.BezPathCreatingPen`,
    which accumulates one closed ``BezPath`` per contour.  The resulting point
    structure is the font's own compatible master structure, identical across
    weights.
    """
    font = hb.Font(face)
    if location:
        font.set_variations(location)
    pen = k.BezPathCreatingPen()
    font.draw_glyph_with_pen(gid, pen)
    return Glyph(contours=list(pen.paths))


def _segments(contour: k.BezPath) -> list[k.PathSeg]:
    return list(contour.segments())


def _tangent_vec(seg: k.PathSeg, t: float) -> k.Vec2:
    """The tangent vector of ``seg`` at parameter ``t`` (unnormalized)."""
    d = seg.deriv().eval(t)
    return k.Vec2(d.x, d.y)


def _norm(v: k.Vec2) -> k.Vec2:
    m = v.hypot()
    if m < 1e-9:
        return k.Vec2(0.0, 0.0)
    return k.Vec2(v.x / m, v.y / m)


def node_position(contour: k.BezPath, i: int) -> k.Point:
    """The on-curve point of node ``i`` (start of segment ``i``)."""
    return _segments(contour)[i].p0


def node_tangents(contour: k.BezPath, i: int) -> tuple[k.Vec2, k.Vec2]:
    """Return ``(incoming, outgoing)`` tangent vectors at node ``i``."""
    segs = _segments(contour)
    n = len(segs)
    outgoing = _tangent_vec(segs[i], 0.0)
    incoming = _tangent_vec(segs[(i - 1) % n], 1.0)
    return incoming, outgoing


def node_frame(contour: k.BezPath, i: int) -> tuple[k.Vec2, k.Vec2]:
    """Return the ``(tangent, normal)`` unit frame at node ``i``.

    ``normal`` points away from the filled stroke: outward for an outer
    contour, into the counter for an inner one.  For a smooth node the tangent
    is the contour direction; for a corner it is the angle bisector, so the
    normal is the corner's bisector normal -- exactly the direction bolding
    pushes a corner.
    """
    incoming, outgoing = node_tangents(contour, i)
    t = _norm(k.Vec2(incoming.x + outgoing.x, incoming.y + outgoing.y))
    if t.x == 0.0 and t.y == 0.0:
        t = _norm(outgoing)
    if t.x == 0.0 and t.y == 0.0:
        return k.Vec2(0.0, 0.0), k.Vec2(0.0, 0.0)
    return t, k.Vec2(-t.y, t.x)


def turn_angle(contour: k.BezPath, i: int) -> float:
    """The (signed) turn angle in radians at node ``i``."""
    incoming, outgoing = node_tangents(contour, i)
    return math.atan2(incoming.cross(outgoing), incoming.dot(outgoing))


def is_corner(
    contour: k.BezPath, i: int, smooth_angle_deg: float = SMOOTH_ANGLE_DEG
) -> bool:
    """Classify a node as corner vs smooth, mirroring img2bez ``compute_smooth``."""
    incoming, outgoing = node_tangents(contour, i)
    cross = incoming.cross(outgoing)
    dot = incoming.dot(outgoing)
    return not (abs(cross) < math.sin(math.radians(smooth_angle_deg)) and dot > 0.0)


def node_curvature(contour: k.BezPath, i: int) -> float:
    """Signed curvature of the outgoing segment at node ``i`` (0 for lines)."""
    seg = _segments(contour)[i]
    if isinstance(seg, k.Line):
        return 0.0
    return seg.curvature(0.0)


def _flattened(glyph: Glyph, tol: float) -> list[list[k.Point]]:
    """Flatten each contour to a closed polyline (first == last vertex)."""
    return [list(path.flatten(tol)) for path in glyph.contours]


def _bbox_diagonal(glyph: Glyph) -> float:
    """The diagonal of the glyph's bounding box, used as an upper bound."""
    diag = 0.0
    for path in glyph.contours:
        if path.is_empty():
            continue
        r = path.bounding_box()
        w = r.max_x() - r.min_x()
        h = r.max_y() - r.min_y()
        diag = max(diag, math.hypot(w, h))
    return diag if diag > 0.0 else 1.0


def _point_seg_dist_sq(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denom = abx * abx + aby * aby
    if denom < 1e-12:
        return apx * apx + apy * apy
    t = (apx * abx + apy * aby) / denom
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * abx, ay + t * aby
    dx, dy = px - cx, py - cy
    return dx * dx + dy * dy


def _circle_touch_radius(
    px: float,
    py: float,
    nx: float,
    ny: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
    rmax: float,
) -> float | None:
    """Smallest ``r`` where the circle ``center = p + r*n, radius = r`` touches
    segment ``(a, b)``, or ``None`` if it never does within ``rmax``.

    ``f(r) = dist(center, segment) - r`` is monotonically non-increasing, so the
    root is unique and a plain bisection finds it.
    """

    def f(r: float) -> float:
        cx = px + r * nx
        cy = py + r * ny
        return math.sqrt(_point_seg_dist_sq(cx, cy, ax, ay, bx, by)) - r

    if f(0.0) <= 0.0 or f(rmax) > 0.0:
        return None
    lo, hi = 0.0, rmax
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _stroke_width_at(
    glyph: Glyph,
    ci: int,
    i: int,
    flat: list[list[k.Point]],
    rmax: float,
) -> float | None:
    contour = glyph.contours[ci]
    p = node_position(contour, i)
    _, outward = node_frame(contour, i)
    # Across the stroke = opposite the outward normal.
    nx, ny = -outward.x, -outward.y
    if nx == 0.0 and ny == 0.0:
        return None

    best: float | None = None
    for pts in flat:
        for j in range(len(pts) - 1):
            a, b = pts[j], pts[j + 1]
            # Skip the segments incident to the node (they are tangent to it).
            if (
                (a.x - p.x) ** 2 + (a.y - p.y) ** 2 < INCIDENT_EPS * INCIDENT_EPS
                or (b.x - p.x) ** 2 + (b.y - p.y) ** 2 < INCIDENT_EPS * INCIDENT_EPS
            ):
                continue
            r = _circle_touch_radius(p.x, p.y, nx, ny, a.x, a.y, b.x, b.y, rmax)
            if r is not None and (best is None or r < best):
                best = r
    if best is None:
        return None
    return 2.0 * best


def stroke_widths(
    glyph: Glyph, tol: float = WIDTH_FLATTEN_TOL
) -> list[list[float | None]]:
    """The stroke width at every on-curve node, indexed ``[contour][node]``.

    Flattens the glyph once and shares the result across all nodes, which is far
    cheaper than calling :func:`stroke_width` per node.
    """
    flat = _flattened(glyph, tol)
    rmax = _bbox_diagonal(glyph)
    return [
        [
            _stroke_width_at(glyph, ci, i, flat, rmax)
            for i in range(len(_segments(glyph.contours[ci])))
        ]
        for ci in range(len(glyph.contours))
    ]


def stroke_width(glyph: Glyph, ci: int, i: int) -> float | None:
    """The local stroke width at node ``i`` of contour ``ci``.

    This is the diameter of the maximal circle inscribed in the stroke and
    tangent to the boundary at the node -- the medial-axis (MAT) "osculating
    circle" diameter.  It is more robust than casting a single ray, because it
    measures the true perpendicular thickness even where the opposite boundary
    is oblique to the node's normal.
    """
    return stroke_widths(glyph)[ci][i]


def decompose(
    regular: Glyph,
    bold: Glyph,
    smooth_angle_deg: float = SMOOTH_ANGLE_DEG,
    with_width: bool = False,
) -> list[NodeDecomposition]:
    """Decompose the regular -> bold displacement into normal/tangent parts.

    Requires ``regular`` and ``bold`` to be compatible (same contour and point
    counts), which variable-font masters guarantee.  Returns one
    :class:`NodeDecomposition` per on-curve node, with ``offset`` measured
    along the outward normal (positive = thickening) and ``tangential`` along
    the contour tangent.
    """
    out: list[NodeDecomposition] = []
    for ci, (cr, cb) in enumerate(zip(regular.contours, bold.contours)):
        segs_r = _segments(cr)
        segs_b = _segments(cb)
        for i, (sr, sb) in enumerate(zip(segs_r, segs_b)):
            pr, pb = sr.p0, sb.p0
            dx = pb.x - pr.x
            dy = pb.y - pr.y
            tangent, normal = node_frame(cr, i)
            offset = dx * normal.x + dy * normal.y
            tangential = dx * tangent.x + dy * tangent.y
            width = stroke_width(regular, ci, i) if with_width else None
            out.append(
                NodeDecomposition(
                    contour=ci,
                    index=i,
                    x=pr.x,
                    y=pr.y,
                    dx=dx,
                    dy=dy,
                    nx=normal.x,
                    ny=normal.y,
                    tx=tangent.x,
                    ty=tangent.y,
                    offset=offset,
                    tangential=tangential,
                    turn_angle=turn_angle(cr, i),
                    is_corner=is_corner(cr, i, smooth_angle_deg),
                    width=width,
                )
            )
    return out
