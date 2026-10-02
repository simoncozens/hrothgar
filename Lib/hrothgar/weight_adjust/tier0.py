"""Tier 0: a scalar, parameter-driven bolding model.

The model predicts, for each on-curve node of a glyph's regular outline, how far
to move it along its outward normal to produce the bold instance::

    offset = (vertical * nx**2 + horizontal * ny**2) * width

where ``(nx, ny)`` is the node's outward unit normal and ``width`` its local
stroke width (the MAT osculating-circle diameter from :mod:`.outlines`).

``vertical`` and ``horizontal`` are per-font scalars: the expansion rate for
vertical stems (normal along x) and horizontal strokes (normal along y).  Their
ratio is the font's stroke contrast.  Fitting them against a handful of
exemplar ``(regular, bold)`` pairs *is* the per-font "bolding style".

The model is deliberately scalar and normal-only.  It is structure-preserving by
construction (it moves existing nodes, never adds or removes them), so the
output is always interpolation-compatible.  Its purpose is to establish how far
a two-parameter model gets and to characterize the residual -- chiefly the
tangential displacement (largest at corners) -- that a learned Tier 1 head will
need to model.
"""

from __future__ import annotations

from dataclasses import dataclass

import kurbopy as k

from hrothgar.weight_adjust.outlines import Glyph, node_frame, stroke_widths


@dataclass
class BoldParams:
    """Per-font bolding parameters (see module docstring)."""

    vertical: float = 0.0
    horizontal: float = 0.0


def predict_offset(
    params: BoldParams, nx: float, ny: float, width: float | None
) -> float:
    """The predicted normal offset for a node with normal ``(nx, ny)`` and width."""
    if width is None:
        return 0.0
    return (params.vertical * nx * nx + params.horizontal * ny * ny) * width


def fit(exemplars: list[tuple[Glyph, Glyph]]) -> BoldParams:
    """Least-squares fit of ``vertical``/``horizontal`` against exemplar pairs.

    Each ``(regular, bold)`` pair contributes one observation per on-curve node:
    the true normal offset ``delta . normal`` regressed on the two basis features
    ``nx**2 * width`` and ``ny**2 * width``.  Nodes whose width could not be
    measured are skipped.
    """
    a = b = c = r1 = r2 = 0.0
    for regular, bold in exemplars:
        widths = stroke_widths(regular)
        for ci, (cr, cb) in enumerate(zip(regular.contours, bold.contours)):
            segs_r = list(cr.segments())
            segs_b = list(cb.segments())
            for i, (sr, sb) in enumerate(zip(segs_r, segs_b)):
                width = widths[ci][i]
                if width is None:
                    continue
                _, normal = node_frame(cr, i)
                dx = sb.p0.x - sr.p0.x
                dy = sb.p0.y - sr.p0.y
                offset = dx * normal.x + dy * normal.y
                f0 = normal.x * normal.x * width
                f1 = normal.y * normal.y * width
                a += f0 * f0
                b += f0 * f1
                c += f1 * f1
                r1 += offset * f0
                r2 += offset * f1
    det = a * c - b * b
    if abs(det) < 1e-12:
        return BoldParams()
    return BoldParams(
        vertical=(r1 * c - r2 * b) / det,
        horizontal=(a * r2 - b * r1) / det,
    )


def apply(regular: Glyph, params: BoldParams) -> Glyph:
    """Predict the bold outline from ``regular``.

    On-curve nodes move along their outward normal by :func:`predict_offset`;
    off-curve handles move by the average of their two neighbouring nodes'
    displacements (a structure-preserving, first-order approximation).
    """
    widths = stroke_widths(regular)
    result = Glyph()
    for ci, contour in enumerate(regular.contours):
        segs = list(contour.segments())
        n = len(segs)
        disp: list[tuple[float, float]] = []
        for i in range(n):
            _, normal = node_frame(contour, i)
            off = predict_offset(params, normal.x, normal.y, widths[ci][i])
            disp.append((off * normal.x, off * normal.y))

        new = k.BezPath()
        first = segs[0].p0
        new.move_to(k.Point(first.x + disp[0][0], first.y + disp[0][1]))
        for i in range(n):
            nxt = segs[(i + 1) % n].p0
            end = k.Point(
                nxt.x + disp[(i + 1) % n][0],
                nxt.y + disp[(i + 1) % n][1],
            )
            if isinstance(segs[i], k.Line):
                new.line_to(end)
            else:
                q = segs[i].p1
                new.quad_to(
                    k.Point(
                        q.x + 0.5 * (disp[i][0] + disp[(i + 1) % n][0]),
                        q.y + 0.5 * (disp[i][1] + disp[(i + 1) % n][1]),
                    ),
                    end,
                )
        new.close_path()
        result.contours.append(new)
    return result
