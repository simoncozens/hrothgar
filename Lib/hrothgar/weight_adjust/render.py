"""Shared-frame rendering for the weight-adjustment model.

The generator crop-to-inks and stretches to a square because its job is to
produce *content* from a style latent, and layout is a confound to strip off.
Weight adjustment is the opposite: the content (skeleton) is already given, and
the thing being manipulated — stroke thickness — is exactly what crop-to-ink
normalizes away.  So this module renders every weight of a glyph into the *same*
``EM_SPAN``-em square at the *same* pen origin and baseline.

The origin is a fixed, known offset from the left edge, so the left sidebearing
can be read back off the raster (``lsb = ink_x0 - origin``).  Only the advance
width is invisible in ink, and it is returned directly by the renderer (and
predicted by the model's advance head at inference time).

Frame layout (``EM_SPAN`` = ascender + descender = 2.0 em square):

* ``ORIGIN_X_EM = 0.5`` — left margin so glyphs with a negative left
  sidebearing (e.g. an italic ``f`` or ``j`` overhang) extend left of the
  origin without clipping.
* ``ASCENDER_EM = 1.0`` / ``DESCENDER_EM = 1.0`` — 1 em of room above and
  below the baseline, so descenders are not clipped.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from hrothgar.render import render_gid_raw, render_size

ASCENDER_EM = 1.0
DESCENDER_EM = 1.0
ORIGIN_X_EM = 0.5
EM_SPAN = ASCENDER_EM + DESCENDER_EM  # 2.0 em square

_INK_THRESHOLD = 0.5


def render_gid_shared_frame(
    font,
    gid: int,
    size: int,
    axis_position: Sequence[float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Render a glyph into the shared, origin-tracked frame.

    Renders supersampled (``render_size``) and downscales with area averaging,
    so the anti-aliased edges survive the downscale.  Both the regular and the
    bold are placed at the same pen origin and baseline, each using its own
    FreeType bitmap offsets, so the stroke-thickness difference between them is
    a first-order pixel signal.

    Args:
        font: a ``Font`` (needs ``.path``).
        gid: glyph index to render.
        size: output square side length.
        axis_position: variable-font design coordinates in fvar order (``None``
            for a static/default instance).

    Returns:
        ``(image, geometry)`` where ``image`` is a ``(size, size)`` float32
        tensor in ``[0, 1]`` (0 = ink, 1 = white) and ``geometry`` carries the
        two labels that matter for weight adjustment — ``advance`` and
        ``left_sidebearing``, both in em units.  ``left_sidebearing`` is the
        ground-truth value for reference; at inference it is read back off the
        image via :func:`measure_geometry`.
    """
    rsize = render_size(size)  # supersampled: 1 em == rsize px
    raw = render_gid_raw(font.path, gid, rsize, axis_position=axis_position)

    canvas_h = int(EM_SPAN * rsize)
    canvas = np.full((canvas_h, canvas_h), 255, dtype=np.uint8)
    origin_x = int(ORIGIN_X_EM * rsize)
    baseline_y = int(ASCENDER_EM * rsize)

    bitmap = raw.bitmap  # (h, w) uint8 coverage: 0 = no ink, 255 = full ink
    if bitmap.size:
        h, w = bitmap.shape
        dst_x0 = origin_x + raw.bitmap_left
        dst_y0 = baseline_y - raw.bitmap_top
        # Clamp the paste so a glyph whose ink extends past the frame does not
        # crash — it just clips (the frame is sized generously, so this is rare).
        src_x0 = max(0, -dst_x0)
        src_y0 = max(0, -dst_y0)
        src_x1 = w - max(0, dst_x0 + w - canvas_h)
        src_y1 = h - max(0, dst_y0 + h - canvas_h)
        if src_x0 < src_x1 and src_y0 < src_y1:
            dx0 = max(0, dst_x0)
            dy0 = max(0, dst_y0)
            canvas[dy0 : dy0 + (src_y1 - src_y0), dx0 : dx0 + (src_x1 - src_x0)] = (
                255 - bitmap[src_y0:src_y1, src_x0:src_x1]
            )

    image = torch.from_numpy(canvas).float().div_(255.0)  # 0 = ink, 1 = white
    image = F.interpolate(
        image.unsqueeze(0).unsqueeze(0), size=(size, size), mode="area"
    )[0, 0]

    geometry = {
        "advance": raw.advance_px / rsize,
        "left_sidebearing": raw.bitmap_left / rsize,
    }
    return image, geometry


def measure_geometry(image: torch.Tensor, size: int) -> dict[str, float]:
    """Read ink geometry back off a shared-frame image.

    Everything except ``advance`` is visible in the raster, because the frame's
    origin and baseline are fixed and known.  ``advance`` is not; the model's
    advance head supplies it.

    Args:
        image: ``(H, W)`` or ``(1, H, W)`` float32 in ``[0, 1]`` (0 = ink).
        size: the image's square side length (must match the render's ``size``).

    Returns:
        ``{scale_x, scale_y, left_sidebearing, descender_depth}`` in em units.
    """
    if image.dim() == 3:
        image = image[0]
    ink = image < _INK_THRESHOLD
    if not bool(ink.any()):
        return {
            "scale_x": 0.0,
            "scale_y": 0.0,
            "left_sidebearing": 0.0,
            "descender_depth": 0.0,
        }
    ys, xs = ink.nonzero(as_tuple=True)
    x0, x1 = int(xs.min().item()), int(xs.max().item())
    y0, y1 = int(ys.min().item()), int(ys.max().item())

    ppem = size / EM_SPAN  # pixels per em in the output image
    origin_x = ORIGIN_X_EM * ppem
    baseline_y = ASCENDER_EM * ppem

    return {
        "scale_x": (x1 - x0 + 1) / ppem,
        "scale_y": (y1 - y0 + 1) / ppem,
        "left_sidebearing": (x0 - origin_x) / ppem,
        # How far the ink hangs below the baseline (positive = descender).
        "descender_depth": (y1 - baseline_y) / ppem,
    }


def measure_lsb(image: torch.Tensor, size: int) -> float:
    """The left sidebearing in em, read off a shared-frame image."""
    return measure_geometry(image, size)["left_sidebearing"]
