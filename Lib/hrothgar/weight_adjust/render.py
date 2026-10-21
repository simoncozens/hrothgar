"""Shared-frame geometry read-back for the weight-adjustment model.

The generator crop-to-inks and stretches to a square because its job is to
produce *content* from a style latent, and layout is a confound to strip off.
Weight adjustment is the opposite: the content (skeleton) is already given, and
the thing being manipulated — stroke thickness — is exactly what crop-to-ink
normalizes away.  So every weight of a glyph is rendered into the *same* frame
at the *same* pen origin and baseline.

That frame is :data:`hrothgar.render.UNIFORM_FRAME` (the 2em window): 0.5em of
left margin, 1.5em above and 0.5em below the baseline.  The origin is a fixed,
known offset from the left edge, so the left sidebearing can be read back off
the raster (``lsb = ink_x0 - origin``).  Only the advance width is invisible in
ink; the model's advance head supplies it.

Rendering itself goes through :func:`hrothgar.render.render`; this module only
holds the frame's derived constants and the functions that read geometry back
off a rendered image at inference time.
"""

from __future__ import annotations

import torch

from hrothgar.render import UNIFORM_FRAME

# Derived from UNIFORM_FRAME: baseline 1.5em from the top, X origin 0.5em from
# the left, 0.5em of descender room.
ASCENDER_EM = UNIFORM_FRAME.top  # 1.5
DESCENDER_EM = -UNIFORM_FRAME.bottom  # 0.5
ORIGIN_X_EM = -UNIFORM_FRAME.left  # 0.5
EM_SPAN = UNIFORM_FRAME.em_span  # 2.0 em square

_INK_THRESHOLD = 0.5


def measure_geometry(image: torch.Tensor, size: int) -> dict[str, float]:
    """Read ink geometry back off a uniform-frame image.

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
    """The left sidebearing in em, read off a uniform-frame image."""
    return measure_geometry(image, size)["left_sidebearing"]
