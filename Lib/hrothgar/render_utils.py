"""Glyph rendering for style extraction (greyscale, crop-to-ink).

Follows the shared normalization policy in ``hrothgar.glyph_rendering``: glyphs
are cropped to their ink bounding box and stretched to a square, so the model
learns shape + style in a canonical box and the bounding box is predicted
separately (as in GTok / AR).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import uharfbuzz as hb

from hrothgar.glyph_rendering import crop_to_ink
from hrothgar.glyph_rendering import render_glyph as _render_glyph_rgb
from hrothgar.glyph_rendering import normalize_bitmap
from hrothgar.render import render_gid_raw


def render_glyph(
    font,
    codepoint: int,
    size: int,
    axis_position: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Render + crop-to-ink a glyph as a ``(size, size)`` greyscale tensor in [0, 1]."""
    rendering = _render_glyph_rgb(font, codepoint, size, axis_position=axis_position)
    return crop_to_ink(rendering, size)[0]


def render_glyph_with_geometry(
    font,
    codepoint: int,
    size: int,
    axis_position: Optional[Sequence[float]] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Render + crop-to-ink a glyph, also returning its geometry labels.

    Unlike :func:`render_glyph`, this reads FreeType's raw bitmap and offsets
    directly, so descenders are not clipped at the baseline and negative left
    sidebearings are not clipped at ``x=0``.

    Returns:
        ``(image, geometry)`` where ``image`` is a ``(size, size)`` greyscale
        tensor in [0, 1] (0 = ink, 1 = white) and ``geometry`` is a dict of the
        five em-unit labels ``scale_x``, ``scale_y``, ``left_sidebearing``,
        ``baseline_offset``, ``advance``.
    """
    gid = hb.Font(font.hb_face).get_nominal_glyph(codepoint)
    raw = render_gid_raw(font.path, gid, size, axis_position=axis_position)
    image, geometry = normalize_bitmap(
        raw.bitmap, raw.bitmap_left, raw.bitmap_top, raw.advance_px, size
    )
    return image[0], geometry
