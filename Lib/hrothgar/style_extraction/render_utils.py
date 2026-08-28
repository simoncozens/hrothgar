"""Glyph rendering for style extraction (greyscale, crop-to-ink).

Follows the shared normalization policy in ``hrothgar.glyph_rendering``: glyphs
are cropped to their ink bounding box and stretched to a square, so the model
learns shape + style in a canonical box and the bounding box is predicted
separately (as in GTok / AR).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from hrothgar.glyph_rendering import crop_to_ink
from hrothgar.glyph_rendering import render_glyph as _render_glyph_rgb


def render_glyph(
    font,
    codepoint: int,
    size: int,
    axis_position: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Render + crop-to-ink a glyph as a ``(size, size)`` greyscale tensor in [0, 1]."""
    rendering = _render_glyph_rgb(font, codepoint, size, axis_position=axis_position)
    return crop_to_ink(rendering, size)[0]
