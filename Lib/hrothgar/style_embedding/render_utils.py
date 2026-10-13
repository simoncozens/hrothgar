"""Style-embedding glyph rendering (greyscale) on top of the shared renderer.

Both the training dataset (``FontStyleDatasetMaker.collate_fn``) and
``FontStyleEmbedder.compute_embedding`` use these functions, so training and
inference cannot drift apart.  Glyphs are rendered onto the fixed square canvas
with the baseline pinned to the bottom, so the image keeps the font's
proportions (and clips descenders).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from hrothgar.render import render_glyph as _render_glyph_rgb


def render_glyph(
    font,
    codepoint: int,
    size: int,
    axis_position: Sequence[float] | None = None,
) -> torch.Tensor:
    """Render a single glyph to a ``(size, size)`` greyscale tensor in [0, 1]."""
    return _render_glyph_rgb(font, codepoint, size, axis_position=axis_position)[0]


def render_input_set(
    font,
    codepoints: Sequence[int],
    size: int,
    axis_position: Sequence[float] | None = None,
) -> torch.Tensor:
    """Render the full input glyph set as ``(G, 1, size, size)`` greyscale."""
    glyphs = [
        render_glyph(font, cp, size, axis_position=axis_position) for cp in codepoints
    ]
    return torch.stack(glyphs).unsqueeze(1)  # (G, 1, size, size)
