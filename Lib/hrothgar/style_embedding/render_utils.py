"""Style-embedding glyph rendering (greyscale) on top of the shared renderer.

Both the training dataset (``FontStyleDatasetMaker.collate_fn``) and
``FontStyleEmbedder.compute_embedding`` use these functions.  The actual
render-to-tensor logic lives in ``hrothgar.glyph_rendering`` so the rendering
path is shared across AR, GTok, and the style embedder.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from hrothgar.glyph_rendering import render_glyph as _render_glyph_rgb


def render_glyph(
    font,
    codepoint: int,
    size: int,
    axis_position: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Render a single glyph to a ``(size, size)`` greyscale tensor in [0, 1]."""
    return _render_glyph_rgb(font, codepoint, size, axis_position=axis_position)[0]


def render_input_set(
    font,
    codepoints: Sequence[int],
    size: int,
    axis_position: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Render the full input glyph set as ``(G, 1, size, size)`` greyscale.

    This is the exact input tensor shape ``FontStyleEmbedder.encode`` /
    ``encode_with_tokens`` expect (``(B, G, 1, H, W)``) without the batch
    dimension.
    """
    glyphs = [
        render_glyph(font, cp, size, axis_position=axis_position)
        for cp in codepoints
    ]
    return torch.stack(glyphs).unsqueeze(1)  # (G, 1, size, size)
