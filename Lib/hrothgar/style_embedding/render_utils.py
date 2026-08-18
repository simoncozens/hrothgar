"""Shared glyph-rendering helpers for the style embedding model.

Both the training dataset (``FontStyleDatasetMaker.collate_fn``) and
``FontStyleEmbedder.compute_embedding`` use these functions so the rendering
path lives in exactly one place and cannot drift out of sync.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from hrothgar.googlefonts import Font


def render_glyph(
    font: Font,
    codepoint: int,
    size: int,
    axis_position: Optional[list[float]] = None,
) -> torch.Tensor:
    """Render a single glyph to a ``(size, size)`` greyscale tensor in [0, 1].

    ``Font.render`` returns a ``(3, size, size)`` float32 array in [0, 1]
    with ink near 0.0 on a white (1.0) background.  Style embedding only
    cares about ink shape, so we take the red channel as a greyscale image.
    """
    arr = font.render(codepoint, size=size, axis_position=axis_position)
    gray = arr[0].copy() if arr.ndim == 3 else arr
    return torch.from_numpy(gray)


def render_input_set(
    font: Font,
    codepoints: Sequence[int],
    size: int,
    axis_position: Optional[list[float]] = None,
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
