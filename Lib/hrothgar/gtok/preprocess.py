"""Glyph preprocessing policy for GTok: crop-to-ink normalization.

GTok's input policy is to normalize each glyph to its ink bounding box, then
stretch it to fill the full tokenizer resolution. This ensures fine detail
(terminals, stroke modulation, curve squareness) occupies the token grid at a
useful scale instead of only a few pixels in a large empty canvas.

Aspect ratio is deliberately discarded. The generator will later predict the
bounding box to denormalize the glyph back onto the shared baseline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def crop_to_ink(rendering: torch.Tensor, size: int) -> torch.Tensor:
    """Crop a glyph rendering to its ink bbox and stretch it to a square.

    Args:
        rendering: ``(3, H, W)`` float32 glyph image in [0, 1], ink near 0.0
            on a white (1.0) background.
        size: side length of the output square.

    Returns:
        ``(3, size, size)`` float32. Blank glyphs (no ink) are returned
        unchanged (they are already ``(3, size, size)`` white).
    """
    if rendering.ndim != 3:
        raise ValueError(
            f"crop_to_ink expects (3, H, W), got {tuple(rendering.shape)}"
        )

    ink = rendering[0] < 0.5
    if not ink.any():
        return rendering

    ys, xs = ink.nonzero(as_tuple=True)
    y0, y1 = int(ys.min().item()), int(ys.max().item())
    x0, x1 = int(xs.min().item()), int(xs.max().item())
    crop = rendering[:, y0 : y1 + 1, x0 : x1 + 1]  # (3, h, w)

    return F.interpolate(
        crop.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
    )[0]
