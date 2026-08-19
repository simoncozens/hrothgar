"""Shared glyph rendering + normalization policy.

Single source of truth for the operations AR, GTok, and the style embedder all
perform on glyphs:

* render a glyph to a tensor
* compute its ink bounding box
* crop-to-ink and stretch-to-square normalization

All three models consume glyphs in the crop-to-ink normalized convention, so
this module is the one place that defines that policy.  Aspect ratio is
deliberately discarded during normalization; the generator predicts the
bounding box to denormalize the glyph back onto the baseline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from hrothgar.googlefonts import Font

# Ink is rendered near 0.0 on a white (1.0) background.
_INK_THRESHOLD = 0.5


def render_glyph(
    font: "Font",
    codepoint: int,
    size: int,
    axis_position: Optional[list[float]] = None,
) -> torch.Tensor:
    """Render a glyph as a ``(3, size, size)`` float32 tensor in [0, 1]."""
    arr = font.render(codepoint, size=size, axis_position=axis_position)
    return torch.from_numpy(arr.copy())


def ink_bbox(rendering: torch.Tensor, size: int) -> torch.Tensor:
    """Return the ink bbox as normalized ``(x0, y0, x1, y1)`` in [0, 1].

    ``rendering`` may be ``(1, H, W)`` or ``(3, H, W)``; ink detection uses
    channel 0.  Blank glyphs return an all-zero bbox.
    """
    ink = rendering[0] < _INK_THRESHOLD
    if not ink.any():
        return torch.zeros(4, dtype=torch.float32, device=rendering.device)
    ys, xs = ink.nonzero(as_tuple=True)
    return torch.tensor(
        [
            xs.min().float(),
            ys.min().float(),
            xs.max().float(),
            ys.max().float(),
        ],
        dtype=torch.float32,
        device=rendering.device,
    ) / size


def bbox_size(rendering: torch.Tensor, size: int) -> torch.Tensor:
    """Return the normalized ink bbox ``(width, height)`` in [0, 1]."""
    x0, y0, x1, y1 = ink_bbox(rendering, size)
    return torch.stack([x1 - x0, y1 - y0])


def crop_to_ink(rendering: torch.Tensor, size: int) -> torch.Tensor:
    """Crop a glyph to its ink bbox and stretch it to ``(C, size, size)``.

    Args:
        rendering: ``(C, H, W)`` float32 glyph image in [0, 1] (C = 1 or 3).
        size: side length of the output square.

    Returns:
        ``(C, size, size)`` float32. Blank glyphs (no ink) return a white
        square of ``size``.
    """
    if rendering.ndim != 3:
        raise ValueError(
            f"crop_to_ink expects (C, H, W), got {tuple(rendering.shape)}"
        )

    channels = rendering.shape[0]
    ink = rendering[0] < _INK_THRESHOLD
    if not ink.any():
        return torch.ones(
            (channels, size, size), dtype=rendering.dtype, device=rendering.device
        )

    ys, xs = ink.nonzero(as_tuple=True)
    y0, y1 = int(ys.min().item()), int(ys.max().item())
    x0, x1 = int(xs.min().item()), int(xs.max().item())
    crop = rendering[:, y0 : y1 + 1, x0 : x1 + 1]  # (C, h, w)

    return F.interpolate(
        crop.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
    )[0]


def render_normalized(
    font: "Font",
    codepoint: int,
    size: int,
    axis_position: Optional[list[float]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render + crop-to-ink a glyph.

    Returns:
        ``(normalized, bbox)`` where ``normalized`` is ``(3, size, size)`` and
        ``bbox`` is normalized ``(x0, y0, x1, y1)`` in [0, 1].
    """
    rendering = render_glyph(font, codepoint, size, axis_position=axis_position)
    return crop_to_ink(rendering, size), ink_bbox(rendering, size)
