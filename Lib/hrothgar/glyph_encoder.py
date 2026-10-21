"""Shared glyph encoder for the style and content critics.

A small strided CNN that maps a ``(B, 1, H, W)`` glyph image to a **spatial**
feature map ``(B, out_dim, H/ds, W/ds)``.  It deliberately performs **no
pooling**: spatial structure is preserved so each downstream model can choose
how to consume it — a global average pool for coarse tasks (content identity,
instance membership) or spatial attention/correlation for fine-grained style
matching.

This is the single implementation shared by ``set_membership``,
``content_detector`` and ``similarity``, which previously each carried a
functionally-identical copy.
"""

from __future__ import annotations

import torch
from torch import nn


class GlyphEncoder(nn.Module):
    """Small shared CNN: ``(1, H, W) -> (out_dim, H/ds, W/ds)`` (no pooling)."""

    def __init__(
        self, base_channels: int = 32, out_dim: int = 256, downsample: int = 4
    ) -> None:
        super().__init__()
        if downsample not in (2, 4, 8):
            raise ValueError(f"downsample must be 2, 4, or 8, got {downsample}")

        layers: list[nn.Module] = []
        c = 1
        layers.extend(self._block(c, base_channels, stride=1))
        c = base_channels
        while downsample >= 2:
            layers.extend(self._block(c, c * 2, stride=2))
            c *= 2
            downsample //= 2
        layers.append(nn.Conv2d(c, out_dim, 3, padding=1))
        self.net = nn.Sequential(*layers)

    @staticmethod
    def _block(in_c: int, out_c: int, stride: int) -> list[nn.Module]:
        return [
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1),
            nn.GroupNorm(min(8, out_c), out_c),
            nn.SiLU(),
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
