"""Upstream GAR-Font StyleEncoder — copied wholesale from GAR-Font/model/generator/generator.py.

Extracts a compact style representation from reference glyph images using
InstanceNorm, reflective padding, and residual blocks with ``scale_var``
stabilisation.  Designed to work with downstream FeatureFusionModule + GPT.
"""

from __future__ import annotations

import math
from functools import partial

import torch.nn as nn

from hrothgar.upstream.blocks import ConvBlock, ResBlock


class StyleEncoder(nn.Module):
    """Style encoder from the GAR-Font generator.

    Architecture:
        Conv(in) → (downsample_times-1)× Conv(downsample) →
        2× ResBlock → ResBlock(downsample) → ResBlock → (optional Sigmoid)

    Uses InstanceNorm (``norm='in'``) throughout, which normalises each sample
    independently and is important for extracting style rather than content.

    Args:
        C_in: Input channels (3 for RGB).
        C: Base channel count (default 32, matches paper).
        C_out: Output feature channels (must match ``encoder_feature_dim``).
        norm: Normalisation type for most layers.
        activ: Activation function.
        pad_type: Convolution padding mode (``'reflect'`` is the paper default).
        sigmoid: Apply ``Sigmoid`` to output if ``True``.
        scale_var: Divide residual branches by ``√2`` for training stability.
        downsample_ratio: Total spatial reduction factor (8 or 16).
    """

    def __init__(
        self,
        C_in: int = 3,
        C: int = 32,
        C_out: int = 256,
        norm: str = "in",
        activ: str = "relu",
        pad_type: str = "reflect",
        sigmoid: bool = False,
        scale_var: bool = True,
        downsample_ratio: int = 8,
    ) -> None:
        super().__init__()
        if downsample_ratio not in (8, 16):
            raise ValueError(
                f"downsample_ratio must be 8 or 16, got {downsample_ratio}"
            )
        downsample_times = int(math.log2(downsample_ratio))

        ConvBlk = partial(ConvBlock, norm=norm, activ=activ, pad_type=pad_type)
        ResBlk = partial(ResBlock, norm=norm, activ=activ, scale_var=scale_var)

        layers: list[nn.Module] = []
        layers.append(ConvBlk(C_in, C, 3, 1, 1, norm="in", activ="relu"))

        in_ch = C
        for _ in range(downsample_times - 1):
            out_ch = in_ch * 2
            layers.append(ConvBlk(in_ch, out_ch, 3, 1, 1, downsample=True))
            in_ch = out_ch

        layers.append(ResBlk(in_ch, in_ch, 3, 1))
        layers.append(ResBlk(in_ch, in_ch, 3, 1))
        layers.append(ResBlk(in_ch, 2 * in_ch, 3, 1, downsample=True))

        layers.append(ResBlk(2 * in_ch, C_out))

        self.net = nn.Sequential(*layers)
        self.if_sigmoid = sigmoid

    def forward(self, x):
        out = self.net(x)
        if self.if_sigmoid:
            out = nn.Sigmoid()(out)
        return out
