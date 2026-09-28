"""Deterministic exemplar-conditioned CNN for weight adjustment.

Three parts:

* :class:`StyleEncoder` — encodes ``K`` ``(regular, bold)`` exemplar pairs into
  a single style vector describing "how this font bolds."
* :class:`Decoder` — a residual CNN that takes the target's *regular* raster and
  the style vector and predicts the bold raster as ``regular + residual``.
  Residual output keeps the skeleton correct by construction; the style vector
  is projected to a small number of channels and concatenated with the input so
  it modulates *how* the strokes thicken.
* :class:`AdvanceHead` — predicts the advance-width delta (bold - regular) from
  the style vector and the regular's advance, the one geometry label not visible
  in the raster.

The model is deterministic on purpose: determinism is what makes thin/bold
masters generated from the same regular mutually consistent (interpolatable),
which a stochastic sampler would not guarantee.
"""

from __future__ import annotations

import torch
from torch import nn

from hrothgar.utils import SaveLoadModel
from hrothgar.weight_adjust.config import WeightAdjustConfig


class ResidualBlock(nn.Module):
    """Simple residual block (same shape as the upscaler's)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class StyleEncoder(nn.Module):
    """Encode ``K`` (regular, bold) exemplar pairs into one style vector.

    Input is ``(B, K, 2, H, W)`` — each exemplar is the pair stacked as two
    channels (regular, bold).  A shared CNN encodes each pair independently,
    then the ``K`` encodings are mean-pooled into ``(B, style_dim)``.
    """

    def __init__(
        self, style_dim: int = 128, base_channels: int = 32, dropout: float = 0.1
    ) -> None:
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(2, c, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c, c * 2, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 4, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c * 4, c * 4, 3, 2, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c * 4, style_dim),
        )

    def forward(self, exemplars: torch.Tensor) -> torch.Tensor:
        b, k, c, h, w = exemplars.shape
        x = exemplars.reshape(b * k, c, h, w)
        feat = self.net(x)  # (B*K, style_dim)
        return feat.reshape(b, k, -1).mean(dim=1)  # (B, style_dim)


class Decoder(nn.Module):
    """Residual decoder: ``regular + style -> bold``.

    The style vector (plus the target-weight scalar) is projected to a small
    spatial bias (16 channels) and concatenated with the regular raster before
    the residual body, so the same body can thicken differently per font and
    per target weight.
    """

    def __init__(
        self,
        style_dim: int = 128,
        base_channels: int = 64,
        num_blocks: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # style_dim channels from the encoder + 1 target-weight scalar.
        self.style_proj = nn.Linear(style_dim + 1, 16)
        self.input_proj = nn.Conv2d(1 + 16, base_channels, 3, 1, 1)
        self.dropout = nn.Dropout2d(dropout)
        self.body = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(num_blocks)]
        )
        self.output_head = nn.Conv2d(base_channels, 1, 3, 1, 1)

    def forward(self, regular: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        s = self.style_proj(cond)[:, :, None, None]
        s = s.expand(-1, -1, regular.shape[-2], regular.shape[-1])
        x = torch.cat([regular, s], dim=1)
        x = self.input_proj(x)
        x = self.dropout(x)
        x = self.body(x)
        residual = self.output_head(x)
        return (regular + residual).clamp(0.0, 1.0)


class AdvanceHead(nn.Module):
    """Predict the advance-width delta (bold - regular) in em units."""

    def __init__(self, style_dim: int = 128, hidden: int = 64) -> None:
        super().__init__()
        # style + weight scalar + regular advance.
        self.net = nn.Sequential(
            nn.Linear(style_dim + 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, cond: torch.Tensor, regular_advance: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([cond, regular_advance[:, None]], dim=-1)
        return self.net(x).squeeze(-1)  # (B,)


class WeightAdjustModel(SaveLoadModel):
    """Facade: ``(regular, exemplars, regular_advance, weight) -> (bold, advance_delta)``.

    ``weight`` is the *normalized* target weight (regular = 0).  It is
    concatenated with the style vector so both the decoder and the advance head
    can scale the font's bolding direction to the requested weight.
    """

    def __init__(self, config: WeightAdjustConfig) -> None:
        super().__init__()
        self.config = config
        self.style_encoder = StyleEncoder(
            config.style_dim, config.style_base_channels, config.dropout
        )
        self.decoder = Decoder(
            config.style_dim,
            config.decoder_base_channels,
            config.decoder_num_blocks,
            config.dropout,
        )
        self.advance_head = AdvanceHead(config.style_dim)

    def forward(
        self,
        regular: torch.Tensor,
        exemplars: torch.Tensor,
        regular_advance: torch.Tensor,
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        style = self.style_encoder(exemplars)
        cond = torch.cat([style, weight[:, None]], dim=-1)
        bold = self.decoder(regular, cond)
        delta = self.advance_head(cond, regular_advance)
        return bold, delta


def build_model(config: WeightAdjustConfig) -> WeightAdjustModel:
    return WeightAdjustModel(config)
