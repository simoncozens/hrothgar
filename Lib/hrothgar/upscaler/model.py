"""Glyph-aware super-resolution model prototype.

The model is intentionally lightweight and supports optional conditioning from
a description encoder (text-based font metadata).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from hrothgar.ar.multimodal import (
    HashedDescriptionEncoder,
    HashedDescriptionEncoderConfig,
)
from hrothgar.utils import SaveLoadModel


@dataclass
class UpscalerConfig:
    """Configuration for ``UpscalerModel``."""

    low_res_size: int = 128
    high_res_size: int = 512
    base_channels: int = 64
    num_residual_blocks: int = 8
    use_description_conditioning: bool = True
    description_fallback_embedding_dim: int = 512

    def __post_init__(self) -> None:
        if self.low_res_size <= 0 or self.high_res_size <= 0:
            raise ValueError("low_res_size and high_res_size must be positive")
        if self.high_res_size <= self.low_res_size:
            raise ValueError("high_res_size must be greater than low_res_size")
        if self.high_res_size % self.low_res_size != 0:
            raise ValueError(
                "high_res_size must be divisible by low_res_size "
                f"(got {self.high_res_size} and {self.low_res_size})"
            )

    @property
    def upscale_factor(self) -> int:
        return self.high_res_size // self.low_res_size


class ResidualBlock(nn.Module):
    """Simple residual block used by the SR body."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class PixelShuffleUpsample(nn.Module):
    """One 2x upsampling stage based on pixel shuffle."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpscalerModel(SaveLoadModel):
    """A lightweight super-resolution model for glyph rasters."""

    def __init__(self, config: UpscalerConfig) -> None:
        super().__init__()
        self.config = config
        self.use_description_conditioning = config.use_description_conditioning

        self.input_projection = nn.Conv2d(3, config.base_channels, 3, 1, 1)

        self.residual_body = nn.Sequential(
            *[
                ResidualBlock(config.base_channels)
                for _ in range(config.num_residual_blocks)
            ]
        )

        self.body_projection = nn.Conv2d(
            config.base_channels, config.base_channels, 3, 1, 1
        )

        num_upsample_stages = config.upscale_factor.bit_length() - 1
        if 2**num_upsample_stages != config.upscale_factor:
            raise ValueError(
                "upscale_factor must be a power of two "
                f"(got {config.upscale_factor})"
            )
        self.upsampler = nn.Sequential(
            *[
                PixelShuffleUpsample(config.base_channels)
                for _ in range(num_upsample_stages)
            ]
        )
        self.output_head = nn.Conv2d(config.base_channels, 3, 3, 1, 1)

        self.description_encoder: Optional[HashedDescriptionEncoder] = None
        self.description_feature_adapter: Optional[nn.Module] = None
        self._init_description_conditioning()

    def _init_description_conditioning(self) -> None:
        if not self.use_description_conditioning:
            return

        self.description_encoder = HashedDescriptionEncoder(
            HashedDescriptionEncoderConfig(
                embedding_dim=self.config.description_fallback_embedding_dim,
            )
        )
        description_embedding_dim = self.config.description_fallback_embedding_dim
        self.description_feature_adapter = nn.Sequential(
            nn.Linear(description_embedding_dim, self.config.base_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.base_channels, self.config.base_channels * 2),
        )

    def _description_embedding(
        self, descriptions: Optional[Sequence[str]], device: torch.device
    ) -> Optional[torch.Tensor]:
        if not self.use_description_conditioning or descriptions is None:
            return None
        if self.description_encoder is None:
            return None

        token_embeddings = self.description_encoder(list(descriptions)).to(device)
        return token_embeddings.mean(dim=1)

    def _apply_description_conditioning(
        self,
        x: torch.Tensor,
        descriptions: Optional[Sequence[str]],
    ) -> torch.Tensor:
        if self.description_feature_adapter is None:
            return x

        description_embedding = self._description_embedding(descriptions, x.device)
        if description_embedding is None:
            return x

        gamma_beta = self.description_feature_adapter(description_embedding)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta

    def forward(
        self,
        low_res: torch.Tensor,
        descriptions: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        x = self.input_projection(low_res)

        x = self._apply_description_conditioning(x, descriptions)

        residual = self.body_projection(self.residual_body(x))
        x = x + residual
        x = self.upsampler(x)
        x = self.output_head(x)
        return torch.sigmoid(x)


__all__ = ["UpscalerConfig", "UpscalerModel"]
