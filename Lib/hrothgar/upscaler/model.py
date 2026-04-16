"""Glyph-aware super-resolution model prototype.

The model is intentionally lightweight and supports optional conditioning from a
pretrained G-Tok encoder. This lets us test whether tokenizer features improve
edge fidelity during upscaling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hrothgar.gtok.model import GtokConfig, GtokModel
from hrothgar.utils import SaveLoadModel


@dataclass
class UpscalerConfig:
    """Configuration for ``UpscalerModel``."""

    low_res_size: int = 128
    high_res_size: int = 512
    base_channels: int = 64
    num_residual_blocks: int = 8
    use_gtok_encoder: bool = True
    use_gtok_vit_features: bool = True
    gtok_model_path: Optional[str] = "models/gtok_model.pth"

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
        self.use_gtok_encoder = config.use_gtok_encoder
        self.use_gtok_vit_features = config.use_gtok_vit_features

        self.input_projection = nn.Conv2d(3, config.base_channels, 3, 1, 1)

        self.residual_body = nn.Sequential(
            *[ResidualBlock(config.base_channels) for _ in range(config.num_residual_blocks)]
        )

        self.body_projection = nn.Conv2d(config.base_channels, config.base_channels, 3, 1, 1)

        num_upsample_stages = config.upscale_factor.bit_length() - 1
        if 2**num_upsample_stages != config.upscale_factor:
            raise ValueError(
                "upscale_factor must be a power of two "
                f"(got {config.upscale_factor})"
            )
        self.upsampler = nn.Sequential(
            *[PixelShuffleUpsample(config.base_channels) for _ in range(num_upsample_stages)]
        )
        self.output_head = nn.Conv2d(config.base_channels, 3, 3, 1, 1)

        self.gtok: Optional[GtokModel] = None
        self.gtok_feature_adapter: Optional[nn.Module] = None
        self._init_gtok_conditioning()

    def _init_gtok_conditioning(self) -> None:
        if not self.use_gtok_encoder:
            return

        gtok_config = GtokConfig(image_size=self.config.low_res_size)
        self.gtok = GtokModel(gtok_config)
        if self.config.gtok_model_path and os.path.exists(self.config.gtok_model_path):
            state_dict = torch.load(self.config.gtok_model_path, map_location="cpu")
            self.gtok.load_state_dict(state_dict)

        for param in self.gtok.parameters():
            param.requires_grad = False
        self.gtok.eval()

        gtok_feature_dim = gtok_config.cnn_latent_channels
        if self.use_gtok_vit_features:
            gtok_feature_dim += gtok_config.vit_hidden_dim

        self.gtok_feature_adapter = nn.Sequential(
            nn.Conv2d(gtok_feature_dim, self.config.base_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.config.base_channels, self.config.base_channels, kernel_size=1),
        )

    def _extract_gtok_feature_map(self, low_res: torch.Tensor) -> Optional[torch.Tensor]:
        if self.gtok is None:
            return None

        with torch.no_grad():
            cnn_features = self.gtok.cnn_encoder(low_res)

            if not self.use_gtok_vit_features:
                features = cnn_features
            else:
                b, c, h, w = cnn_features.shape
                cnn_sequence = cnn_features.permute(0, 2, 3, 1).reshape(b, h * w, c)
                vit_tokens = self.gtok.vit_encoder(cnn_sequence)[:, 1:, :]
                vit_features = vit_tokens.reshape(b, h, w, -1).permute(0, 3, 1, 2)
                features = torch.cat([cnn_features, vit_features], dim=1)

        if self.gtok_feature_adapter is None:
            return None

        adapted = self.gtok_feature_adapter(features)
        return F.interpolate(
            adapted,
            size=(self.config.low_res_size, self.config.low_res_size),
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, low_res: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(low_res)

        gtok_features = self._extract_gtok_feature_map(low_res)
        if gtok_features is not None:
            x = x + gtok_features

        residual = self.body_projection(self.residual_body(x))
        x = x + residual
        x = self.upsampler(x)
        x = self.output_head(x)
        return torch.sigmoid(x)


__all__ = ["UpscalerConfig", "UpscalerModel"]
