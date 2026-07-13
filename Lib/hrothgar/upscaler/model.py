"""Glyph-aware super-resolution model.

The model is intentionally lightweight and uses two forms of conditioning:

1. **Style conditioning** — a learned embedding of a font's visual style
   derived from K existing high-resolution glyphs, applied via FiLM after
   the residual body (close to pixel-level decisions about terminals and
   corners).
2. **Learned fallback** — when no style references are provided, a learned
   parameter is used in place of the style encoder output, so the model
   can still produce reasonable output.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from hrothgar.utils import SaveLoadModel


@dataclass
class UpscalerConfig:
    """Configuration for ``UpscalerModel``."""

    low_res_size: int = 128
    high_res_size: int = 512
    base_channels: int = 64
    num_residual_blocks: int = 8
    use_style_conditioning: bool = True
    style_reference_count: int = 4
    style_embedding_dim: int = 256

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
        if self.style_reference_count < 0:
            raise ValueError(
                f"style_reference_count must be non-negative "
                f"(got {self.style_reference_count})"
            )
        if self.style_embedding_dim <= 0:
            raise ValueError(
                f"style_embedding_dim must be positive "
                f"(got {self.style_embedding_dim})"
            )

    @property
    def upscale_factor(self) -> int:
        return self.high_res_size // self.low_res_size

    def save_sidecar(self, model_path):
        """Save config as a sidecar JSON alongside the model weights."""
        from pathlib import Path as _Path
        import json as _json
        from dataclasses import asdict as _asdict
        config_path = _Path(str(model_path).replace('.pth', '.conf.json'))
        with config_path.open('w', encoding='utf-8') as f:
            _json.dump(_asdict(self), f, indent=2, sort_keys=True)
        print(f'Saved upscaler config to {config_path}')

    @classmethod
    def from_sidecar(cls, model_path):
        """Load config from a sidecar JSON alongside the model weights."""
        from pathlib import Path as _Path
        import json as _json
        config_path = _Path(model_path).with_suffix('.conf.json')
        if not config_path.exists():
            config_path = _Path(str(model_path).replace('.pth', '.conf.json'))
        if not config_path.exists():
            raise FileNotFoundError(
                f'Upscaler config sidecar not found: {config_path}\n'
                'Run upscaler training first so the .conf.json is written '
                'alongside the .pth.'
            )
        with config_path.open('r', encoding='utf-8') as f:
            data = _json.load(f)
        return cls(**data)


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


class GlyphStyleEncoder(nn.Module):
    """Encode a font's visual style from a set of K reference glyph rasters.

    Each reference is independently passed through a lightweight conv backbone.
    Features are mean-pooled across references to produce a single style vector,
    which is then projected to FiLM (γ, β) parameters for channel-wise
    conditioning of the upscaler body.

    This is designed to capture global style properties — stroke contrast,
    terminal sharpness, corner treatment, serif presence — from high-resolution
    exemplars of an existing font, and inject that knowledge to disambiguate
    the subpixel decisions the upscaler must make at terminals and corners.
    """

    def __init__(self, base_channels: int, style_dim: int = 256) -> None:
        super().__init__()
        # Shared conv backbone: 512→256→128→64→32, then global pool.
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(
            nn.Linear(256, style_dim),
            nn.ReLU(inplace=True),
            nn.Linear(style_dim, base_channels * 2),
        )

    def forward(self, references: torch.Tensor) -> torch.Tensor:
        """Encode K reference glyphs into FiLM parameters.

        Args:
            references: ``(B, K, 3, H, W)`` tensor of high-res glyph rasters
               from the target font.

        Returns:
            ``(B, base_channels * 2)`` — concatenated γ, β for FiLM.
        """
        B, K, C, H, W = references.shape
        refs_flat = references.reshape(B * K, C, H, W)
        features = self.backbone(refs_flat)        # (B*K, 256, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (B*K, 256)
        features = features.view(B, K, 256)
        pooled = features.mean(dim=1)               # (B, 256)
        return self.projection(pooled)              # (B, base_channels * 2)


class UpscalerModel(SaveLoadModel):
    """A lightweight super-resolution model for glyph rasters.

    Style conditioning (via reference glyphs) provides the primary signal
    for disambiguating subpixel decisions.  A learned fallback embedding
    is used when no style references are available.
    """

    def __init__(self, config: UpscalerConfig) -> None:
        super().__init__()
        self.config = config
        self.use_style_conditioning = config.use_style_conditioning

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

        # --- Style conditioning ---
        self.style_encoder: Optional[GlyphStyleEncoder] = None
        self._no_style_embedding: Optional[nn.Parameter] = None
        self._init_style_conditioning()

    # ------------------------------------------------------------------
    # Style conditioning
    # ------------------------------------------------------------------

    def _init_style_conditioning(self) -> None:
        if not self.use_style_conditioning:
            return

        self.style_encoder = GlyphStyleEncoder(
            base_channels=self.config.base_channels,
            style_dim=self.config.style_embedding_dim,
        )
        self._no_style_embedding = nn.Parameter(
            torch.zeros(1, self.config.style_embedding_dim)
        )

    def _style_conditioning_vector(
        self, style_references: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Produce the FiLM (γ, β) tensor from style references.

        Falls back to the learned no-style embedding when references are absent.
        """
        if not self.use_style_conditioning or self.style_encoder is None:
            return None

        if style_references is None:
            return self._style_encoder_fallback(style_references)

        return self.style_encoder(style_references)

    def _style_encoder_fallback(
        self, _refs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return FiLM parameters from the learned no-style embedding."""
        assert self.style_encoder is not None
        assert self._no_style_embedding is not None
        style_vec = self._no_style_embedding  # (1, style_dim)
        return self.style_encoder.projection(style_vec)

    def _apply_style_conditioning(
        self,
        x: torch.Tensor,
        style_references: Optional[torch.Tensor],
    ) -> torch.Tensor:
        gamma_beta = self._style_conditioning_vector(style_references)
        if gamma_beta is None:
            return x

        if gamma_beta.shape[0] == 1 and x.shape[0] > 1:
            gamma_beta = gamma_beta.expand(x.shape[0], -1)

        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        low_res: torch.Tensor,
        style_references: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Upscale a low-resolution glyph raster.

        Args:
            low_res: ``(B, 3, low_res_size, low_res_size)`` input rasters.
            style_references: Optional ``(B, K, 3, high_res_size, high_res_size)``
                tensor of existing glyphs from the target font, used to encode
                the font's visual style.  Pass ``None`` to use the learned
                no-style fallback.

        Returns:
            ``(B, 3, high_res_size, high_res_size)`` upscaled glyphs in [0, 1].
        """
        x = self.input_projection(low_res)

        residual = self.body_projection(self.residual_body(x))
        x = x + residual

        x = self._apply_style_conditioning(x, style_references)

        x = self.upsampler(x)
        x = self.output_head(x)
        return torch.sigmoid(x)


__all__ = ["UpscalerConfig", "UpscalerModel"]
