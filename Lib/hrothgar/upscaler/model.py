"""Glyph super-resolution model.

A lightweight, **content-preserving** upscaler: it de-aliases a crop-to-ink
``low_res_size`` glyph raster to ``high_res_size`` for cleaner vectorization,
without changing the glyph's construction.  The construction is already decided
by the upstream diffusion model; this model only interpolates the raster.

There is deliberately no style conditioning — the earlier style-reference path
existed to *repair* fine detail the AR generator could not produce.  The
diffusion model already emits correct terminals/corners, and conditioning on
the font's native reference glyphs would push the output toward the font's GT
construction, i.e. *undo* the diffusion model's (valid) design choices.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from hrothgar.utils import SaveLoadModel


@dataclass
class UpscalerConfig:
    """Configuration for ``UpscalerModel``."""

    low_res_size: int = 128
    high_res_size: int = 512
    base_channels: int = 64
    num_residual_blocks: int = 8

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

    def save_sidecar(self, model_path):
        """Save config as a sidecar JSON alongside the model weights."""
        import json as _json
        from dataclasses import asdict as _asdict
        from pathlib import Path as _Path

        from hrothgar.utils import git_short_sha

        config_path = _Path(str(model_path).replace(".pth", ".conf.json"))
        data = _asdict(self)
        data["git_sha"] = git_short_sha()
        with config_path.open("w", encoding="utf-8") as f:
            _json.dump(data, f, indent=2, sort_keys=True)
        print(f"Saved upscaler config to {config_path}")

    @classmethod
    def from_sidecar(cls, model_path):
        """Load config from a sidecar JSON alongside the model weights."""
        import json as _json
        from pathlib import Path as _Path

        config_path = _Path(model_path).with_suffix(".conf.json")
        if not config_path.exists():
            config_path = _Path(str(model_path).replace(".pth", ".conf.json"))
        if not config_path.exists():
            raise FileNotFoundError(
                f"Upscaler config sidecar not found: {config_path}\n"
                "Run upscaler training first so the .conf.json is written "
                "alongside the .pth."
            )
        with config_path.open("r", encoding="utf-8") as f:
            data = _json.load(f)
        import dataclasses as _dc

        known = {f.name for f in _dc.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


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
    """A lightweight content-preserving super-resolution model for glyph rasters."""

    def __init__(self, config: UpscalerConfig) -> None:
        super().__init__()
        self.config = config

        self.input_projection = nn.Conv2d(1, config.base_channels, 3, 1, 1)

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
                f"upscale_factor must be a power of two (got {config.upscale_factor})"
            )
        self.upsampler = nn.Sequential(
            *[
                PixelShuffleUpsample(config.base_channels)
                for _ in range(num_upsample_stages)
            ]
        )
        self.output_head = nn.Conv2d(config.base_channels, 1, 3, 1, 1)

    def forward(self, low_res: torch.Tensor) -> torch.Tensor:
        """Upscale a low-resolution glyph raster.

        Args:
            low_res: ``(B, 1, low_res_size, low_res_size)`` input rasters in
                ``[0, 1]`` (0 = ink, 1 = white).

        Returns:
            ``(B, 1, high_res_size, high_res_size)`` upscaled glyphs in ``[0, 1]``.
        """
        x = self.input_projection(low_res)
        x = x + self.body_projection(self.residual_body(x))
        x = self.upsampler(x)
        x = self.output_head(x)
        return torch.sigmoid(x)


__all__ = ["UpscalerConfig", "UpscalerModel"]
