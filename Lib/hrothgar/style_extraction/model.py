"""Style autoencoder: encode a font's glyph set into a rich style map and
decode a target codepoint back into a glyph image.

Design notes
------------
- **Content-invariance via glyph pooling, not Gram.**  Each glyph is encoded
  into a *spatially-preserved* feature map, then pooled *across glyphs*.  This
  cancels per-glyph content (which varies across glyphs) while keeping style
  (which is shared) *and* keeping spatial structure (which is where fine
  geometric detail — terminal angles, serifs, stroke weight — lives).
- **Reconstruction is the contract.**  The embedding is "rich" iff the decoder
  can reconstruct high-fidelity glyphs from ``(codepoint, style_map)``.  There
  is no separate codebook or tokenizer.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hrothgar.style_extraction.config import StyleExtractionConfig
from hrothgar.utils import SaveLoadModel


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    return nn.GroupNorm(groups, channels)


class PerGlyphEncoder(nn.Module):
    """Shared CNN: ``(1, H, W)`` glyph -> ``(C, H/ds, W/ds)`` feature map."""

    def __init__(self, base_channels: int, out_dim: int, downsample: int) -> None:
        super().__init__()
        if downsample not in (2, 4, 8, 16):
            raise ValueError(f"downsample must be 2/4/8/16, got {downsample}")

        layers: list[nn.Module] = []
        c = 1
        layers += [
            nn.Conv2d(c, base_channels, 3, padding=1),
            _group_norm(base_channels),
            nn.SiLU(),
        ]
        c = base_channels
        ds = downsample
        while ds >= 2:
            layers += [
                nn.Conv2d(c, c * 2, 3, stride=2, padding=1),
                _group_norm(c * 2),
                nn.SiLU(),
            ]
            c *= 2
            ds //= 2
        layers.append(nn.Conv2d(c, out_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = _group_norm(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = _group_norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class _UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = _group_norm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.norm(self.conv(self.upsample(x))))


class GlyphDecoder(nn.Module):
    """Render ``(codepoint, style_map)`` -> glyph image.

    The codepoint embedding is broadcast to a spatial map and concatenated
    with the (spatially-aligned) style map, then a CNN upsamples to the glyph.
    """

    def __init__(
        self, in_channels: int, base_channels: int, num_res_blocks: int, downsample: int
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.blocks = nn.Sequential(
            *[_ResBlock(base_channels) for _ in range(num_res_blocks)]
        )
        num_ups = int(math.log2(downsample))
        self.ups = nn.ModuleList(
            [_UpsampleBlock(base_channels, base_channels) for _ in range(num_ups)]
        )
        self.norm = _group_norm(base_channels)
        self.out = nn.Conv2d(base_channels, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.blocks(x)
        for up in self.ups:
            x = up(x)
        x = self.out(F.silu(self.norm(x)))
        return torch.sigmoid(x)


class StyleExtractionModel(SaveLoadModel):
    """Font-level style autoencoder.

    ``encode_style`` maps a set of glyph images to a spatial style map;
    ``decode`` renders a target codepoint in that style.
    """

    def __init__(self, config: StyleExtractionConfig) -> None:
        super().__init__()
        self.config = config

        self.glyph_encoder = PerGlyphEncoder(
            base_channels=config.glyph_encoder_base_channels,
            out_dim=config.glyph_encoder_feature_dim,
            downsample=config.glyph_encoder_downsample,
        )

        # Codepoint identity -> embedding.  Content is just "which glyph".
        self.codepoint_embedding = nn.Embedding(
            config.num_codepoints, config.glyph_encoder_feature_dim
        )

        # Decoder input = [codepoint_map (C) | style_map (C)] along channels.
        self.decoder = GlyphDecoder(
            in_channels=2 * config.glyph_encoder_feature_dim,
            base_channels=config.decoder_base_channels,
            num_res_blocks=config.decoder_num_res_blocks,
            downsample=config.glyph_encoder_downsample,
        )

    def encode_style(
        self,
        style_images: torch.Tensor,
        style_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a font's glyph set into a spatial style map.

        Args:
            style_images: ``(B, G, 1, H, W)`` greyscale glyph renderings in [0, 1].
            style_mask: Optional ``(B, G)`` boolean; ``True`` = glyph visible.

        Returns:
            ``(B, C, h, w)`` style feature map.
        """
        b, g, c, h, w = style_images.shape
        assert c == 1, f"Expected single-channel glyphs, got {c} channels"

        flat = style_images.flatten(0, 1)  # (B*G, 1, H, W)
        features = self.glyph_encoder(flat)  # (B*G, C, h', w')
        f_c, f_h, f_w = features.shape[1:]
        features = features.reshape(b, g, f_c, f_h, f_w)  # (B, G, C, h', w')

        # Pool across glyphs to remove content, keep spatial style structure.
        if style_mask is not None:
            mask = style_mask.to(features.dtype).reshape(b, g, 1, 1, 1)
            style_map = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            style_map = features.mean(dim=1)
        return style_map  # (B, C, h', w')

    def build_conditioning(
        self,
        target_codepoint_idx: torch.Tensor,
        style_map: torch.Tensor,
    ) -> torch.Tensor:
        """Return the conditioning map ``(B, 2C, h, w)`` consumed by the decoder.

        This is ``[codepoint_map | style_map]`` along the channel axis.  The
        same map is used to condition the discriminator, so it can reject
        outputs that don't match the requested codepoint + style.
        """
        _, c, h, w = style_map.shape
        codepoint_emb = self.codepoint_embedding(target_codepoint_idx)  # (B, C)
        codepoint_map = codepoint_emb[:, :, None, None].expand(-1, -1, h, w)  # (B, C, h, w)
        return torch.cat([codepoint_map, style_map], dim=1)  # (B, 2C, h, w)

    def decode(
        self,
        target_codepoint_idx: torch.Tensor,
        style_map: torch.Tensor,
    ) -> torch.Tensor:
        """Render a target codepoint in the given style.

        Args:
            target_codepoint_idx: ``(B,)`` long indices into the codepoint table.
            style_map: ``(B, C, h, w)`` style feature map.

        Returns:
            ``(B, 1, H, W)`` glyph images in [0, 1].
        """
        x = self.build_conditioning(target_codepoint_idx, style_map)
        return self.decoder(x)  # (B, 1, H, W)

    def forward(
        self,
        style_images: torch.Tensor,
        target_codepoint_idx: torch.Tensor,
        style_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        style_map = self.encode_style(style_images, style_mask=style_mask)
        return self.decode(target_codepoint_idx, style_map)
