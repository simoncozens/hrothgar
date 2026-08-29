"""Style autoencoder v2: Perceiver style tokens + cross-attention decoder.

Replaces v1's mean-pooled ``(C, h, w)`` style map with a set of learned style
tokens produced by a Perceiver over the reference glyphs, and a decoder that
*cross-attends* to those tokens.  This gives each output region local access to
style detail (terminal angle, serif shape, stroke curvature) instead of a
single global average — the v1 bottleneck that left ``glyphloss`` flat.

Memory note
-----------
The Perceiver attends ``K`` latent queries to ``G * N`` reference tokens
(``G`` evidence glyphs × ``N`` spatial positions).  ``G * N`` grows quickly, so
keep ``num_evidence_glyphs`` and ``glyph_encoder_downsample`` modest (defaults
here are ``G=16``, downsample 4 → ``N=1024``).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hrothgar.style_extraction.config import StyleExtractionV2Config
from hrothgar.utils import SaveLoadModel


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, channels), channels)


class PerGlyphEncoder(nn.Module):
    """Shared CNN: ``(1, H, W)`` -> ``(C, H/ds, W/ds)`` spatial feature map."""

    def __init__(self, base_channels: int, out_dim: int, downsample: int) -> None:
        super().__init__()
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


class MultiHeadAttention(nn.Module):
    """Multi-head attention via ``scaled_dot_product_attention`` (flash-capable)."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = dropout

    def forward(self, query, key, value):
        b, nq, _ = query.shape
        nkv = key.shape[1]
        q = self.q_proj(query).view(b, nq, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(b, nkv, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(b, nkv, self.heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(b, nq, -1)
        return self.out_proj(out)


class _FFN(nn.Module):
    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mult * dim), nn.GELU(), nn.Linear(mult * dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class PerceiverBlock(nn.Module):
    """Latent queries cross-attend to reference tokens, then to each other.

    The canonical Perceiver alternates cross-attention (latents ↔ inputs) with
    self-attention (latents ↔ latents).  Without the self-attention step the K
    latents are processed independently in parallel and collapse to near-
    identical summaries of the same inputs; self-attention lets them coordinate
    and specialise onto distinct style axes.
    """

    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.cross = MultiHeadAttention(dim, heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = _FFN(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, latents, inputs):
        latents = self.norm1(latents + self.cross(latents, inputs, inputs))
        latents = self.norm2(latents + self.self_attn(latents, latents, latents))
        latents = self.norm3(latents + self.ffn(latents))
        return latents


class DecoderBlock(nn.Module):
    """Content queries: self-attn, cross-attn to style tokens, then FFN."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = _FFN(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, style_tokens):
        x = self.norm1(x + self.self_attn(x, x, x))
        x = self.norm2(x + self.cross_attn(x, style_tokens, style_tokens))
        x = self.norm3(x + self.ffn(x))
        return x


class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = _group_norm(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = _group_norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class _UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = _group_norm(out_channels)

    def forward(self, x):
        return F.silu(self.norm(self.conv(self.up(x))))


class CNNHead(nn.Module):
    """``(B, D, h, w)`` -> ``(B, 1, H, W)``."""

    def __init__(self, in_channels, base_channels, num_res_blocks, downsample):
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

    def forward(self, x):
        x = self.proj(x)
        x = self.blocks(x)
        for up in self.ups:
            x = up(x)
        return torch.sigmoid(self.out(F.silu(self.norm(x))))


class StyleExtractionModelV2(SaveLoadModel):
    """Font-level style autoencoder with Perceiver + cross-attention decoding."""

    def __init__(self, config: StyleExtractionV2Config) -> None:
        super().__init__()
        self.config = config
        d = config.glyph_encoder_feature_dim
        self.grid_size = config.image_size // config.glyph_encoder_downsample
        self.grid_n = self.grid_size ** 2

        self.glyph_encoder = PerGlyphEncoder(
            config.glyph_encoder_base_channels, d, config.glyph_encoder_downsample
        )
        self.codepoint_embedding = nn.Embedding(config.num_codepoints, d)

        # Decoder content-query positional embedding: fine, per output-grid
        # position (the output grid is well-defined, so fine position is valid).
        self.query_pos_embed = nn.Parameter(torch.randn(self.grid_n, d) * 0.02)

        # Reference-token tags.  Codepoint identity is the semantic hook; a
        # coarse structural region replaces fine spatial position, which is
        # ill-posed across fonts because the skeleton itself is part of style.
        coarse = config.coarse_grid_size
        assert self.grid_size % coarse == 0
        self.coarse_region_embed = nn.Parameter(torch.randn(coarse * coarse, d) * 0.02)
        ys = torch.arange(self.grid_size) // (self.grid_size // coarse)
        xs = torch.arange(self.grid_size) // (self.grid_size // coarse)
        self.register_buffer(
            "coarse_idx", (ys[:, None] * coarse + xs[None, :]).reshape(-1)
        )

        # Perceiver: K latent tokens summarise the G*N reference tokens.
        self.style_latents = nn.Parameter(
            torch.randn(config.num_style_tokens, d) * 0.02
        )
        self.perceiver = nn.ModuleList(
            [
                PerceiverBlock(d, config.perceiver_num_heads, config.decoder_dropout)
                for _ in range(config.perceiver_num_layers)
            ]
        )

        # Cross-attention decoder over the style tokens.
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(d, config.decoder_num_heads, config.decoder_dropout)
                for _ in range(config.decoder_num_layers)
            ]
        )

        self.cnn_head = CNNHead(
            d,
            config.decoder_base_channels,
            config.decoder_num_res_blocks,
            config.glyph_encoder_downsample,
        )

    def encode_style(
        self,
        style_images: torch.Tensor,
        style_codepoint_idx: Optional[torch.Tensor] = None,
        style_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a font's glyph set into ``(B, K, D)`` style tokens.

        Args:
            style_images: ``(B, G, 1, H, W)`` greyscale glyph renderings in [0, 1].
            style_codepoint_idx: Optional ``(B, G)`` long indices identifying
                which codepoint each evidence glyph is (the semantic hook).
            style_mask: Optional ``(B, G)`` boolean; ``True`` = glyph visible.

        Returns:
            ``(B, K, D)`` Perceiver style tokens.
        """
        b, g, c, h, w = style_images.shape
        flat = style_images.flatten(0, 1)  # (B*G, 1, H, W)
        feats = self.glyph_encoder(flat)  # (B*G, D, gh, gw)
        d, gh, gw = feats.shape[1:]
        n = gh * gw
        feats = feats.reshape(b, g, d, gh, gw)  # (B, G, D, gh, gw)
        tokens = feats.permute(0, 1, 3, 4, 2).reshape(b, g, n, d)  # (B, G, n, D)

        # Coarse structural region tag (replaces fine spatial position).
        tokens = tokens + self.coarse_region_embed[self.coarse_idx][None, None]

        # Codepoint identity tag: which glyph this token came from.
        if style_codepoint_idx is not None:
            cp_emb = self.codepoint_embedding(style_codepoint_idx)  # (B, G, D)
            tokens = tokens + cp_emb[:, :, None, :]

        if style_mask is not None:
            mask = style_mask.to(tokens.dtype).reshape(b, g, 1, 1)
            tokens = tokens * mask
        tokens = tokens.reshape(b, g * n, d)  # (B, G*n, D)

        latents = self.style_latents[None].expand(b, -1, -1)  # (B, K, D)
        for blk in self.perceiver:
            latents = blk(latents, tokens)
        return latents  # (B, K, D)

    def decode(
        self,
        target_codepoint_idx: torch.Tensor,
        style_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Render a target codepoint in the given style tokens.

        Args:
            target_codepoint_idx: ``(B,)`` long indices into the codepoint table.
            style_tokens: ``(B, K, D)`` Perceiver style tokens.

        Returns:
            ``(B, 1, H, W)`` glyph images in [0, 1].
        """
        b = target_codepoint_idx.shape[0]
        d = self.config.glyph_encoder_feature_dim
        codepoint_emb = self.codepoint_embedding(target_codepoint_idx)  # (B, D)
        queries = codepoint_emb[:, None, :] + self.query_pos_embed[None]  # (B, n, D)

        for blk in self.decoder_blocks:
            queries = blk(queries, style_tokens)

        x = queries.transpose(1, 2).reshape(b, d, self.grid_size, self.grid_size)
        return self.cnn_head(x)  # (B, 1, H, W)

    def forward(
        self,
        style_images: torch.Tensor,
        target_codepoint_idx: torch.Tensor,
        style_codepoint_idx: Optional[torch.Tensor] = None,
        style_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        style_tokens = self.encode_style(
            style_images,
            style_codepoint_idx=style_codepoint_idx,
            style_mask=style_mask,
        )
        return self.decode(target_codepoint_idx, style_tokens)
