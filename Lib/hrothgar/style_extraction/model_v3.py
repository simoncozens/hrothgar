"""Style autoencoder v3: content-conditioned SPADE decoder.

v2's decoder collapsed its cross-attention to a *global* operation (``q_var ≈ 0``):
every output position attended to the style tokens the same way, so style entered
as a single global vector and could render coarse style (weight/slant/width) but
not localized fine detail (terminals, inktraps, counters).

v3 fixes that by:

* producing a *per-position* style map — the content queries (codepoint +
  position) cross-attend to the style tokens, giving a ``(B, D, grid, grid)``
  map, and
* applying that map through SPADE (spatially-adaptive normalization) in the CNN
  head, so style modulates each output position directly and locally.

Because the modulation map is derived from the *content query* attending to the
token *set* (not from a fixed spatial grid over the evidence), it stays aligned
to the output skeleton even though "skeleton is style" — the terminal can sit in
a different place for each font, and the content query asks for the right style
at each place.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hrothgar.style_extraction.config import StyleExtractionV3Config
from hrothgar.style_extraction.model_v2 import (
    MultiHeadAttention,
    PerceiverBlock,
    PerGlyphEncoder,
    _UpsampleBlock,
    _group_norm,
)
from hrothgar.utils import SaveLoadModel


class SelfAttnBlock(nn.Module):
    """Content-query self-attention + FFN (skeleton refinement)."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(dim, heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attn(x, x, x))
        x = self.norm2(x + self.ffn(x))
        return x


def _fixed_sinusoidal_pos_bias(
    grid_size: int, num_tokens: int, scale: float = 0.5
) -> torch.Tensor:
    """Fixed (non-learnable) sinusoidal positional bias of shape ``(nq, K)``.

    Position-dependent by construction — the model cannot learn it away, which is
    what we need to force cross-attention out of the global-collapse regime.
    """
    nq = grid_size * grid_size
    y = torch.arange(grid_size, dtype=torch.float32).reshape(-1, 1)
    x = torch.arange(grid_size, dtype=torch.float32).reshape(1, -1)
    yy = y.expand(grid_size, grid_size).reshape(-1) / max(grid_size - 1, 1)
    xx = x.expand(grid_size, grid_size).reshape(-1) / max(grid_size - 1, 1)
    sigs = []
    for octave in range(4):
        f = 2.0 ** octave
        sigs += [
            torch.sin(2 * math.pi * f * yy),
            torch.cos(2 * math.pi * f * yy),
            torch.sin(2 * math.pi * f * xx),
            torch.cos(2 * math.pi * f * xx),
        ]
    sig = torch.stack(sigs, dim=1)  # (nq, 16)
    proj = torch.randn(sig.shape[1], num_tokens) * scale  # (16, K)
    return sig @ proj  # (nq, K)


class CrossAttentionWithWeights(nn.Module):
    """Cross-attention that also returns the (post-softmax) attention weights.

    The weights are needed to compute the position-dependence guard (``q_var``).
    When ``use_pos_bias`` and not ``pos_bias_trainable``, a *fixed* sinusoidal
    positional bias is added to the logits — non-learnable, so it cannot collapse
    to position-independent like a learned bias does.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.0,
        nq: Optional[int] = None,
        num_tokens: Optional[int] = None,
        grid_size: Optional[int] = None,
        use_pos_bias: bool = False,
        pos_bias_trainable: bool = False,
        pos_bias_scale: float = 0.5,
    ) -> None:
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = dropout
        if use_pos_bias:
            assert nq is not None and num_tokens is not None
            if pos_bias_trainable:
                self.pos_bias = nn.Parameter(torch.randn(nq, num_tokens) * 0.02)
            else:
                assert grid_size is not None
                self.register_buffer(
                    "pos_bias",
                    _fixed_sinusoidal_pos_bias(grid_size, num_tokens, pos_bias_scale),
                )
        else:
            self.pos_bias = None
                #)

    def forward(
        self, query: torch.Tensor, key_value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, nq, _ = query.shape
        nkv = key_value.shape[1]
        q = self.q_proj(query).view(b, nq, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(b, nkv, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(b, nkv, self.heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.pos_bias is not None:
            attn = attn + self.pos_bias[None, None]  # (1,1,nq,K) broadcast over (B,h,nq,K)
        attn = attn.softmax(dim=-1)  # (B, heads, nq, nkv)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, nq, -1)
        return self.out_proj(out), attn


class SPADENorm(nn.Module):
    """Per-position scale/shift modulation from a style map (SPADE)."""

    def __init__(self, channels: int, style_dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.gamma = nn.Conv2d(style_dim, channels, 3, padding=1)
        self.beta = nn.Conv2d(style_dim, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return x * self.gamma(style) + self.beta(style)


class SPADEResBlock(nn.Module):
    def __init__(self, channels: int, style_dim: int) -> None:
        super().__init__()
        self.norm1 = SPADENorm(channels, style_dim)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = SPADENorm(channels, style_dim)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x, style)))
        h = self.conv2(F.silu(self.norm2(h, style)))
        return x + h


class SPADECNNHead(nn.Module):
    """``(B, D, h, w)`` content + style map → ``(B, 1, H, W)``.

    SPADE is applied at the base (pre-upsample) resolution, where the skeleton
    and terminal placement are decided; the upsample blocks then render it.
    """

    def __init__(
        self, in_channels, style_dim, base_channels, num_res_blocks, downsample
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.blocks = nn.Sequential(
            *[SPADEResBlock(base_channels, style_dim) for _ in range(num_res_blocks)]
        )
        num_ups = int(math.log2(downsample))
        self.ups = nn.ModuleList(
            [_UpsampleBlock(base_channels, base_channels) for _ in range(num_ups)]
        )
        self.norm = _group_norm(base_channels)
        self.out = nn.Conv2d(base_channels, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, style_map: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        for blk in self.blocks:
            x = blk(x, style_map)
        for up in self.ups:
            x = up(x)
        return torch.sigmoid(self.out(F.silu(self.norm(x))))


class StyleExtractionModelV3(SaveLoadModel):
    """Font-level style autoencoder: Perceiver tokens + content-conditioned SPADE."""

    def __init__(self, config: StyleExtractionV3Config) -> None:
        super().__init__()
        self.config = config
        d = config.glyph_encoder_feature_dim
        self.grid_size = config.image_size // config.glyph_encoder_downsample
        self.grid_n = self.grid_size ** 2

        self.glyph_encoder = PerGlyphEncoder(
            config.glyph_encoder_base_channels, d, config.glyph_encoder_downsample
        )
        self.codepoint_embedding = nn.Embedding(config.num_codepoints, d)

        # Decoder content-query positional embedding (fine, output-grid position).
        self.query_pos_embed = nn.Parameter(torch.randn(self.grid_n, d) * 0.02)

        # Reference-token tags (codepoint identity + coarse structural region).
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

        # Decoder: content-query self-attention, then cross-attention to tokens
        # to produce the per-position style map, then a SPADE CNN head.
        self.self_attn = nn.ModuleList(
            [
                SelfAttnBlock(d, config.decoder_num_heads, config.decoder_dropout)
                for _ in range(config.decoder_self_attn_layers)
            ]
        )
        self.cross_attn = CrossAttentionWithWeights(
            d,
            config.decoder_num_heads,
            config.decoder_dropout,
            nq=self.grid_n,
            num_tokens=config.num_style_tokens,
            grid_size=self.grid_size,
            use_pos_bias=config.cross_attn_pos_bias,
            pos_bias_trainable=config.cross_attn_pos_bias_trainable,
        )
        self.cnn_head = SPADECNNHead(
            d,
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
        """Encode a font's glyph set into ``(B, K, D)`` style tokens."""
        b, g, c, h, w = style_images.shape
        flat = style_images.flatten(0, 1)  # (B*G, 1, H, W)
        feats = self.glyph_encoder(flat)  # (B*G, D, gh, gw)
        d, gh, gw = feats.shape[1:]
        n = gh * gw
        feats = feats.reshape(b, g, d, gh, gw)  # (B, G, D, gh, gw)
        tokens = feats.permute(0, 1, 3, 4, 2).reshape(b, g, n, d)  # (B, G, n, D)

        tokens = tokens + self.coarse_region_embed[self.coarse_idx][None, None]

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
        return_attention: bool = False,
    ) -> torch.Tensor:
        """Render a target codepoint in the given style tokens.

        When ``return_attention`` is True, returns ``(image, attn)`` where
        ``attn`` is the ``(B, heads, nq, K)`` cross-attention weights (for the
        position-dependence regularizer).
        """
        b = target_codepoint_idx.shape[0]
        d = self.config.glyph_encoder_feature_dim
        codepoint_emb = self.codepoint_embedding(target_codepoint_idx)  # (B, D)
        queries = codepoint_emb[:, None, :] + self.query_pos_embed[None]  # (B, n, D)

        for blk in self.self_attn:
            queries = blk(queries)

        style, attn = self.cross_attn(queries, style_tokens)  # (B, n, D), (B, h, n, K)

        content = queries.transpose(1, 2).reshape(b, d, self.grid_size, self.grid_size)
        style_map = style.transpose(1, 2).reshape(b, d, self.grid_size, self.grid_size)
        image = self.cnn_head(content, style_map)

        if return_attention:
            return image, attn
        return image

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
