"""Style autoencoder v4: two-stage coarse-to-fine decoder.

v2/v3's cross-attention query (codepoint + learned positional embedding)
collapsed to a global style vector because that query is a weak, learnable
spatial signal.  v4 fixes this by giving the *fine* decoder a real spatial
condition: a coarse glyph produced by a first, L1-driven stage.

Stage 1 produces a coarse glyph (skeleton + high-level style: weight/slant/width)
from the content query + a global style summary.  Stage 2 encodes that coarse
glyph back into spatial features and uses those features — not a learned
positional embedding — as the query for the style cross-attention, then refines
via SPADE.  Position-dependence is therefore grounded in the coarse output,
which stage 1's L1 loss forces to be an actual, position-dependent glyph.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from hrothgar.style_extraction.config import StyleExtractionV4Config
from hrothgar.style_extraction.model_v2 import (
    PerceiverBlock,
    PerGlyphEncoder,
)
from hrothgar.style_extraction.model_v3 import (
    CrossAttentionWithWeights,
    SelfAttnBlock,
    SPADECNNHead,
)
from hrothgar.utils import SaveLoadModel


class StyleExtractionModelV4(SaveLoadModel):
    """Font-level style autoencoder with a coarse-to-fine two-stage decoder."""

    def __init__(self, config: StyleExtractionV4Config) -> None:
        super().__init__()
        self.config = config
        d = config.glyph_encoder_feature_dim
        self.grid_size = config.image_size // config.glyph_encoder_downsample
        self.grid_n = self.grid_size ** 2

        # ---- Encoder / Perceiver (unchanged from v2/v3) ----
        self.glyph_encoder = PerGlyphEncoder(
            config.glyph_encoder_base_channels, d, config.glyph_encoder_downsample
        )
        self.codepoint_embedding = nn.Embedding(config.num_codepoints, d)

        self.query_pos_embed = nn.Parameter(torch.randn(self.grid_n, d) * 0.02)

        coarse = config.coarse_grid_size
        assert self.grid_size % coarse == 0
        self.coarse_region_embed = nn.Parameter(torch.randn(coarse * coarse, d) * 0.02)
        ys = torch.arange(self.grid_size) // (self.grid_size // coarse)
        xs = torch.arange(self.grid_size) // (self.grid_size // coarse)
        self.register_buffer(
            "coarse_idx", (ys[:, None] * coarse + xs[None, :]).reshape(-1)
        )

        self.style_latents = nn.Parameter(
            torch.randn(config.num_style_tokens, d) * 0.02
        )
        self.perceiver = nn.ModuleList(
            [
                PerceiverBlock(d, config.perceiver_num_heads, config.decoder_dropout)
                for _ in range(config.perceiver_num_layers)
            ]
        )

        # ---- Stage 1: coarse decoder (content query + global style) ----
        self.coarse_self_attn = nn.ModuleList(
            [
                SelfAttnBlock(d, config.decoder_num_heads, config.decoder_dropout)
                for _ in range(config.decoder_self_attn_layers)
            ]
        )
        self.coarse_head = SPADECNNHead(
            d,
            d,
            config.decoder_base_channels,
            config.decoder_num_res_blocks,
            config.glyph_encoder_downsample,
        )

        # ---- Stage 2: fine decoder (spatially-grounded style query) ----
        self.fine_encoder = PerGlyphEncoder(
            config.glyph_encoder_base_channels, d, config.glyph_encoder_downsample
        )
        # Grounded query + fixed sinusoidal positional bias.  The grounding alone
        # produced position-dependence only weakly (q_var rose linearly), which is
        # not enough to reproduce fine axis detail.  A fixed, non-collapsible bias
        # forces position-dependence so we can test whether that is the bottleneck.
        # Scale (config.cross_attn_pos_bias_scale, default 0.5) is set so the bias
        # clearly raises q_var (~3e-4) without underusing the token set.
        self.fine_cross_attn = CrossAttentionWithWeights(
            d,
            config.decoder_num_heads,
            config.decoder_dropout,
            nq=self.grid_n,
            num_tokens=config.num_style_tokens,
            grid_size=self.grid_size,
            use_pos_bias=config.cross_attn_pos_bias,
            pos_bias_trainable=config.cross_attn_pos_bias_trainable,
            pos_bias_scale=config.cross_attn_pos_bias_scale,
        )
        self.fine_head = SPADECNNHead(
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

    def decode_coarse(
        self, target_codepoint_idx: torch.Tensor, style_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Produce a coarse glyph (skeleton + global style) → ``(B, 1, H, W)``."""
        b = target_codepoint_idx.shape[0]
        d = self.config.glyph_encoder_feature_dim
        codepoint_emb = self.codepoint_embedding(target_codepoint_idx)  # (B, D)
        queries = codepoint_emb[:, None, :] + self.query_pos_embed[None]  # (B, n, D)

        for blk in self.coarse_self_attn:
            queries = blk(queries)

        content = queries.transpose(1, 2).reshape(b, d, self.grid_size, self.grid_size)
        # Detach the global style summary.  The coarse stage's strong L1 gradient
        # must not flow back into the style tokens, or it pushes all tokens toward
        # their mean (a uniform gradient) and collapses the token set — which then
        # starves the fine cross-attention and drives q_var back to zero.
        style_summary = style_tokens.mean(dim=1).detach()  # (B, D)
        style_map = style_summary[:, :, None, None].expand(
            -1, -1, self.grid_size, self.grid_size
        )
        return self.coarse_head(content, style_map)  # (B, 1, H, W)

    def decode_fine(
        self,
        target_codepoint_idx: torch.Tensor,
        coarse: torch.Tensor,
        style_tokens: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """Refine the coarse glyph using a spatially-grounded style query.

        When ``return_attention`` is True, returns ``(fine, attn)`` where ``attn``
        is ``(B, heads, nq, K)``.
        """
        b = target_codepoint_idx.shape[0]
        d = self.config.glyph_encoder_feature_dim
        coarse_feats = self.fine_encoder(coarse)  # (B, d, grid, grid)
        queries = coarse_feats.flatten(2).transpose(1, 2)  # (B, n, d)
        style, attn = self.fine_cross_attn(queries, style_tokens)  # (B, n, d), (B,h,n,K)
        style_map = style.transpose(1, 2).reshape(b, d, self.grid_size, self.grid_size)
        fine = self.fine_head(coarse_feats, style_map)  # (B, 1, H, W)

        if return_attention:
            return fine, attn
        return fine

    def decode(
        self,
        target_codepoint_idx: torch.Tensor,
        style_tokens: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        coarse = self.decode_coarse(target_codepoint_idx, style_tokens)
        return self.decode_fine(
            target_codepoint_idx, coarse, style_tokens, return_attention=return_attention
        )

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
