"""Style autoencoder v5: global style bottleneck + slot-attention style atoms.

v4 pooled the evidence into a *spatial region grid*, which could place fine
detail but encoded global style (e.g. terminal roundness) as per-region
patterns — tied to specific glyph geometry, so it could not transfer across
glyphs and reverted to per-glyph memoization.

v5 fixes that by moving the bottleneck *upstream* of any spatial structure:

    evidence → z_global  (one vector per font; the transferable bottleneck)
             → slot attention → K style atoms  (a learned continuous basis)
             → content-conditioned decoder → glyph

The atoms are a *function of z only* (they attend to a learned continuous
basis, never to the per-glyph evidence), so per-glyph memoization is
structurally impossible.  The decoder is content-conditioned: content queries
(codepoint + position) cross-attend to the atoms, so "instroke at glyph end" and
"terminal at stem top" are resolved by the *content* asking the *atoms* for the
right style at each place.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hrothgar.style_extraction.config import StyleExtractionV5Config
from hrothgar.style_extraction.model_v2 import MultiHeadAttention, PerGlyphEncoder
from hrothgar.style_extraction.model_v3 import (
    CrossAttentionWithWeights,
    SPADECNNHead,
    SelfAttnBlock,
)
from hrothgar.utils import SaveLoadModel


class AttentionPool(nn.Module):
    """Pool evidence tokens into a rich global vector via learned queries.

    A small set of learned queries cross-attend to the evidence tokens and are
    then averaged into a single ``z``.  Unlike mean-pooling (which dilutes a
    fine-detail signal like terminal roundness across ~N spatial tokens), the
    queries can attend to the salient regions and preserve the signal.
    """

    def __init__(self, dim: int, num_queries: int, heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.attn = MultiHeadAttention(dim, heads, dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b = tokens.shape[0]
        queries = self.queries[None].expand(b, -1, -1)  # (B, Q, D)
        pooled = self.attn(queries, tokens, tokens)  # (B, Q, D)
        return pooled.mean(dim=1)  # (B, D)


class SlotAttention(nn.Module):
    """Expand a global style vector ``z`` into ``K`` style atoms.

    ``K`` slot vectors, seeded by ``z``, iteratively attend to a *learned
    continuous basis* with the softmax normalised over slots (competition), so
    the slots specialise into distinct style aspects (terminal / stroke /
    instroke / serif …).  The basis is continuous and shared, so unseen styles
    are reached by interpolation — not by requiring a new codebook entry.
    """

    def __init__(
        self,
        dim: int,
        num_slots: int,
        basis_size: int,
        num_iters: int = 3,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.num_iters = num_iters
        self.dim = dim

        self.slots_init = nn.Parameter(torch.randn(num_slots, dim) * 0.02)
        self.basis = nn.Parameter(torch.randn(basis_size, dim) * 0.02)
        self.z_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        slots = self.z_proj(z).unsqueeze(1) + self.slots_init[None]  # (B, K, D)
        basis = self.basis[None].expand(b, -1, -1)  # (B, M, D)

        for _ in range(self.num_iters):
            q = self.q_proj(slots)  # (B, K, D)
            k = self.k_proj(basis)  # (B, M, D)
            v = self.v_proj(basis)  # (B, M, D)
            attn = torch.matmul(k, q.transpose(-2, -1)) / math.sqrt(self.dim)  # (B, M, K)
            attn = F.softmax(attn, dim=-1)  # normalise over slots (competition)
            updates = torch.matmul(attn.transpose(-2, -1), v)  # (B, K, D)

            slots = self.gru(
                slots.reshape(b * self.num_slots, self.dim),
                updates.reshape(b * self.num_slots, self.dim),
            )
            slots = slots.reshape(b, self.num_slots, self.dim)
            slots = slots + self.mlp(self.norm(slots))

        return slots  # (B, K, D)


class StyleExtractionModelV5(SaveLoadModel):
    """Font-level style autoencoder: global bottleneck → style atoms → SPADE."""

    def __init__(self, config: StyleExtractionV5Config) -> None:
        super().__init__()
        self.config = config
        d = config.glyph_encoder_feature_dim
        self.grid_size = config.image_size // config.glyph_encoder_downsample
        self.grid_n = self.grid_size ** 2

        # ---- Encoder (evidence -> global z) ----
        self.glyph_encoder = PerGlyphEncoder(
            config.glyph_encoder_base_channels, d, config.glyph_encoder_downsample
        )
        self.codepoint_embedding = nn.Embedding(config.num_codepoints, d)
        self.z_mlp = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)
        )
        self.attention_pool = AttentionPool(
            d, config.global_pool_queries, config.decoder_num_heads
        )

        # ---- Global latent -> style atoms ----
        self.slot_attention = SlotAttention(
            d, config.num_style_slots, config.slot_basis_size, config.slot_num_iters
        )

        # ---- Content-conditioned decoder ----
        self.query_pos_embed = nn.Parameter(torch.randn(self.grid_n, d) * 0.02)
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
            num_tokens=config.num_style_slots,
            grid_size=self.grid_size,
            use_pos_bias=config.cross_attn_pos_bias,
            pos_bias_trainable=config.cross_attn_pos_bias_trainable,
            pos_bias_scale=config.cross_attn_pos_bias_scale,
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
        """Encode a font's evidence into ``(B, K, D)`` style atoms.

        Evidence is attention-pooled into a single global vector ``z`` (the
        bottleneck), which slot attention then expands into ``K`` atoms.
        """
        b, g, c, h, w = style_images.shape
        flat = style_images.flatten(0, 1)  # (B*G, 1, H, W)
        feats = self.glyph_encoder(flat)  # (B*G, D, gh, gw)
        d, gh, gw = feats.shape[1:]
        n = gh * gw
        feats = feats.reshape(b, g, d, gh, gw)  # (B, G, D, gh, gw)
        tokens = feats.permute(0, 1, 3, 4, 2).reshape(b, g, n, d)  # (B, G, n, D)

        if style_mask is not None:
            mask = style_mask.to(tokens.dtype).reshape(b, g, 1, 1)
            tokens = tokens * mask

        z = self.attention_pool(tokens.reshape(b, g * n, d))  # (B, D)
        z = self.z_mlp(z)  # (B, D)
        return self.slot_attention(z)  # (B, K, D)

    def decode(
        self,
        target_codepoint_idx: torch.Tensor,
        atoms: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """Render a glyph from content queries conditioned on the style atoms."""
        b = target_codepoint_idx.shape[0]
        d = self.config.glyph_encoder_feature_dim

        codepoint_emb = self.codepoint_embedding(target_codepoint_idx)  # (B, D)
        queries = codepoint_emb[:, None, :] + self.query_pos_embed[None]  # (B, n, D)

        for blk in self.self_attn:
            queries = blk(queries)

        style, attn = self.cross_attn(queries, atoms)  # (B, n, D), (B, h, n, K)
        content = queries.transpose(1, 2).reshape(b, d, self.grid_size, self.grid_size)
        style_map = style.transpose(1, 2).reshape(b, d, self.grid_size, self.grid_size)
        glyph = self.cnn_head(content, style_map)  # (B, 1, H, W)

        if return_attention:
            return glyph, attn
        return glyph

    def forward(
        self,
        style_images: torch.Tensor,
        target_codepoint_idx: torch.Tensor,
        style_codepoint_idx: Optional[torch.Tensor] = None,
        style_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        atoms = self.encode_style(
            style_images,
            style_codepoint_idx=style_codepoint_idx,
            style_mask=style_mask,
        )
        return self.decode(target_codepoint_idx, atoms)
