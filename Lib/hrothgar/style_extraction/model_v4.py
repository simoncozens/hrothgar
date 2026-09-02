"""Style autoencoder v4: two-stage coarse-to-fine decoder with a spatial style grid.

The Perceiver in earlier versions pooled the evidence glyphs into a global,
permutation-invariant token set.  That turned every fine-detail axis (terminal
roundness, counter shape) into a *global shift* — the tokens carried the axis as
a uniform offset, so the per-position cross-attention read the same global style
everywhere and the decoder could never place the detail at a specific location.

v4 fixes that by replacing the Perceiver with a *spatial region grid*:

* the evidence features are pooled into ``coarse_grid_size × coarse_grid_size``
  region descriptors (one per coarse spatial location), and
* those descriptors are applied *explicitly, per region*, through a ``RegionAdaIN``
  layer — within each region the content is instance-normalised and rescaled by
  that region's own descriptor, so region *r*'s style is applied to region *r*
  and cannot be collapsed or ignored.

Stage 1 (coarse) still produces a coarse glyph (skeleton + high-level style) from
the content query + a global style summary; stage 2 (fine) encodes that coarse
glyph, restyles it per-region, and renders it.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hrothgar.style_extraction.config import StyleExtractionV4Config
from hrothgar.style_extraction.model_v2 import (
    CNNHead,
    PerGlyphEncoder,
)
from hrothgar.style_extraction.model_v3 import (
    SelfAttnBlock,
    SPADECNNHead,
)
from hrothgar.utils import SaveLoadModel


class RegionAdaIN(nn.Module):
    """Explicitly apply per-region style descriptors to content features.

    Content ``(B, D, grid, grid)`` is segmented into ``coarse × coarse`` regions.
    Within each region the features are instance-normalised (so the content's own
    statistics cannot survive) and then scaled/shifted by an affine map of that
    region's own descriptor.  Region *r*'s style is applied to region *r* only;
    there is no learnable global pooling that could collapse it away.
    """

    def __init__(self, dim: int, coarse_grid_size: int) -> None:
        super().__init__()
        self.dim = dim
        self.coarse = coarse_grid_size
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
        b, d, grid, _ = x.shape
        c = self.coarse
        rh = grid // c
        rw = grid // c
        # (B, D, grid, grid) -> (B, D, c, rh, c, rw) -> (B, c, c, D, rh, rw)
        xr = x.reshape(b, d, c, rh, c, rw).permute(0, 2, 4, 1, 3, 5)
        xr = xr.reshape(b, c * c, d, rh * rw)  # (B, R, D, rh*rw)
        mean = xr.mean(dim=-1, keepdim=True)
        var = xr.var(dim=-1, keepdim=True, unbiased=False)
        xr = (xr - mean) / torch.sqrt(var + 1e-5)
        g = self.gamma(region).unsqueeze(-1)  # (B, R, D, 1)
        beta = self.beta(region).unsqueeze(-1)  # (B, R, D, 1)
        xr = xr * g + beta
        xr = xr.reshape(b, c, c, d, rh, rw).permute(0, 3, 1, 4, 2, 5)
        return xr.reshape(b, d, grid, grid)


class StyleExtractionModelV4(SaveLoadModel):
    """Font-level style autoencoder with a coarse-to-fine two-stage decoder."""

    def __init__(self, config: StyleExtractionV4Config) -> None:
        super().__init__()
        self.config = config
        d = config.glyph_encoder_feature_dim
        self.grid_size = config.image_size // config.glyph_encoder_downsample
        self.grid_n = self.grid_size ** 2
        self.coarse = config.coarse_grid_size
        assert self.grid_size % self.coarse == 0

        # ---- Encoder ----
        self.glyph_encoder = PerGlyphEncoder(
            config.glyph_encoder_base_channels, d, config.glyph_encoder_downsample
        )
        self.codepoint_embedding = nn.Embedding(config.num_codepoints, d)

        self.query_pos_embed = nn.Parameter(torch.randn(self.grid_n, d) * 0.02)

        # Coarse-region tags for the evidence tokens (positional identity).
        self.coarse_region_embed = nn.Parameter(
            torch.randn(self.coarse * self.coarse, d) * 0.02
        )
        ys = torch.arange(self.grid_size) // (self.grid_size // self.coarse)
        xs = torch.arange(self.grid_size) // (self.grid_size // self.coarse)
        self.register_buffer(
            "coarse_idx", (ys[:, None] * self.coarse + xs[None, :]).reshape(-1)
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

        # ---- Stage 2: fine decoder (per-region style injection) ----
        self.fine_encoder = PerGlyphEncoder(
            config.glyph_encoder_base_channels, d, config.glyph_encoder_downsample
        )
        self.region_adain = RegionAdaIN(d, self.coarse)
        self.fine_head = CNNHead(
            d,
            config.decoder_base_channels,
            config.decoder_num_res_blocks,
            config.glyph_encoder_downsample,
            activation=False,
        )

    def encode_style(
        self,
        style_images: torch.Tensor,
        style_codepoint_idx: Optional[torch.Tensor] = None,
        style_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a font's glyph set into ``(B, R, D)`` spatial region descriptors."""
        b, g, c, h, w = style_images.shape
        flat = style_images.flatten(0, 1)  # (B*G, 1, H, W)
        feats = self.glyph_encoder(flat)  # (B*G, D, gh, gw)
        d, gh, gw = feats.shape[1:]
        feats = feats.reshape(b, g, d, gh, gw)  # (B, G, D, gh, gw)
        tokens = feats.permute(0, 1, 3, 4, 2)  # (B, G, gh, gw, D)

        region_tag = self.coarse_region_embed[self.coarse_idx].reshape(gh, gw, d)
        tokens = tokens + region_tag[None, None]  # broadcast over (B, G)

        if style_codepoint_idx is not None:
            cp_emb = self.codepoint_embedding(style_codepoint_idx)  # (B, G, D)
            tokens = tokens + cp_emb[:, :, None, None, :]

        if style_mask is not None:
            mask = style_mask.to(tokens.dtype).reshape(b, g, 1, 1, 1)
            tokens = tokens * mask

        rh = gh // self.coarse
        rw = gw // self.coarse
        # Group spatial positions into coarse regions, then mean over glyphs and
        # within-region positions -> (B, coarse, coarse, D) -> (B, R, D).
        tokens = tokens.reshape(b, g, self.coarse, rh, self.coarse, rw, d)
        region = tokens.mean(dim=(1, 3, 5))  # (B, coarse, coarse, D)
        return region.reshape(b, self.coarse * self.coarse, d)

    def decode_coarse(
        self, target_codepoint_idx: torch.Tensor, region_feats: torch.Tensor
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
        # must not flow back into the region descriptors, or it pushes them toward
        # their mean (a uniform gradient) and erases the per-region structure the
        # fine stage relies on.
        style_summary = region_feats.mean(dim=1).detach()  # (B, D)
        style_map = style_summary[:, :, None, None].expand(
            -1, -1, self.grid_size, self.grid_size
        )
        return self.coarse_head(content, style_map)  # (B, 1, H, W)

    def _fine_stage(
        self, coarse: torch.Tensor, region_feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the fine stage; return ``(fine, styled)`` (styled = post-AdaIN).

        The fine head emits an unconstrained *residual* which is added to the
        coarse skeleton, so the fine stage can only refine the coarse output — it
        cannot replace it — and the coarse stage is forced to carry the skeleton.
        """
        coarse_feats = self.fine_encoder(coarse)  # (B, d, grid, grid)
        styled = self.region_adain(coarse_feats, region_feats)  # (B, d, grid, grid)
        residual = self.fine_head(styled)  # (B, 1, H, W) unconstrained
        fine = (coarse + residual).clamp(0.0, 1.0)
        return fine, styled

    def decode_fine(
        self,
        target_codepoint_idx: torch.Tensor,
        coarse: torch.Tensor,
        region_feats: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """Restyle the coarse glyph per region, then render it.

        ``return_attention`` is kept for API compatibility with the cross-attention
        decoder; there is no attention here, so it returns ``(fine, None)``.
        """
        fine, _styled = self._fine_stage(coarse, region_feats)
        if return_attention:
            return fine, None
        return fine

    def decode_with_intermediates(
        self,
        target_codepoint_idx: torch.Tensor,
        region_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode and return ``(coarse, region_map, fine, attn)``.

        ``region_map`` is the spatial region grid upsampled to the full grid
        (region-constant within each region); ``attn`` is ``None`` (no
        cross-attention).  Used by the axis probe.
        """
        b = target_codepoint_idx.shape[0]
        d = self.config.glyph_encoder_feature_dim
        coarse = self.decode_coarse(target_codepoint_idx, region_feats)
        fine, _styled = self._fine_stage(coarse, region_feats)
        region_grid = region_feats.reshape(b, self.coarse, self.coarse, d).permute(
            0, 3, 1, 2
        )
        region_map = F.interpolate(
            region_grid, size=(self.grid_size, self.grid_size), mode="nearest"
        )
        return coarse, region_map, fine, None

    def decode(
        self,
        target_codepoint_idx: torch.Tensor,
        region_feats: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        coarse = self.decode_coarse(target_codepoint_idx, region_feats)
        return self.decode_fine(
            target_codepoint_idx, coarse, region_feats, return_attention=return_attention
        )

    def forward(
        self,
        style_images: torch.Tensor,
        target_codepoint_idx: torch.Tensor,
        style_codepoint_idx: Optional[torch.Tensor] = None,
        style_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        region_feats = self.encode_style(
            style_images,
            style_codepoint_idx=style_codepoint_idx,
            style_mask=style_mask,
        )
        return self.decode(target_codepoint_idx, region_feats)
