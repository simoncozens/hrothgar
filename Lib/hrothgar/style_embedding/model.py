"""Font-level style embedder from a fixed set of glyph renderings.

Encodes each glyph with a shared CNN, attention-pools each glyph's spatial
features into a small number of tokens, then aggregates the per-glyph tokens
into a global style token set plus a 256-d summary vector.  The summary vector
feeds contrastive, tag-prediction, and category heads.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hrothgar.style_embedding.config import FontStyleEmbedderConfig
from hrothgar.utils import SaveLoadModel

ALL_CATEGORIES = ["Serif", "Sans", "Handwriting", "Script", "Monospace", "Display"]


class GlyphEncoder(nn.Module):
    """Small shared CNN: (1, H, W) -> (out_dim, H/ds, W/ds)."""

    def __init__(
        self,
        base_channels: int = 32,
        out_dim: int = 256,
        downsample: int = 4,
    ) -> None:
        super().__init__()
        if downsample not in (2, 4, 8):
            raise ValueError(f"downsample must be 2, 4, or 8, got {downsample}")

        layers: list[nn.Module] = []
        c = 1
        # Initial block (no downsampling).
        layers.extend(self._block(c, base_channels, stride=1))
        c = base_channels

        # Strided blocks until the requested spatial reduction is reached.
        while downsample >= 2:
            layers.extend(self._block(c, c * 2, stride=2))
            c *= 2
            downsample //= 2

        # Final projection to the target channel count.
        layers.append(nn.Conv2d(c, out_dim, 3, padding=1))
        self.net = nn.Sequential(*layers)

    @staticmethod
    def _block(in_c: int, out_c: int, stride: int) -> list[nn.Module]:
        groups = min(8, out_c)
        return [
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1),
            nn.GroupNorm(groups, out_c),
            nn.SiLU(),
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionPool(nn.Module):
    """Pool a spatial feature map into ``num_queries`` tokens via learned queries."""

    def __init__(self, dim: int, num_queries: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * (dim ** -0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, h, w) -> (N, Q, C)
        n, c, h, w = x.shape
        k = x.flatten(2).transpose(1, 2)  # (N, h*w, C)
        q = self.queries.unsqueeze(0).expand(n, -1, -1)  # (N, Q, C)
        attn = torch.bmm(q, k.transpose(1, 2)) * (c ** -0.5)  # (N, Q, h*w)
        attn = F.softmax(attn, dim=-1)
        return torch.bmm(attn, k)  # (N, Q, C)


class PerceiverBlock(nn.Module):
    """Cross-attention from learned latents to a token sequence (1 block + FFN)."""

    def __init__(self, dim: int, num_latents: int, num_heads: int = 8) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * (dim ** -0.5))
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (B, S, C) -> (B, K, C)
        b = x.shape[0]
        lat = self.latents.unsqueeze(0).expand(b, -1, -1)  # (B, K, C)
        h, _ = self.cross_attn(
            lat, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        lat = self.norm1(lat + h)
        lat = self.norm2(lat + self.ffn(lat))
        return lat


class ReconstructionHead(nn.Module):
    """Decode a glyph from a style summary + glyph-slot identity.

    Input is a single style vector (``style_dim``) concatenated with a learned
    embedding for the glyph slot.  A small transposed-conv decoder maps that
    to a ``(1, glyph_size, glyph_size)`` greyscale glyph in [0, 1].
    """

    def __init__(
        self,
        style_dim: int,
        num_slots: int,
        codepoint_dim: int = 64,
        glyph_size: int = 64,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.glyph_size = glyph_size
        self.hidden = hidden
        self.start = glyph_size // 16
        self.codepoint_embedding = nn.Embedding(num_slots, codepoint_dim)

        in_dim = style_dim + codepoint_dim
        self.linear = nn.Linear(in_dim, hidden * self.start * self.start)

        layers: list[nn.Module] = []
        c = hidden
        cur = self.start
        while cur < glyph_size:
            c_out = max(c // 2, 32)
            layers.append(nn.ConvTranspose2d(c, c_out, 4, 2, 1))
            layers.append(nn.GroupNorm(min(8, c_out), c_out))
            layers.append(nn.SiLU())
            c = c_out
            cur *= 2
        layers.append(nn.Conv2d(c, 1, 3, padding=1))
        layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*layers)

    def forward(
        self,
        style: torch.Tensor,
        slot_indices: torch.Tensor,
    ) -> torch.Tensor:
        # style: (M, style_dim), slot_indices: (M,) long in [0, num_slots)
        cp = self.codepoint_embedding(slot_indices)  # (M, codepoint_dim)
        x = torch.cat([style, cp], dim=-1)  # (M, in_dim)
        x = self.linear(x)  # (M, hidden * start * start)
        x = x.reshape(-1, self.hidden, self.start, self.start)
        return self.decoder(x)  # (M, 1, glyph_size, glyph_size)


class LayoutHead(nn.Module):
    """Predict a glyph's normalized ink bounding box from style + glyph slot.

    Output is ``(x0, y0, x1, y1)`` in [0, 1], i.e. the glyph's left sidebearing,
    top placement, width and height relative to the rendered frame.  This is a
    readout from the style summary — not from metrics — so a low loss proves the
    summary has captured typographic proportions.
    """

    def __init__(
        self,
        style_dim: int,
        num_slots: int,
        codepoint_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.codepoint_embedding = nn.Embedding(num_slots, codepoint_dim)
        in_dim = style_dim + codepoint_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )

    def forward(
        self,
        style: torch.Tensor,
        slot_indices: torch.Tensor,
    ) -> torch.Tensor:
        cp = self.codepoint_embedding(slot_indices)  # (M, codepoint_dim)
        x = torch.cat([style, cp], dim=-1)  # (M, in_dim)
        return self.net(x)  # (M, 4) in [0, 1]


class TagPredictionHead(nn.Module):
    """Predict tag values from the embedding.

    When ``num_classes == 0``: regression head outputs a scalar in [0, 1].
    When ``num_classes >= 2``: classification head outputs logits over bins.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_classes: int = 0,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        out_dim = num_classes if num_classes > 0 else 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.num_classes == 0:
            return out.squeeze(-1)
        return out


class CategoryPredictionHead(nn.Module):
    """Predict the broad font category (6-way classification)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 6)


class FontStyleEmbedder(SaveLoadModel):
    """Font-level style embedder from a fixed set of glyph renderings."""

    def __init__(self, config: FontStyleEmbedderConfig):
        super().__init__()
        self.config = config

        self.encoder = GlyphEncoder(
            base_channels=config.encoder_base_channels,
            out_dim=config.encoder_feature_dim,
            downsample=config.encoder_downsample,
        )
        self.glyph_pool = AttentionPool(
            config.encoder_feature_dim,
            config.per_glyph_tokens,
        )
        self.aggregator = PerceiverBlock(
            config.encoder_feature_dim,
            config.style_latents,
            config.aggregation_heads,
        )

        self.layout_head: Optional[LayoutHead] = None
        if config.use_layout:
            self.layout_head = LayoutHead(
                style_dim=config.encoder_feature_dim,
                num_slots=len(config.input_codepoints),
                codepoint_dim=config.layout_codepoint_dim,
            )

        self.enc_dropout = nn.Dropout(config.encoder_dropout)

        # Projection head for contrastive loss.
        self.projection = nn.Sequential(
            nn.Linear(config.encoder_feature_dim, config.encoder_feature_dim),
            nn.SiLU(),
            nn.Linear(config.encoder_feature_dim, config.projection_dim),
        )

        # Tag prediction heads.
        self.tag_heads: Optional[nn.ModuleDict] = None
        if config.tag_names:
            self.tag_heads = nn.ModuleDict({
                name: TagPredictionHead(
                    config.encoder_feature_dim,
                    config.tag_hidden_dim,
                    num_classes=config.tag_num_classes,
                    dropout=config.tag_dropout,
                )
                for name in config.tag_names
            })

        # Broad category head.
        self.category_head: Optional[CategoryPredictionHead] = None
        if config.use_category_head:
            self.category_head = CategoryPredictionHead(
                config.encoder_feature_dim,
                hidden_dim=config.category_hidden_dim,
                dropout=config.category_dropout,
            )

        # Optional text projection — maps frozen text embeddings into the
        # same space as image projections for multi-positive contrastive loss.
        self.text_projection: Optional[nn.Linear] = None
        if config.text_encoder_name:
            self.text_projection = nn.Linear(
                config.text_embedding_dim, config.projection_dim
            )

    def encode(
        self,
        images: torch.Tensor,
        glyph_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode glyph sets into font embeddings.

        Args:
            images: ``(B, G, 1, H, W)`` greyscale glyph renderings in [0, 1].
            glyph_mask: Optional ``(B, G)`` boolean, ``True`` = glyph visible.
                Masked (False) glyphs are ignored by the aggregator.

        Returns:
            ``(B, encoder_feature_dim)`` font embedding.
        """
        b, g, c, h, w = images.shape
        assert c == 1, f"Expected single-channel glyphs, got {c} channels"

        flat = images.flatten(0, 1)  # (B*G, 1, H, W)
        features = self.encoder(flat)  # (B*G, F, h', w')
        tokens = self.glyph_pool(features)  # (B*G, T, F)
        tokens = tokens.reshape(
            b, g * self.config.per_glyph_tokens, self.config.encoder_feature_dim
        )

        key_padding_mask = None
        if glyph_mask is not None:
            # nn.MultiheadAttention key_padding_mask: True = ignore this token.
            pad = ~glyph_mask  # (B, G)
            key_padding_mask = (
                pad.unsqueeze(2)
                .expand(b, g, self.config.per_glyph_tokens)
                .reshape(b, g * self.config.per_glyph_tokens)
            )

        latents = self.aggregator(tokens, key_padding_mask=key_padding_mask)
        summary = latents.mean(dim=1)  # (B, F)
        return self.enc_dropout(summary)

    def predict_layout(
        self,
        style: torch.Tensor,
        slot_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Predict glyph ink bounding boxes from style summaries + slot indices.

        Args:
            style: ``(M, encoder_feature_dim)`` style summaries.
            slot_indices: ``(M,)`` indices into ``config.input_codepoints``.

        Returns:
            ``(M, 4)`` normalized ``(x0, y0, x1, y1)`` in [0, 1].
        """
        assert self.layout_head is not None, (
            "layout_head is disabled; set use_layout=True"
        )
        return self.layout_head(style, slot_indices)

    def project_text(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Project frozen text embeddings into the contrastive projection space."""
        assert self.text_projection is not None, (
            "text_projection not initialised; set text_encoder_name in config"
        )
        if torch.isnan(text_embeddings).any():
            raise RuntimeError("NaN in text_embeddings input to project_text")
        projected = self.text_projection(text_embeddings)
        if torch.isnan(projected).any():
            raise RuntimeError("NaN in text_projection output")
        return F.normalize(projected, p=2, dim=-1, eps=1e-8)

    def forward(
        self,
        images: torch.Tensor,
        glyph_mask: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[dict[str, torch.Tensor]],
        Optional[torch.Tensor],
    ]:
        """Full forward pass.

        Args:
            images: ``(B, G, 1, H, W)`` greyscale glyph renderings.
            glyph_mask: Optional ``(B, G)`` boolean visibility mask.

        Returns:
            ``(embedding, projection, tags, category_logits)``.
        """
        embedding = self.encode(images, glyph_mask=glyph_mask)
        projection = F.normalize(self.projection(embedding), p=2, dim=-1)

        tags: Optional[dict[str, torch.Tensor]] = None
        if self.tag_heads is not None:
            tags = {
                name: head(embedding)
                for name, head in self.tag_heads.items()
            }

        category_logits: Optional[torch.Tensor] = None
        if self.category_head is not None:
            category_logits = self.category_head(embedding)

        return embedding, projection, tags, category_logits
