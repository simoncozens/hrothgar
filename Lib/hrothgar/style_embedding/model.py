"""Font-level style embedder from a fixed set of glyph renderings (Gram pooling).

Encodes each glyph with a shared CNN, then computes a per-glyph Gram matrix
(channel correlations, spatial order destroyed), averages across glyphs, and
projects the flattened Gram into a 256-d summary.  The summary feeds
contrastive, tag-prediction, and category heads.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hrothgar.googlefonts import Font
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
    """Font-level style embedder from glyph renderings via Gram (texture) pooling."""

    def __init__(self, config: FontStyleEmbedderConfig):
        super().__init__()
        self.config = config

        self.encoder = GlyphEncoder(
            base_channels=config.encoder_base_channels,
            out_dim=config.encoder_feature_dim,
            downsample=config.encoder_downsample,
        )

        # 1x1 channel projection before the Gram so the Gram matrix is compact.
        self.gram_proj = nn.Conv2d(
            config.encoder_feature_dim, config.gram_channels, 1, bias=False
        )
        self.gram_embed = nn.Linear(
            config.gram_channels * (config.gram_channels + 1) // 2,
            config.encoder_feature_dim,
        )
        self.register_buffer(
            "_gram_idx",
            torch.triu_indices(config.gram_channels, config.gram_channels),
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
        """Encode glyph sets into the summary style embedding.

        Args:
            images: ``(B, G, 1, H, W)`` greyscale glyph renderings in [0, 1].
            glyph_mask: Optional ``(B, G)`` boolean, ``True`` = glyph visible.

        Returns:
            ``(B, encoder_feature_dim)`` Gram-based summary vector.
        """
        b, g, c, h, w = images.shape
        assert c == 1, f"Expected single-channel glyphs, got {c} channels"

        flat = images.flatten(0, 1)  # (B*G, 1, H, W)
        features = self.encoder(flat)  # (B*G, F, h', w')

        k = self.config.gram_channels
        proj = self.gram_proj(features)  # (B*G, K, h', w')
        n = proj.shape[2] * proj.shape[3]
        proj = proj.reshape(b, g, k, n)  # (B, G, K, N)
        flat_proj = proj.reshape(b * g, k, n)
        gram = torch.bmm(flat_proj, flat_proj.transpose(1, 2)) / n  # (B*G, K, K)
        gram = gram.reshape(b, g, k, k)
        if glyph_mask is not None:
            mask = glyph_mask.to(gram.dtype).reshape(b, g, 1, 1)
            gram = (gram * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            gram = gram.mean(dim=1)  # (B, K, K)

        tri = gram[:, self._gram_idx[0], self._gram_idx[1]]  # (B, K(K+1)/2)
        summary = self.gram_embed(tri)  # (B, encoder_feature_dim)
        return self.enc_dropout(summary)

    def compute_embedding(
        self,
        font: Font,
        device: Optional[torch.device] = None,
        axis_position: Optional[list[float]] = None,
    ) -> torch.Tensor:
        """Render a font's full input set and return its style embedding.

        This is the single source of truth for turning a font into the vector
        consumed by downstream tasks (similarity search, tag prediction, and
        the AR generator).  It renders the same glyph set and applies the same
        rendering logic used by the training dataset, so the two can never
        drift apart.
        """
        from hrothgar.style_embedding.render_utils import render_input_set

        images = render_input_set(
            font,
            self.config.input_codepoints,
            self.config.glyph_size,
            axis_position=axis_position,
        ).unsqueeze(0)  # (1, G, 1, H, W)

        # ``Font.render`` silently returns an all-white image on failure, so
        # surface blank glyphs here so callers can apply their unrenderable-font
        # fallback instead of embedding blank space.
        blank = images.amin(dim=(-2, -1)) > 0.995  # (1, G, 1)
        if bool(blank.any()):
            cp_idx = int(blank.squeeze(0).squeeze(1).nonzero()[0].item())
            cp = self.config.input_codepoints[cp_idx]
            raise ValueError(f"blank glyph {chr(cp)!r} (U+{cp:04X}) for {font.path}")

        if device is not None:
            images = images.to(device)

        with torch.no_grad():
            return self.encode(images).squeeze(0).cpu()

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

        Returns ``(embedding, projection, tags, category_logits)``.
        """
        embedding = self.encode(images, glyph_mask)
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
