"""Font-level style embedder from phrase renderings.

Renders a short phrase in a font, encodes the image with a CNN,
global-average-pools to a single embedding, and projects through
contrastive + tag-prediction + category heads.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hrothgar.upstream.style_encoder import StyleEncoder
from hrothgar.style_embedding.config import FontStyleEmbedderConfig
from hrothgar.utils import SaveLoadModel

ALL_CATEGORIES = ["Serif", "Sans", "Handwriting", "Script", "Monospace", "Display"]


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
    """Font-level style embedder from phrase renderings.

    Encodes a single phrase rendering into a font embedding suitable for
    contrastive retrieval, tag prediction, and generator conditioning.
    """

    def __init__(self, config: FontStyleEmbedderConfig):
        super().__init__()
        self.config = config

        # CNN encoder operating on the square phrase image.
        # 8× downsampling: 512 → 64×64 feature map → GAP → 1D vector.
        downsample_ratio = 8
        self.encoder = StyleEncoder(
            C_in=3,
            C=config.encoder_base_channels,
            C_out=config.encoder_feature_dim,
            norm="in",
            activ="relu",
            pad_type="reflect",
            sigmoid=False,
            scale_var=True,
            downsample_ratio=downsample_ratio,
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

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode phrase images into font embeddings.

        Args:
            images: ``(B, 3, H, W)`` — square phrase renderings.

        Returns:
            ``(B, encoder_feature_dim)`` font embedding.
        """
        B, C, H, W = images.shape
        assert C == 3, f"Expected RGB images, got {C} channels"

        # Normalize to [-1, 1] (matches StyleEncoder convention).
        x = (images.float() - 0.5) / 0.5

        features = self.encoder(x)            # (B, encoder_feature_dim, H', W')
        features = self.enc_dropout(features)
        embedding = features.mean(dim=[-2, -1])  # (B, encoder_feature_dim)
        return embedding

    def project_text(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Project frozen text embeddings into the contrastive projection space.

        Args:
            text_embeddings: ``(B, text_embedding_dim)`` — raw frozen embeddings.

        Returns:
            ``(B, projection_dim)`` L2-normalized projections.
        """
        assert self.text_projection is not None, (
            "text_projection not initialised; set text_encoder_name in config"
        )
        projected = self.text_projection(text_embeddings)
        # Guard against zero vectors (fonts with no description).
        return F.normalize(projected, p=2, dim=-1, eps=1e-8)

    def forward(
        self, images: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[dict[str, torch.Tensor]],
        Optional[torch.Tensor],
    ]:
        """Full forward pass.

        Args:
            images: ``(B, 3, H, W)`` phrase renderings (two views stacked).

        Returns:
            ``(embedding, projection, tags, category_logits)``.
        """
        embedding = self.encode(images)
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
