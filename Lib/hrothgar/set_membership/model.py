"""Instance-membership model: a per-glyph style encoder plus instance prototypes.

A glyph is encoded (uniform scale-and-pad image) into an L2-normalized
embedding.  An *instance* is represented by the mean of its evidence glyphs'
embeddings.  Membership of a query glyph is the cosine between its embedding
and the instance's prototype — single glyphs are therefore scored in the same
space the model was trained in.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from hrothgar.set_membership.config import SetMembershipConfig
from hrothgar.glyph_encoder import GlyphEncoder
from hrothgar.utils import SaveLoadModel


class SetMembershipModel(SaveLoadModel):
    """Encode glyphs and form instance prototypes in a shared embedding space."""

    def __init__(self, config: SetMembershipConfig):
        super().__init__()
        self.config = config

        self.encoder = GlyphEncoder(
            base_channels=config.encoder_base_channels,
            out_dim=config.encoder_feature_dim,
            downsample=config.encoder_downsample,
        )
        self.image_head = nn.Linear(config.encoder_feature_dim, config.embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized ``(B, embedding_dim)`` glyph embeddings."""
        features = self.encoder(images)  # (B, F, h, w)
        z = self.image_head(F.adaptive_avg_pool2d(features, 1).flatten(1))
        return F.normalize(z, p=2, dim=-1)

    @staticmethod
    def prototype(embeddings: torch.Tensor) -> torch.Tensor:
        """Instance reference: mean of its glyph embeddings, L2-normalized."""
        return F.normalize(embeddings.mean(dim=0), p=2, dim=-1)
