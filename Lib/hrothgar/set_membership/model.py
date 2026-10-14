"""Instance-membership model: a per-glyph style encoder plus instance prototypes.

A glyph is encoded (crop-to-ink image, optionally + geometry) into an
L2-normalized embedding.  An *instance* is represented by the mean of its
evidence glyphs' embeddings.  Membership of a query glyph is the cosine between
its embedding and the instance's prototype — single glyphs are therefore scored
in the same space the model was trained in.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from hrothgar.set_membership.config import SetMembershipConfig
from hrothgar.utils import SaveLoadModel


class GlyphEncoder(nn.Module):
    """Small shared CNN: ``(1, H, W) -> (out_dim, H/ds, W/ds)``."""

    def __init__(
        self, base_channels: int = 32, out_dim: int = 256, downsample: int = 4
    ):
        super().__init__()
        if downsample not in (2, 4, 8):
            raise ValueError(f"downsample must be 2, 4, or 8, got {downsample}")

        layers: list[nn.Module] = []
        c = 1
        layers.extend(self._block(c, base_channels, stride=1))
        c = base_channels
        while downsample >= 2:
            layers.extend(self._block(c, c * 2, stride=2))
            c *= 2
            downsample //= 2
        layers.append(nn.Conv2d(c, out_dim, 3, padding=1))
        self.net = nn.Sequential(*layers)

    @staticmethod
    def _block(in_c: int, out_c: int, stride: int) -> list[nn.Module]:
        return [
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1),
            nn.GroupNorm(min(8, out_c), out_c),
            nn.SiLU(),
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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

        self.geometry_mlp: nn.Sequential | None = None
        if config.use_geometry:
            self.geometry_mlp = nn.Sequential(
                nn.Linear(5, config.geometry_hidden_dim),
                nn.SiLU(),
                nn.Linear(config.geometry_hidden_dim, config.embedding_dim),
            )

    def forward(
        self, images: torch.Tensor, geometry: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return L2-normalized ``(B, embedding_dim)`` glyph embeddings."""
        features = self.encoder(images)  # (B, F, h, w)
        z = self.image_head(F.adaptive_avg_pool2d(features, 1).flatten(1))
        if self.geometry_mlp is not None and geometry is not None:
            z = z + self.geometry_mlp(geometry)
        return F.normalize(z, p=2, dim=-1)

    @staticmethod
    def prototype(embeddings: torch.Tensor) -> torch.Tensor:
        """Instance reference: mean of its glyph embeddings, L2-normalized."""
        return F.normalize(embeddings.mean(dim=0), p=2, dim=-1)
