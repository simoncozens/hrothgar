"""Content detector: a shared glyph encoder plus a bank of per-codepoint anchors.

Each codepoint the model knows about owns one learned vector in an anchor bank
— the "slot" whose weights answering "is this an e?" pull on.  A glyph is encoded
to an L2-normalized embedding, scored by cosine against every anchor, and turned
into a probability by a scaled logit with a per-anchor bias.  Probabilities are
independent across anchors (one-vs-rest), so a glyph that is none of the known
codepoints can score low everywhere.

This mirrors the *content* half of the diffusion generator's factorized
conditioning (``codepoint_vectors[]``); here the same vector is also the class
prototype the detector scores against.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from hrothgar.content_detector.config import ContentDetectorConfig
from hrothgar.glyph_encoder import GlyphEncoder
from hrothgar.utils import SaveLoadModel


class ContentDetector(SaveLoadModel):
    """Encode glyphs and score them against a bank of codepoint anchors."""

    def __init__(self, config: ContentDetectorConfig):
        super().__init__()
        self.config = config
        self.anchors = list(config.anchors)

        self.encoder = GlyphEncoder(
            base_channels=config.encoder_base_channels,
            out_dim=config.encoder_feature_dim,
            downsample=config.encoder_downsample,
        )
        self.image_head = nn.Linear(config.encoder_feature_dim, config.embedding_dim)
        # Auxiliary regression head: predicts ink height relative to the font's
        # x-height.  Bounded to [0, 1] because the target (``normalised_height``)
        # is, so the two agree on scale.  Its only job is to bend the *shared*
        # encoder toward representing height; its own output is discarded at
        # inference.
        self.height_head = nn.Sequential(
            nn.Linear(config.encoder_feature_dim, config.height_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.height_hidden_dim, 1),
            nn.Sigmoid(),
        )

        n_anchors = len(self.anchors)
        if n_anchors == 0:
            raise ValueError("ContentDetectorConfig.anchors must be non-empty")

        # The anchor bank is the ``codepoint_vectors[]``: one learned slot per
        # codepoint.  Normalized at use so cosine similarity is exact.
        self.anchor_bank = nn.Parameter(
            F.normalize(torch.randn(n_anchors, config.embedding_dim), dim=-1)
        )
        # Per-anchor bias is the operating-point lever: it lets each codepoint
        # sit at its own prevalence, which per-anchor calibration later refines.
        self.anchor_bias = nn.Parameter(torch.zeros(n_anchors))
        self.logit_scale = nn.Parameter(
            torch.tensor(float(config.initial_logit_scale))
        )

    def _encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(embedding, pooled)`` for a batch of glyph images.

        The pooled feature is the encoder's global average pool, shared by the
        embedding and the auxiliary height head so that training the latter keeps
        height information in the features the former consumes.
        """
        pooled = F.adaptive_avg_pool2d(self.encoder(images), 1).flatten(1)
        return F.normalize(self.image_head(pooled), p=2, dim=-1), pooled

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized ``(B, embedding_dim)`` glyph embeddings."""
        return self._encode(images)[0]

    def normalized_anchors(self) -> torch.Tensor:
        """The anchor bank as unit vectors, ``(K, embedding_dim)``."""
        return F.normalize(self.anchor_bank, p=2, dim=-1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return ``(B, K)`` anchor logits for a batch of glyph images."""
        return self.scores(images)[0]

    def scores(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, cosine)`` for a batch of glyph images.

        Exposing the raw cosine as well lets the training loop apply margin hinges
        *before* the logit scale and per-anchor bias, where a bias cannot absorb
        them.
        """
        logits, cosine, _height = self.forward_with_aux(images)
        return logits, cosine

    def forward_with_aux(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(logits, cosine, predicted_height)`` for a batch of images."""
        z, pooled = self._encode(images)
        cosine = z @ self.normalized_anchors().T  # (B, K)
        logits = self.logit_scale * cosine + self.anchor_bias
        predicted_height = self.height_head(pooled).squeeze(-1)  # (B,)
        return logits, cosine, predicted_height

    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        """Return ``(B, K)`` independent "is this codepoint k?" probabilities."""
        return torch.sigmoid(self.forward(images))

    def probability_for(self, images: torch.Tensor, codepoint: int) -> torch.Tensor:
        """Return ``(B,)`` probability that each glyph is ``codepoint``."""
        index = self.anchors.index(codepoint)
        return self.predict_proba(images)[:, index]
