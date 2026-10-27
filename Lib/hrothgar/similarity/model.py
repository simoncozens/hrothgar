"""Font-level style embedder from a fixed set of glyph renderings.

Encodes each glyph with a shared CNN into a coarse spatial map, then aggregates
the per-glyph maps with a permutation-invariant set encoder (self-attention over
glyphs) instead of a mean pool, so the summary can retain the fine per-glyph
style (aperture, terminal shape) that averaging destroys.  The summary feeds
contrastive, tag-prediction, and category heads.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from hrothgar.googlefonts import Font
from hrothgar.similarity.config import FontStyleEmbedderConfig
from hrothgar.glyph_encoder import GlyphEncoder
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


class SetEncoder(nn.Module):
    """Permutation-invariant aggregation of per-glyph features via self-attention.

    Replaces the mean-over-glyphs pool: a learned ``[CLS]`` token attends over the
    glyph set, so the summary can focus on style-bearing glyphs and their
    inter-glyph structure instead of averaging them away.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.net = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self, x: torch.Tensor, glyph_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``x`` is ``(B, G, d_model)``; returns the ``[CLS]`` summary ``(B, d_model)``."""
        b = x.shape[0]
        tokens = torch.cat([self.cls_token.expand(b, -1, -1), x], dim=1)  # (B, 1+G, d_model)
        if glyph_mask is not None:
            # True = pad (masked out); the [CLS] token is never masked.
            pad = torch.cat(
                [torch.zeros(b, 1, dtype=torch.bool, device=x.device), ~glyph_mask],
                dim=1,
            )
        else:
            pad = None
        out = self.net(tokens, src_key_padding_mask=pad)  # (B, 1+G, d_model)
        return out[:, 0]  # (B, d_model)


class FontStyleEmbedder(SaveLoadModel):
    """Font-level style embedder via max-pooled glyph maps + set encoder."""

    def __init__(self, config: FontStyleEmbedderConfig):
        super().__init__()
        self.config = config

        self.encoder = GlyphEncoder(
            base_channels=config.encoder_base_channels,
            out_dim=config.spatial_map_channels,
            downsample=config.encoder_downsample,
        )

        # Spatial reduction.  A 1x1 projection compresses the encoder's
        # ``spatial_map_channels`` down to ``spatial_channels``, then each glyph
        # is max-pooled to ``coarse_grid_size`` (max, not average, so sparse
        # local features like terminal rounding are preserved) before the set
        # encoder pools across glyphs.
        self.spatial_proj = nn.Conv2d(
            config.spatial_map_channels, config.spatial_channels, 1, bias=False
        )
        self.glyph_proj = nn.Linear(
            config.spatial_channels * config.coarse_grid_size * config.coarse_grid_size,
            config.encoder_feature_dim,
        )
        self.set_encoder = SetEncoder(
            config.encoder_feature_dim,
            num_heads=config.set_encoder_heads,
            num_layers=config.set_encoder_layers,
            dropout=config.set_encoder_dropout,
        )

        # Reconstruction head: decodes the summary back to the raw
        # content-averaged glyph shape.  Trained with a cosine loss against the
        # (fixed) shape computed from the input images, so the summary must
        # retain the fine visual style it would otherwise discard.
        self.reconstruction_head = nn.Linear(
            config.encoder_feature_dim,
            config.reconstruction_grid * config.reconstruction_grid,
        )

        self.enc_dropout = nn.Dropout(config.encoder_dropout)

        # Projection head for contrastive loss.  Dropout is applied to the
        # output so the sequential's parameter names do not shift.
        self.projection = nn.Sequential(
            nn.Linear(config.encoder_feature_dim, config.encoder_feature_dim),
            nn.SiLU(),
            nn.Linear(config.encoder_feature_dim, config.projection_dim),
        )
        self.projection_dropout = nn.Dropout(config.projection_dropout)

        # Tag prediction heads.
        self.tag_heads: nn.ModuleDict | None = None
        if config.tag_names:
            self.tag_heads = nn.ModuleDict(
                {
                    name: TagPredictionHead(
                        config.encoder_feature_dim,
                        config.tag_hidden_dim,
                        num_classes=config.tag_num_classes,
                        dropout=config.tag_dropout,
                    )
                    for name in config.tag_names
                }
            )

        # Broad category head.
        self.category_head: CategoryPredictionHead | None = None
        if config.use_category_head:
            self.category_head = CategoryPredictionHead(
                config.encoder_feature_dim,
                hidden_dim=config.category_hidden_dim,
                dropout=config.category_dropout,
            )

        # Optional text projection — maps frozen text embeddings into the
        # same space as image projections for multi-positive contrastive loss.
        self.text_projection: nn.Linear | None = None
        if config.text_encoder_name:
            self.text_projection = nn.Linear(
                config.text_embedding_dim, config.projection_dim
            )

    def _encode(
        self,
        images: torch.Tensor,
        glyph_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a batch of glyph sets into ``(B, encoder_feature_dim)`` summaries."""
        b, g, c, h, w = images.shape
        assert c == 1, f"Expected single-channel glyphs, got {c} channels"

        flat = images.flatten(0, 1)  # (B*G, 1, H, W)
        features = self.encoder(flat)  # (B*G, F, h', w')

        s = self.config.spatial_channels
        cs = self.config.coarse_grid_size
        proj = self.spatial_proj(features)  # (B*G, S, h', w')
        proj = F.adaptive_max_pool2d(proj, (cs, cs))  # (B*G, S, cs, cs) — max pool
        proj = proj.reshape(b, g, s, cs, cs)  # (B, G, S, cs, cs)

        glyphs = proj.flatten(2)  # (B, G, S*cs*cs)
        glyphs = self.glyph_proj(glyphs)  # (B, G, encoder_feature_dim)
        summary = self.set_encoder(glyphs, glyph_mask)  # (B, encoder_feature_dim)
        return self.enc_dropout(summary)

    def encode(
        self,
        images: torch.Tensor,
        glyph_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode glyph sets into the summary style embedding.

        Args:
            images: ``(B, G, 1, H, W)`` greyscale glyph renderings in [0, 1].
            glyph_mask: Optional ``(B, G)`` boolean, ``True`` = glyph visible.

        Returns:
            ``(B, encoder_feature_dim)`` summary vector.
        """
        return self._encode(images, glyph_mask)

    def compute_embedding(
        self,
        font: Font,
        device: torch.device | None = None,
        axis_position: list[float] | None = None,
    ) -> torch.Tensor:
        """Render a font's full input set and return its style embedding.

        This is the single source of truth for turning a font into the vector
        consumed by downstream tasks (similarity search, tag prediction, and
        the AR generator).  It renders the same glyph set and applies the same
        rendering logic used by the training dataset, so the two can never
        drift apart.
        """
        from hrothgar.render import frame_for_codepoints, render

        frame = frame_for_codepoints(font, self.config.input_codepoints)
        images = (
            torch.stack(
                [
                    render(
                        font,
                        codepoint=cp,
                        size=self.config.glyph_size,
                        axis_position=axis_position,
                        frame=frame,
                    )[0]
                    for cp in self.config.input_codepoints
                ]
            )
            .unsqueeze(1)
            .unsqueeze(0)
        )  # (1, G, 1, H, W)

        # A blank (no-ink) glyph renders all-white, so surface blank glyphs
        # here so callers can apply their unrenderable-font fallback instead of
        # embedding blank space.
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
        assert (
            self.text_projection is not None
        ), "text_projection not initialised; set text_encoder_name in config"
        if torch.isnan(text_embeddings).any():
            raise RuntimeError("NaN in text_embeddings input to project_text")
        projected = self.text_projection(text_embeddings)
        if torch.isnan(projected).any():
            raise RuntimeError("NaN in text_projection output")
        return F.normalize(projected, p=2, dim=-1, eps=1e-8)

    def forward(
        self,
        images: torch.Tensor,
        glyph_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor] | None,
        torch.Tensor | None,
        torch.Tensor,
    ]:
        """Full forward pass.

        Returns ``(embedding, projection, tags, category_logits, reconstruction)``.
        ``reconstruction`` is the summary decoded back to the raw glyph shape and
        feeds the visual reconstruction auxiliary (see ``reconstruction_loss``).
        """
        embedding = self._encode(images, glyph_mask)
        projection = F.normalize(
            self.projection_dropout(self.projection(embedding)), p=2, dim=-1
        )
        rg = self.config.reconstruction_grid
        reconstruction = self.reconstruction_head(embedding).view(-1, 1, rg, rg)

        tags: dict[str, torch.Tensor] | None = None
        if self.tag_heads is not None:
            tags = {name: head(embedding) for name, head in self.tag_heads.items()}

        category_logits: torch.Tensor | None = None
        if self.category_head is not None:
            category_logits = self.category_head(embedding)

        return embedding, projection, tags, category_logits, reconstruction
