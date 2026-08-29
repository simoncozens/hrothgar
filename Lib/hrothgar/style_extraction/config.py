"""Configuration for the style-extraction (autoencoding) model.

Unlike ``hrothgar.style_embedding`` (which learns a *compact, contrastive*
style vector for similarity/recommendation), this module learns a *rich,
complete* style representation that must be decodable back into glyphs at
high fidelity.  Reconstruction is the primary objective and the primary
acceptance signal.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

from hrothgar.dataset import LATIN_CORE


@dataclass
class StyleExtractionConfig:
    """Configuration for the style autoencoder."""

    # Rendering.
    image_size: int = 128

    # Target codepoint vocabulary.  These are the codepoints we know how to
    # *render* (embedded via ``CodepointEmbedding``).  The evidence glyphs are
    # sampled from the same set for now; widen this to LGC_ALL or beyond for
    # richer style evidence.
    character_set: list[int] = field(default_factory=lambda: list(LATIN_CORE))

    # Number of evidence glyphs sampled per font per step.  More evidence gives
    # a richer style summary; the aggregator is permutation-invariant so the
    # exact count can vary at inference time.
    num_evidence_glyphs: int = 32

    # Per-glyph CNN encoder.  Preserves spatial structure (no Gram) so fine
    # geometric style detail (terminals, serifs, stroke modulation) survives.
    glyph_encoder_base_channels: int = 32
    glyph_encoder_feature_dim: int = 256
    glyph_encoder_downsample: int = 4  # image_size 128 -> 32

    # Decoder.
    decoder_base_channels: int = 128
    decoder_num_res_blocks: int = 2

    def save_sidecar(self, model_path) -> None:
        """Save config as a JSON sidecar alongside the model weights."""
        import hrothgar.utils as u

        config_path = Path(str(model_path)).with_suffix(".conf.json")
        data = dataclasses.asdict(self)
        data["git_sha"] = u.git_short_sha()
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")

    @classmethod
    def from_sidecar(cls, model_path):
        """Load config from a JSON sidecar alongside the model weights."""
        config_path = Path(str(model_path)).with_suffix(".conf.json")
        if not config_path.exists():
            raise FileNotFoundError(f"Config sidecar not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def num_codepoints(self) -> int:
        """Size of the codepoint embedding table."""
        return len(self.character_set)


@dataclass(frozen=True)
class StyleExtractionLossWeights:
    """Weights for the reconstruction objectives.

    ``adversarial`` defaults to 0 so the first version trains as a stable,
    deterministic autoencoder.  Raise it (e.g. 0.1) to enable the PatchGAN
    adversarial term for crisper terminals.
    """

    l1: float = 1.0
    glyphloss: float = 1.0
    perceptual_lpips: float = 1.0
    adversarial: float = 0.0
    ink_coverage: float = 0.5


@dataclass
class StyleExtractionV2Config(StyleExtractionConfig):
    """v2 config: Perceiver style tokens + cross-attention decoder.

    Adds transformer hyperparameters.  The token dimension is
    ``glyph_encoder_feature_dim`` (no extra projection), and the positional
    embedding is shared between the reference glyph tokens and the decoder
    content queries (both live on the normalized glyph grid).
    """

    num_evidence_glyphs: int = 16  # fewer than v1: G*N feeds the Perceiver
    num_style_tokens: int = 128
    perceiver_num_layers: int = 2
    perceiver_num_heads: int = 8
    decoder_num_layers: int = 4
    decoder_num_heads: int = 8
    decoder_dropout: float = 0.0
    # Coarse structural grid for reference-token position tags (replaces a fine
    # per-pixel positional embedding on the style side).
    coarse_grid_size: int = 8
