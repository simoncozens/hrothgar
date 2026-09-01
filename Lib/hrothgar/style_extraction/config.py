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
import math
from dataclasses import dataclass, field
from pathlib import Path

from hrothgar.dataset import LATIN_KERNEL


@dataclass
class StyleExtractionConfig:
    """Configuration for the style autoencoder."""

    # Rendering.
    image_size: int = 128

    # Target codepoint vocabulary.  These are the codepoints we know how to
    # *render* (embedded via ``CodepointEmbedding``).  The evidence glyphs are
    # sampled from the same set for now; widen this to LGC_ALL or beyond for
    # richer style evidence.
    character_set: list[int] = field(default_factory=lambda: list(LATIN_KERNEL))

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
    style_contrastive: float = 0.1
    token_diversity: float = 0.1
    position_reg: float = 0.0  # guard against global (position-independent) cross-attention


@dataclass(frozen=True)
class LossSchedule:
    """Cosine coarse-to-fine schedule for reconstruction loss weights.

    ``_ramp`` interpolates a weight from its start value to ``*_final`` over
    ``schedule_steps`` using a cosine curve (smooth, zero derivative at both
    ends).  After ``schedule_steps`` the weight stays at its final value.

    Defaults are deliberately aggressive: L1 (the blur ceiling) and LPIPS
    (structural) ramp *down* while glyphloss (fine-detail, curvature-weighted)
    ramps *up*, so the localized signal dominates before the decoder can lock
    into the global-style shortcut (the v2/v3 collapse).
    """

    schedule_steps: int = 20000
    l1_final: float = 0.05
    glyphloss_final: float = 3.0
    lpips_final: float = 0.3

    def _ramp(self, step: int, start: float, end: float) -> float:
        if self.schedule_steps <= 0 or start == end:
            return start
        t = min(max(int(step), 0), self.schedule_steps) / self.schedule_steps
        return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * t))

    def l1(self, step: int, start: float) -> float:
        return self._ramp(step, start, self.l1_final)

    def glyphloss(self, step: int, start: float) -> float:
        return self._ramp(step, start, self.glyphloss_final)

    def lpips(self, step: int, start: float) -> float:
        return self._ramp(step, start, self.lpips_final)


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


@dataclass
class StyleExtractionV3Config(StyleExtractionV2Config):
    """v3 config: content-conditioned SPADE decoder.

    Same encoder/Perceiver as v2; the decoder replaces the additive
    cross-attention (which collapsed to a global style vector) with a
    content-conditioned SPADE head — cross-attention produces a per-position
    style map that spatially-adaptively normalizes the CNN features.
    """

    # Self-attention layers applied to the content queries before cross-attention,
    # so the skeleton can cohere before the style map is read out.
    decoder_self_attn_layers: int = 2

    # Learned absolute positional bias in the cross-attention logits.  Breaks the
    # position-independence symmetry that let v2/v3 collapse to a global style
    # vector (q_var ≈ 0): each query position gets a fixed, learnable bias over
    # the K tokens, which the content-conditioned q·k term then refines.
    cross_attn_pos_bias: bool = True
    # When False (default), the bias is *fixed* (non-learnable sinusoidal) so the
    # model cannot collapse it away.  True uses a learnable bias (v2-style).
    cross_attn_pos_bias_trainable: bool = False
    # Magnitude of the fixed positional bias (only used when
    # ``cross_attn_pos_bias`` and not ``cross_attn_pos_bias_trainable``).  This is
    # the *scale* of the random projection in ``_fixed_sinusoidal_pos_bias``, not
    # a std directly.  ``q_var`` is the variance of near-uniform attention weights
    # (~1/K each), so it scales as the *square* of the logit perturbation:
    #   0.1 -> q_var ~5e-6  (invisible against the content baseline)
    #   0.5 -> q_var ~3e-4  (clear, still reads ~28 tokens/position)
    #   1.0 -> q_var ~1.8e-3 (strong, but underuses tokens: eff_tokens ~6.5)
    # 0.5 is the default: strong enough to force position-dependence for the
    # experiment, not so strong it degenerates to a content-blind lookup.
    cross_attn_pos_bias_scale: float = 0.5


@dataclass
class StyleExtractionV4Config(StyleExtractionV3Config):
    """v4 config: two-stage coarse-to-fine decoder.

    Stage 1 (coarse) is L1-driven and produces a coarse glyph; stage 2 (fine)
    encodes that coarse glyph into spatial features and uses them as the query
    for a glyphloss-driven SPADE refinement.  The coarse output supplies the
    spatial condition that v3's cross-attention query (codepoint + learned
    position) was missing.
    """

    two_stage: bool = True
