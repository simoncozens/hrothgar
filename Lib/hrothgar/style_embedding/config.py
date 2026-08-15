"""Font style embedding — model configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

# Fixed glyph set rendered as the embedder input.  This is the Latin core
# (upper/lower/digits) plus two punctuation marks chosen to be style-revealing.
DEFAULT_INPUT_CHARS = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    "&?"
)
DEFAULT_INPUT_CODEPOINTS = [ord(c) for c in DEFAULT_INPUT_CHARS]


@dataclass
class FontStyleEmbedderConfig:
    """Configuration for the font-level style embedder.

    The model renders a fixed set of glyphs (``input_codepoints``) for each
    font, encodes each glyph with a shared CNN, attention-pools each glyph's
    spatial features into a small number of tokens, and then aggregates those
    per-glyph tokens into a global style token set plus a 256-d summary vector.
    """

    # Input rendering.
    input_codepoints: list[int] = field(
        default_factory=lambda: list(DEFAULT_INPUT_CODEPOINTS)
    )
    glyph_size: int = 128

    # CNN encoder.
    encoder_base_channels: int = 32
    encoder_feature_dim: int = 256
    encoder_downsample: int = 4  # 64 -> 16

    # Hierarchical pooling.
    per_glyph_tokens: int = 4
    style_latents: int = 16
    aggregation_heads: int = 8

    # Glyph-layout head.  Predicts hidden glyphs' ink bounding boxes from the
    # summary vector + glyph-slot identity, forcing the summary to retain
    # typographic proportions (width / height / sidebearing / placement).
    use_layout: bool = True
    layout_codepoint_dim: int = 64
    layout_samples: int = 4

    # Shape head.  Reconstructs each hidden glyph's bbox-normalized square image
    # from the summary vector + glyph-slot identity.
    use_shape: bool = True
    shape_codepoint_dim: int = 64
    shape_samples: int = 4

    # Final embedding dimensionality for contrastive loss.
    projection_dim: int = 128

    # Tag prediction.
    tag_names: list[str] = field(default_factory=list)
    tag_hidden_dim: int = 16
    tag_dropout: float = 0.4
    tag_num_classes: int = 0  # 0=regression, 2=binary, 4=quartiles

    # Broad category classification.
    use_category_head: bool = False
    category_hidden_dim: int = 16
    category_dropout: float = 0.3

    # Regularization.
    encoder_dropout: float = 0.3

    # Text conditioning — frozen text encoder provides a training signal
    # via multi-positive contrastive loss.  Not needed at inference.
    text_encoder_name: str = ""
    text_embedding_dim: int = 384
    multipos_use_family_positives: bool = True

    # Training.
    contrastive_temperature: float = 0.07

    def save_sidecar(self, model_path):
        """Save config as a sidecar JSON alongside the model weights."""
        import json
        from pathlib import Path as _Path
        from dataclasses import asdict

        config_path = _Path(str(model_path).replace(".pth", ".conf.json"))
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)
            f.write("\n")

    @classmethod
    def from_sidecar(cls, model_path):
        """Load config from a sidecar JSON alongside the model weights."""
        import json
        import dataclasses
        from pathlib import Path as _Path

        config_path = _Path(str(model_path).replace(".pth", ".conf.json"))
        if not config_path.exists():
            config_path = _Path(str(model_path)).with_suffix(".conf.json")
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config sidecar not found: {config_path}\n"
                "Run training first so the .conf.json is written alongside the .pth."
            )
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class FontStyleEmbeddingLossWeights:
    """Weights for each loss term."""

    contrastive: float = 1.0
    multipos_contrastive: float = 1.0
    tag_prediction: float = 0.5
    layout: float = 1.0
    shape: float = 1.0
    use_family_positives: bool = True
    tag_positive_weight: float = 1.0
