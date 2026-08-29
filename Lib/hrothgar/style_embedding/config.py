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
    font, encodes each glyph with a shared CNN, downsamples each to a coarse
    spatial grid, pools across glyphs (content-invariance), and projects the
    flattened map into a compact summary.  That summary feeds contrastive,
    tag-prediction, and category objectives.
    """

    # Input rendering.
    input_codepoints: list[int] = field(
        default_factory=lambda: list(DEFAULT_INPUT_CODEPOINTS)
    )
    glyph_size: int = 128
    # Number of glyphs sampled per font per step (memory + robustness).
    glyph_sample_size: int = 32

    # CNN encoder.
    encoder_base_channels: int = 32
    encoder_feature_dim: int = 256
    encoder_downsample: int = 4  # 64 -> 16

    # Coarse-spatial pooling (replaces Gram).  Each glyph is downsampled to a
    # coarse grid, then pooled across glyphs; a linear maps the flattened map
    # back to ``encoder_feature_dim``.  Coarse enough to blur the skeleton
    # (content-invariant), fine enough to keep serif/terminal shape.
    coarse_grid_size: int = 8
    spatial_channels: int = 32

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
    # Soft positive weight for same-family, different-font pairs in the
    # multi-positive contrastive loss (font-level objective).  A weight of 0
    # disables the family-level soft positive entirely.
    family_positive_weight: float = 0.3

    # Training.
    contrastive_temperature: float = 0.07

    def save_sidecar(self, model_path):
        """Save config as a sidecar JSON alongside the model weights."""
        import json
        from pathlib import Path as _Path
        from dataclasses import asdict

        from hrothgar.utils import git_short_sha
        config_path = _Path(str(model_path).replace(".pth", ".conf.json"))
        data = asdict(self)
        data["git_sha"] = git_short_sha()
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
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

    multipos_contrastive: float = 1.0
    tag_prediction: float = 0.5
    family_positive_weight: float = 0.3
    tag_positive_weight: float = 1.0
