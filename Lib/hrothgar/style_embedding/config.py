"""Font style embedding — model configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FontStyleEmbedderConfig:
    """Configuration for the font-level style embedder.

    The model renders a short phrase (e.g. "THE quick brown fox 1234") in the
    target font, encodes the resulting image with a CNN, global-average-pools
    to a single embedding vector, and projects through contrastive and
    tag-prediction heads.
    """

    # Input rendering.
    # Native render size for the phrase image.
    phrase_width: int = 1536
    phrase_font_size: int = 72
    # Target size to resize the phrase rendering to before CNN encoding.
    # Rectangular (6:1) preserves text shape without dead pixels.
    phrase_target_width: int = 768
    phrase_target_height: int = 128

    # CNN encoder (matches StyleEncoder convention).
    encoder_base_channels: int = 32
    encoder_feature_dim: int = 256

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
    encoder_dropout: float = 0.1

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
    tag_prediction: float = 0.5
