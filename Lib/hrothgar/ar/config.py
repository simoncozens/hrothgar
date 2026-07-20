from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ARModelConfig:
    """Configuration for the MaskGIT glyph generator."""

    image_size: int = 128
    encoder_feature_dim: int = 256

    style_encoder_base_channels: int = 32

    aggregator_num_layers: int = 3
    aggregator_num_heads: int = 8

    decoder_hidden_dim: int = 832
    decoder_num_layers: int = 16
    decoder_num_heads: int = 16
    decoder_dropout: float = 0.1
    decoder_attention_dropout: float = 0.1

    style_dropout: float = 0.2

    # Global style vector: pool frozen G-Tok ViT features into a single
    # (B, encoder_feature_dim) vector and broadcast-add it to every
    # spatial position of the fused conditioning map.  This gives every
    # generation token an identical, globally-consistent style signal —
    # helpful for scripts (e.g. Latin) whose glyphs carry sparse,
    # non-redundant style cues.
    use_global_style: bool = True
    # Dropout probability applied to the global style vector (per-batch).
    global_style_dropout: float = 0.2

    content_only_prob: float = 0.0
    style_only_prob: float = 0.0

    freeze_gtok: bool = True

    maskgit_num_inference_steps: int = 8
    maskgit_temperature: float = 1.0

    # Metric conditioning (concern 3: baseline/x-height/width alignment).
    # If False, the metric embedder and width head are not created.
    use_metrics: bool = True
    metric_embedding_hidden_dim: int = 128
    width_head_hidden_dim: int = 128

    # Training metadata — used at inference to validate inputs.
    # None means "trained on full Latin Core" (the default).
    target_codepoints: Optional[list[int]] = None
    target_only: bool = False
    style_codepoints: Optional[list[int]] = None

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.encoder_feature_dim % self.aggregator_num_heads != 0:
            raise ValueError(
                "encoder_feature_dim must be divisible by aggregator_num_heads "
                f"(got {self.encoder_feature_dim} and {self.aggregator_num_heads})"
            )
        if self.decoder_hidden_dim % self.decoder_num_heads != 0:
            raise ValueError(
                "decoder_hidden_dim must be divisible by decoder_num_heads "
                f"(got {self.decoder_hidden_dim} and {self.decoder_num_heads})"
            )

    def save_sidecar(self, model_path):
        """Save config as a sidecar JSON alongside the model weights."""
        from pathlib import Path as _Path
        import json as _json
        from dataclasses import asdict as _asdict
        config_path = _Path(str(model_path).replace(".pth", ".conf.json"))
        with config_path.open("w", encoding="utf-8") as f:
            _json.dump(_asdict(self), f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"Saved AR model config to {config_path}")

    @classmethod
    def from_sidecar(cls, model_path):
        """Load config from a sidecar JSON alongside the model weights."""
        from pathlib import Path as _Path
        import json as _json
        import dataclasses as _dc
        config_path = _Path(model_path).with_suffix(".conf.json")
        if not config_path.exists():
            config_path = _Path(str(model_path).replace(".pth", ".conf.json"))
        if not config_path.exists():
            raise FileNotFoundError(
                f"AR model config sidecar not found: {config_path}\n"
                "Run AR training first so the .conf.json is written "
                "alongside the .pth."
            )
        with config_path.open("r", encoding="utf-8") as f:
            data = _json.load(f)
        known = {f.name for f in _dc.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
