"""Configuration for the weight-adjustment model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class WeightAdjustConfig:
    """Configuration for the deterministic exemplar-conditioned CNN."""

    # Frame / conditioning.
    image_size: int = 128
    num_exemplars: int = 5

    # Style encoder.
    style_dim: int = 128
    style_base_channels: int = 32

    # Residual decoder.
    decoder_base_channels: int = 64
    decoder_num_blocks: int = 8

    # Regularization.
    dropout: float = 0.1

    # Weight axis locations used to synthesize training pairs (variable fonts).
    regular_weight: int = 400
    bold_weight: int = 700

    # Training.
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    advance_loss_weight: float = 1.0

    def save_sidecar(self, model_path) -> None:
        import hrothgar.utils as u

        config_path = Path(str(model_path)).with_suffix(".conf.json")
        data = asdict(self)
        data["git_sha"] = u.git_short_sha()
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")

    @classmethod
    def from_sidecar(cls, model_path):
        config_path = Path(str(model_path)).with_suffix(".conf.json")
        if not config_path.exists():
            raise FileNotFoundError(f"Config sidecar not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
