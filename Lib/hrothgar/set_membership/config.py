"""Instance-membership model — configuration.

A "font" here is an *instance* (a file at a specific axis position), so the
model can judge whether a glyph belongs to a specific weight/width/style of a
family, not just the family.  This is the generator-evaluation use case: a
generated glyph is scored against the instance it was asked to match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hrothgar.dataset_constants import LATIN_CORE

@dataclass
class SetMembershipConfig:
    """Configuration for the instance-membership model."""

    # Input rendering: uniform scale-and-pad glyph images (see
    # ``hrothgar.render.render``), matching the framing the generator will
    # denormalize to.
    input_codepoints: list[int] = field(
        default_factory=lambda: list(LATIN_CORE)
    )
    glyph_size: int = 128
    # Glyphs rendered per instance per batch (>=2, so positives exist).
    codepoints_per_instance: int = 8
    # Instances drawn per family per batch (same-family hard negatives).
    instances_per_family: int = 2
    # Validation panel: reference/query glyph split (disjoint sets).
    reference_glyphs: int = 8
    query_glyphs: int = 8

    # CNN encoder.
    encoder_base_channels: int = 32
    encoder_feature_dim: int = 256
    encoder_downsample: int = 4
    # L2-normalized embedding space shared by glyphs and instance prototypes.
    embedding_dim: int = 128

    # Geometry labels are kept in the data pipeline for easy revert, but the
    # model currently ignores them (image-only).  ``use_geometry`` /
    # ``geometry_hidden_dim`` are retained for that reason and are unused.
    use_geometry: bool = True
    geometry_hidden_dim: int = 32

    # Instance enumeration: how many axis positions a variable font contributes.
    max_instances_per_font: int = 4

    # Training.
    contrastive_temperature: float = 0.07

    def save_sidecar(self, model_path):
        """Save config as a sidecar JSON alongside the model weights."""
        import json
        from dataclasses import asdict
        from pathlib import Path as _Path

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
        import dataclasses
        import json
        from pathlib import Path as _Path

        config_path = _Path(str(model_path).replace(".pth", ".conf.json"))
        if not config_path.exists():
            raise FileNotFoundError(f"Config sidecar not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
