"""Expressive tag prediction — configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from hrothgar.similarity.config import DEFAULT_INPUT_CODEPOINTS

# The five texture-y expressive tags we currently predict.
DEFAULT_TAG_NAMES = [
    "/Expressive/Childlike",
    "/Expressive/Fancy",
    "/Expressive/Rugged",
    "/Expressive/Vintage",
    "/Expressive/Playful",
]


@dataclass
class TaggingConfig:
    """Configuration for the frozen-encoder tag prediction head.

    The input is a *gap* feature computed from the frozen encoder's per-glyph
    spatial map: global average pool per glyph, then mean across glyphs
    (optionally concatenated with the cross-glyph standard deviation, i.e.
    ``gap_mv``).  The head is a small MLP predicting one scalar per tag.
    """

    # Input rendering (matches the frozen similarity encoder).
    input_codepoints: list[int] = field(
        default_factory=lambda: list(DEFAULT_INPUT_CODEPOINTS)
    )
    glyph_size: int = 128

    # Feature extraction.
    use_variance: bool = True  # gap_mv (mean + cross-glyph std) vs gap (mean)

    # Head.
    tag_names: list[str] = field(default_factory=lambda: list(DEFAULT_TAG_NAMES))
    hidden_dim: int = 64
    hidden_layers: int = 1
    dropout: float = 0.0

    # Loss.
    loss: str = "huber"  # "mse" | "huber"
    # Centiles are on 0-100; we normalise to [0, 1] before the loss and
    # denormalise for reporting.
    normalize_target: bool = True

    def save_sidecar(self, model_path) -> None:
        import json
        from dataclasses import asdict
        from pathlib import Path as _Path

        from hrothgar.utils import git_short_sha

        path = _Path(str(model_path).replace(".pth", ".conf.json"))
        data = asdict(self)
        data["git_sha"] = git_short_sha()
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")

    @classmethod
    def from_sidecar(cls, model_path):
        import dataclasses
        import json
        from pathlib import Path as _Path

        path = _Path(str(model_path).replace(".pth", ".conf.json"))
        if not path.exists():
            raise FileNotFoundError(f"Config sidecar not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
