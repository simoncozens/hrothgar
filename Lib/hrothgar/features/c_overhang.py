"""c-overhang recogniser: the terminal treatment of the lowercase ``c``'s overhang.

Reads the CEDARS+ ``c-overhang`` label from the ``c`` glyph.  The overhang is the
stroke that extends right of the bowl; its terminal shape/orientation (hook /
diagonal / circular / horizontal / rounded / vertical / serif / other) is the
style-bearing signal.  It is a local feature, so like e-crossbar / finials it uses
a spatial-grid pool rather than a global max-pool.

All-caps/small-caps fonts are excluded from training/eval (their lowercase ``c``
slot holds an uppercase ``C``, so the feature is out of domain) — the same
confound the ``stem-terminals`` recogniser skips.

Usage::

    python -m hrothgar.features.c_overhang --mode train \
        --repo ~/others-repos/fonts --model-path models/c_overhang.pth
    python -m hrothgar.features.c_overhang --mode evaluate \
        --repo ~/others-repos/fonts --model-path models/c_overhang.pth
    python -m hrothgar.features.c_overhang --mode recognise \
        --model-path models/c_overhang.pth --ttf path/to/Font-Regular.ttf
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

from hrothgar.features.base import (
    FeatureRecognizer,
    FeatureRecognizerConfig,
    _all_caps_families,
)
from hrothgar.cedars import load_cedars
from hrothgar.features.build import build_task
from hrothgar.googlefonts import families_to_fonts

C_OVERHANG_CLASSES = [
    "hook",
    "diagonal",
    "circular",
    "horizontal",
    "rounded",
    "vertical",
    "serif",
    "other",
]


@dataclass
class COverhangConfig(FeatureRecognizerConfig):
    glyphs: str = "c"
    classes: list[str] = field(default_factory=lambda: list(C_OVERHANG_CLASSES))
    spatial_grid: int = 8
    # Crop to the right half of the glyph (columns [0.5, 1.0]): the c-overhang
    # terminal lives on the right edge, so the left bowl is discarded and the
    # canvas is spent on the feature of interest.
    crop_right_frac: float = 0.5
    # Exclude all-caps/small-caps fonts (fontquant ``casing/lowercase_shapes``):
    # their lowercase ``c`` slot holds an uppercase ``C``.
    skip_uppercase: bool = True


class COverhangRecognizer(FeatureRecognizer):
    config_cls = COverhangConfig

    @staticmethod
    def label_fn(fq: dict | None, cedars_row: dict | None) -> str | None:
        if cedars_row is None:
            return None
        val = cedars_row.get("c-overhang", "").strip()
        return val if val else None

    @classmethod
    def excluded_families(cls, family_to_font: dict, cedars: dict) -> set:
        """All-caps/small-caps fonts (uppercase ``C`` in the lowercase slot)."""
        return _all_caps_families(family_to_font)

    def build_real(self, repo: str):
        cedars = load_cedars()
        family_to_font = families_to_fonts(repo)
        if self.config.skip_uppercase:
            skip = _all_caps_families(family_to_font)
            cedars = {k: v for k, v in cedars.items() if k not in skip}
        task = {
            "glyphs": self.config.glyphs,
            "classes": self.class_names,
            "source": "cedars",
            "col": "c-overhang",
        }
        X, y, class_names, _inj, families = build_task(
            task, family_to_font, cedars, self.config.glyph_size,
            seed=self.config.seed, crop_right_frac=self.config.crop_right_frac,
            expand_axes=self.config.expand_axes,
        )
        return X, y, class_names, families


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("train", "evaluate", "recognise"), required=True)
    p.add_argument("--repo", default=os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts"))
    p.add_argument("--model-path", default="models/c_overhang.pth")
    p.add_argument("--ttf", help="TTF to judge (--mode recognise)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.mode == "train":
        config = COverhangConfig(epochs=args.epochs, glyph_size=args.size)
        COverhangRecognizer.train_from_repo(args.repo, args.model_path, config=config, device=args.device)
    elif args.mode == "evaluate":
        COverhangRecognizer.evaluate(args.repo, args.model_path, device=args.device)
    else:
        if not args.ttf:
            p.error("--ttf is required for --mode recognise")
        model = COverhangRecognizer.from_checkpoint(args.model_path, device=args.device)
        print(model.recognise(args.ttf, device=args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
