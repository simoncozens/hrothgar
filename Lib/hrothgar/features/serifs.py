"""serifs recogniser: ``slab`` / ``bracketed`` / ``wedge`` / ``hairline`` /
``asymmetric`` + "unrecognisable" (``calligraphic``/``bifurcated`` are dropped).

The serif counterpart to the ``terminals`` recogniser, routed to by
``has_serifs``.  Reads the CEDARS+ ``serifs`` label from the ``lmn`` glyphs
(the feet of ``l``, ``m``, ``n`` show the serif construction).  Serif shape is a
local feature at the baseline, so — like ``terminals`` — it crops the bottom of
the glyph and uses a spatial grid pool.

Usage::

    python -m hrothgar.features.serifs --mode train \
        --repo ~/others-repos/fonts --model-path models/serifs.pth
    python -m hrothgar.features.serifs --mode evaluate \
        --repo ~/others-repos/fonts --model-path models/serifs.pth
    python -m hrothgar.features.serifs --mode recognise \
        --model-path models/serifs.pth --ttf path/to/Font-Regular.ttf
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

SERIFS_CLASSES = ["slab", "bracketed", "wedge", "hairline", "asymmetric"]

#: Serif styles we deliberately do not model.  ``calligraphic`` is a catch-all for
#: script-like serifs, and ``bifurcated`` has only a handful of (very unusual)
#: examples.  These are filtered out of both training and eval.
DROPPED_SERIFS = ["calligraphic", "bifurcated"]


@dataclass
class SerifsConfig(FeatureRecognizerConfig):
    glyphs: str = "lmn"
    classes: list[str] = field(default_factory=lambda: list(SERIFS_CLASSES))
    spatial_grid: int = 8
    crop_bottom_frac: float = 0.25
    # Exclude all-caps/small-caps fonts (fontquant ``casing/lowercase_shapes``):
    # their lowercase ``lmn`` slots hold uppercase shapes, whose serifs differ.
    skip_uppercase: bool = True
    expand_axes: bool = False


class SerifsRecognizer(FeatureRecognizer):
    config_cls = SerifsConfig

    @staticmethod
    def label_fn(fq: dict | None, cedars_row: dict | None) -> str | None:
        if cedars_row is None:
            return None
        val = cedars_row.get("serifs", "").strip()
        return val if val else None

    @classmethod
    def excluded_families(cls, family_to_font: dict, cedars: dict) -> set:
        """All-caps/small-caps fonts, plus the dropped calligraphic/bifurcated styles."""
        skip = _all_caps_families(family_to_font)
        for fam, row in cedars.items():
            if (row.get("serifs", "") or "").strip() in DROPPED_SERIFS:
                skip.add(fam)
        return skip

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
            "col": "serifs",
        }
        X, y, class_names, _inj, families = build_task(
            task, family_to_font, cedars, self.config.glyph_size,
            seed=self.config.seed, crop_bottom_frac=self.config.crop_bottom_frac,
            expand_axes=self.config.expand_axes,
        )
        return X, y, class_names, families


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("train", "evaluate", "recognise"), required=True)
    p.add_argument("--repo", default=os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts"))
    p.add_argument("--model-path", default="models/serifs.pth")
    p.add_argument("--ttf", help="TTF to judge (--mode recognise)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.mode == "train":
        config = SerifsConfig(epochs=args.epochs, glyph_size=args.size)
        SerifsRecognizer.train_from_repo(args.repo, args.model_path, config=config, device=args.device)
    elif args.mode == "evaluate":
        SerifsRecognizer.evaluate(args.repo, args.model_path, device=args.device)
    else:
        if not args.ttf:
            p.error("--ttf is required for --mode recognise")
        model = SerifsRecognizer.from_checkpoint(args.model_path, device=args.device)
        print(model.recognise(args.ttf, device=args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
