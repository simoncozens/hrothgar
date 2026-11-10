"""a-storey recogniser: ``single`` / ``double`` / ``uppercase`` + "unrecognisable".

The first concrete member of the feature-recogniser family.  It folds fontquant's
``appearance/lowercase_a_style`` and ``casing/lowercase_shapes`` into three closed
classes (a glyph whose ``a`` slot holds an all-caps/small-caps shape is
"uppercase"), and bulk-ads the ``uppercase`` class by injecting each font's real
``A`` glyph.  "unrecognisable" is read from a calibrated threshold on the max
sigmoid.

Usage::

    python -m hrothgar.features.a_storey --mode train \
        --repo ~/others-repos/fonts --model-path models/a_storey.pth
    python -m hrothgar.features.a_storey --mode evaluate \
        --repo ~/others-repos/fonts --model-path models/a_storey.pth
    python -m hrothgar.features.a_storey --mode recognise \
        --model-path models/a_storey.pth --ttf path/to/Font-Regular.ttf
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

from hrothgar.cedars import load_cedars
from hrothgar.features.base import FeatureRecognizer, FeatureRecognizerConfig
from hrothgar.features.build import build_task
from hrothgar.googlefonts import families_to_fonts

A_STOREY_CLASSES = ["single", "double", "uppercase"]


def _a_storey_label(fq: dict | None, cedars_row: dict | None) -> str | None:
    """Fold ``lowercase_a_style`` + ``casing/lowercase_shapes`` into 3 classes.

    ``uppercase`` covers all-caps and small-caps designs: the ``a`` slot does
    not hold a real lowercase ``a``, so asking "single or double storey" of it
    is nonsense.  (Small-caps is folded into ``uppercase`` for now; a separate
    small-caps / unrecognisable class can come later.)
    """
    if fq is None:
        return None
    shapes = fq.get("casing/lowercase_shapes", "").strip()
    a_style = fq.get("appearance/lowercase_a_style", "").strip()
    if shapes in ("uppercase", "smallcaps"):
        return "uppercase"
    if a_style == "single_story":
        return "single"
    if a_style == "double_story":
        return "double"
    return None


@dataclass
class AStoreyConfig(FeatureRecognizerConfig):
    glyphs: str = "a"
    classes: list[str] = field(default_factory=lambda: list(A_STOREY_CLASSES))
    inject_uppercase_A: int = 240


class AStoreyRecognizer(FeatureRecognizer):
    config_cls = AStoreyConfig

    @staticmethod
    def label_fn(fq: dict | None, cedars_row: dict | None) -> str | None:
        """Fold fontquant ``a_style`` + ``lowercase_shapes`` into the 3 classes."""
        return _a_storey_label(fq, cedars_row)

    def build_real(self, repo: str):
        """Build the real ``a``-slot dataset (with the uppercase-``A`` injection)."""
        cedars = load_cedars()
        family_to_font = families_to_fonts(repo)
        task = {
            "glyphs": self.config.glyphs,
            "classes": self.class_names,
            "label_fn": self.label_fn,
            "inject": {"glyphs": "A", "label": "uppercase", "n": self.config.inject_uppercase_A},
        }
        X, y, class_names, _inj, families = build_task(
            task, family_to_font, cedars, self.config.glyph_size,
            seed=self.config.seed, expand_axes=self.config.expand_axes,
        )
        return X, y, class_names, families


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("train", "evaluate", "recognise"), required=True)
    p.add_argument("--repo", default=os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts"))
    p.add_argument("--model-path", default="models/a_storey.pth")
    p.add_argument("--ttf", help="TTF to judge (--mode recognise)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.mode == "train":
        config = AStoreyConfig(epochs=args.epochs, glyph_size=args.size)
        AStoreyRecognizer.train_from_repo(args.repo, args.model_path, config=config, device=args.device)
    elif args.mode == "evaluate":
        AStoreyRecognizer.evaluate(args.repo, args.model_path, device=args.device)
    else:
        if not args.ttf:
            p.error("--ttf is required for --mode recognise")
        model = AStoreyRecognizer.from_checkpoint(args.model_path, device=args.device)
        print(model.recognise(args.ttf, device=args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
