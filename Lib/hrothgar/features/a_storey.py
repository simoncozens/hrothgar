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

from hrothgar.features.base import FeatureRecognizer, FeatureRecognizerConfig
from hrothgar.outline_tokens.oracle_baseline import _load_fontquant_raw
from hrothgar.outline_tokens.recognizer_audit import (
    _a_storey_label,
    _build_task,
    _family_to_font,
    _load_cedars,
)

A_STOREY_CLASSES = ["single", "double", "uppercase"]


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
        cedars = _load_cedars()
        fontquant = _load_fontquant_raw()
        family_to_font = _family_to_font(repo)
        task = {
            "glyphs": self.config.glyphs,
            "classes": self.class_names,
            "label_fn": self.label_fn,
            "inject": {"glyphs": "A", "label": "uppercase", "n": self.config.inject_uppercase_A},
        }
        X, y, class_names, _inj, families = _build_task(
            task, family_to_font, cedars, fontquant, self.config.glyph_size,
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
