"""e-crossbar recogniser: ``flat`` / ``angled`` / ``rounded`` / ``na`` + "unrecognisable".

Second member of the feature-recogniser family.  Reads the CEDARS+
``e-crossbar`` label from the ``e`` glyph.  ``na`` is a real (hand-labelled)
"the crossbar doesn't fit flat/angled/rounded" class, kept as a closed class;
"unrecognisable" is the threshold verdict for a non-``e`` glyph (symbol / barcode
/ redaction fonts).

Usage::

    python -m hrothgar.features.e_crossbar --mode train \
        --repo ~/others-repos/fonts --model-path models/e_crossbar.pth
    python -m hrothgar.features.e_crossbar --mode evaluate \
        --repo ~/others-repos/fonts --model-path models/e_crossbar.pth
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

from hrothgar.features.base import FeatureRecognizer, FeatureRecognizerConfig
from hrothgar.cedars import load_cedars
from hrothgar.features.build import build_task
from hrothgar.googlefonts import families_to_fonts

E_CROSSBAR_CLASSES = ["flat", "angled", "rounded", "na"]


@dataclass
class ECrossbarConfig(FeatureRecognizerConfig):
    glyphs: str = "e"
    classes: list[str] = field(default_factory=lambda: list(E_CROSSBAR_CLASSES))
    spatial_grid: int = 8


class ECrossbarRecognizer(FeatureRecognizer):
    config_cls = ECrossbarConfig

    @staticmethod
    def label_fn(fq: dict | None, cedars_row: dict | None) -> str | None:
        if cedars_row is None:
            return None
        val = cedars_row.get("e-crossbar", "").strip()
        return val if val else None

    def build_real(self, repo: str):
        cedars = load_cedars()
        family_to_font = families_to_fonts(repo)
        task = {
            "glyphs": self.config.glyphs,
            "classes": self.class_names,
            "source": "cedars",
            "col": "e-crossbar",
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
    p.add_argument("--model-path", default="models/e_crossbar.pth")
    p.add_argument("--ttf", help="TTF to judge (--mode recognise)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.mode == "train":
        config = ECrossbarConfig(epochs=args.epochs, glyph_size=args.size)
        ECrossbarRecognizer.train_from_repo(args.repo, args.model_path, config=config, device=args.device)
    elif args.mode == "evaluate":
        ECrossbarRecognizer.evaluate(args.repo, args.model_path, device=args.device)
    else:
        if not args.ttf:
            p.error("--ttf is required for --mode recognise")
        model = ECrossbarRecognizer.from_checkpoint(args.model_path, device=args.device)
        print(model.recognise(args.ttf, device=args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
