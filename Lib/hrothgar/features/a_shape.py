"""double-storey-``a`` shape recogniser: ``flat`` / ``angled`` / ``convex`` /
``concave`` + "unrecognisable".

The finer-grained counterpart to ``a_storey`` (which only says single vs double
storey vs uppercase).  This recogniser reads the CEDARS+ ``double-storey-a``
label from the ``a`` glyph — the shape of the **bowl/stem junction** in the
middle of a double-storey ``a``.  Because the column is only populated for
double-storey constructions, the gating (run only on known double-storey ``a``)
is automatic: single-storey, all-caps and small-caps fonts leave it empty and
are skipped.

It crops to the middle half of the ink box (``crop_middle=(0.25, 0.75)``) to
focus on the junction, and uses a spatial grid pool rather than a global
max-pool.

Usage::

    python -m hrothgar.features.a_shape --mode train \
        --repo ~/others-repos/fonts --model-path models/a_shape.pth
    python -m hrothgar.features.a_shape --mode evaluate \
        --repo ~/others-repos/fonts --model-path models/a_shape.pth
    python -m hrothgar.features.a_shape --mode recognise \
        --model-path models/a_shape.pth --ttf path/to/Font-Regular.ttf
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

from hrothgar.features.base import FeatureRecognizer, FeatureRecognizerConfig
from hrothgar.outline_tokens.oracle_baseline import _load_fontquant_raw
from hrothgar.outline_tokens.recognizer_audit import (
    _build_task,
    _family_to_font,
    _load_cedars,
)

A_SHAPE_CLASSES = ["flat", "angled", "convex", "concave"]


@dataclass
class AShapeConfig(FeatureRecognizerConfig):
    glyphs: str = "a"
    classes: list[str] = field(default_factory=lambda: list(A_SHAPE_CLASSES))
    spatial_grid: int = 8
    crop_middle: tuple[float, float] = (0.25, 0.75)


class AShapeRecognizer(FeatureRecognizer):
    config_cls = AShapeConfig

    @staticmethod
    def label_fn(fq: dict | None, cedars_row: dict | None) -> str | None:
        if cedars_row is None:
            return None
        val = cedars_row.get("double-storey-a", "").strip()
        return val if val else None

    def build_real(self, repo: str):
        cedars = _load_cedars()
        fontquant = _load_fontquant_raw()
        family_to_font = _family_to_font(repo)
        task = {
            "glyphs": self.config.glyphs,
            "classes": self.class_names,
            "source": "cedars",
            "col": "double-storey-a",
        }
        X, y, class_names, _inj, families = _build_task(
            task, family_to_font, cedars, fontquant, self.config.glyph_size,
            seed=self.config.seed, crop_middle=self.config.crop_middle,
            expand_axes=self.config.expand_axes,
        )
        return X, y, class_names, families


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("train", "evaluate", "recognise"), required=True)
    p.add_argument("--repo", default=os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts"))
    p.add_argument("--model-path", default="models/a_shape.pth")
    p.add_argument("--ttf", help="TTF to judge (--mode recognise)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.mode == "train":
        config = AShapeConfig(epochs=args.epochs, glyph_size=args.size)
        AShapeRecognizer.train_from_repo(args.repo, args.model_path, config=config, device=args.device)
    elif args.mode == "evaluate":
        AShapeRecognizer.evaluate(args.repo, args.model_path, device=args.device)
    else:
        if not args.ttf:
            p.error("--ttf is required for --mode recognise")
        model = AShapeRecognizer.from_checkpoint(args.model_path, device=args.device)
        print(model.recognise(args.ttf, device=args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
