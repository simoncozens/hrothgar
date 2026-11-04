"""finials recogniser: ``restrained`` / ``flourished`` / ``mixed`` + "unrecognisable".

Third member of the feature-recogniser family.  Reads the CEDARS+ ``finials``
label from the ``alt`` glyph set (``a``, ``l``, ``t`` — stroke ends show the
finial treatment).  Finials are a local feature (the stroke terminals), so it
uses a spatial grid pool like e-crossbar rather than a global max-pool.

Usage::

    python -m hrothgar.features.finials --mode train \
        --repo ~/others-repos/fonts --model-path models/finials.pth
    python -m hrothgar.features.finials --mode evaluate \
        --repo ~/others-repos/fonts --model-path models/finials.pth
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

FINIALS_CLASSES = ["restrained", "flourished", "mixed"]


@dataclass
class FinialsConfig(FeatureRecognizerConfig):
    glyphs: str = "alt"
    classes: list[str] = field(default_factory=lambda: list(FINIALS_CLASSES))
    spatial_grid: int = 8


class FinialsRecognizer(FeatureRecognizer):
    config_cls = FinialsConfig

    @staticmethod
    def label_fn(fq: dict | None, cedars_row: dict | None) -> str | None:
        if cedars_row is None:
            return None
        val = cedars_row.get("finials", "").strip()
        return val if val else None

    def build_real(self, repo: str):
        cedars = _load_cedars()
        fontquant = _load_fontquant_raw()
        family_to_font = _family_to_font(repo)
        task = {
            "glyphs": self.config.glyphs,
            "classes": self.class_names,
            "source": "cedars",
            "col": "finials",
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
    p.add_argument("--model-path", default="models/finials.pth")
    p.add_argument("--ttf", help="TTF to judge (--mode recognise)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.mode == "train":
        config = FinialsConfig(epochs=args.epochs, glyph_size=args.size)
        FinialsRecognizer.train_from_repo(args.repo, args.model_path, config=config, device=args.device)
    elif args.mode == "evaluate":
        FinialsRecognizer.evaluate(args.repo, args.model_path, device=args.device)
    else:
        if not args.ttf:
            p.error("--ttf is required for --mode recognise")
        model = FinialsRecognizer.from_checkpoint(args.model_path, device=args.device)
        print(model.recognise(args.ttf, device=args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
