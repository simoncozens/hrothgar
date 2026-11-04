"""has-serifs recogniser: ``serif`` / ``sans`` + "unrecognisable".

The routing gate for the two-end-of-stem recognisers.  CEDARS+ tags *serif
shape* (``serifs``) only for serif fonts and *stem terminals*
(``stem-terminals``) only for non-serif fonts — the two columns are mutually
exclusive (confirmed: 0 fonts tagged with both).  This recogniser learns that
partition from the ``lmn`` glyphs, so an unseen font can be routed to the
``serifs`` recogniser (``serif``) or the ``terminals`` recogniser (``sans``)
without hand tags.

Usage::

    python -m hrothgar.features.has_serifs --mode train \
        --repo ~/others-repos/fonts --model-path models/has_serifs.pth
    python -m hrothgar.features.has_serifs --mode evaluate \
        --repo ~/others-repos/fonts --model-path models/has_serifs.pth
    python -m hrothgar.features.has_serifs --mode recognise \
        --model-path models/has_serifs.pth --ttf path/to/Font-Regular.ttf
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

HAS_SERIFS_CLASSES = ["serif", "sans"]


@dataclass
class HasSerifsConfig(FeatureRecognizerConfig):
    glyphs: str = "lmn"
    classes: list[str] = field(default_factory=lambda: list(HAS_SERIFS_CLASSES))
    spatial_grid: int = 8


class HasSerifsRecognizer(FeatureRecognizer):
    config_cls = HasSerifsConfig

    @staticmethod
    def label_fn(fq: dict | None, cedars_row: dict | None) -> str | None:
        """``serif`` if ``serifs`` is assessed, ``sans`` if ``stem-terminals`` is."""
        if cedars_row is None:
            return None
        if (cedars_row.get("serifs", "") or "").strip():
            return "serif"
        if (cedars_row.get("stem-terminals", "") or "").strip():
            return "sans"
        return None

    def build_real(self, repo: str):
        cedars = _load_cedars()
        fontquant = _load_fontquant_raw()
        family_to_font = _family_to_font(repo)
        task = {
            "glyphs": self.config.glyphs,
            "classes": self.class_names,
            "label_fn": self.label_fn,
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
    p.add_argument("--model-path", default="models/has_serifs.pth")
    p.add_argument("--ttf", help="TTF to judge (--mode recognise)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.mode == "train":
        config = HasSerifsConfig(epochs=args.epochs, glyph_size=args.size)
        HasSerifsRecognizer.train_from_repo(args.repo, args.model_path, config=config, device=args.device)
    elif args.mode == "evaluate":
        HasSerifsRecognizer.evaluate(args.repo, args.model_path, device=args.device)
    else:
        if not args.ttf:
            p.error("--ttf is required for --mode recognise")
        model = HasSerifsRecognizer.from_checkpoint(args.model_path, device=args.device)
        print(model.recognise(args.ttf, device=args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
