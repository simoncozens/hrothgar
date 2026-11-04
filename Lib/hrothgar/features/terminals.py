"""stem-terminals recogniser: ``blunted`` / ``soft`` / ``flared`` / ``angled`` /
``tapered`` / ``rounded`` + "unrecognisable".

The non-serif counterpart to the ``serifs`` recogniser, routed to by
``has_serifs``.  Reads the CEDARS+ ``stem-terminals`` label from the ``lmn``
glyphs (the stems of ``l``, ``m``, ``n`` show how a stroke ends in a sans/display
face).  Stem terminals are a local feature, so it uses a spatial grid pool like
e-crossbar and finials rather than a global max-pool.

Usage::

    python -m hrothgar.features.terminals --mode train \
        --repo ~/others-repos/fonts --model-path models/terminals.pth
    python -m hrothgar.features.terminals --mode evaluate \
        --repo ~/others-repos/fonts --model-path models/terminals.pth
    python -m hrothgar.features.terminals --mode recognise \
        --model-path models/terminals.pth --ttf path/to/Font-Regular.ttf
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

from hrothgar.features.base import (
    FeatureRecognizer,
    FeatureRecognizerConfig,
    _all_caps_families,
)
from hrothgar.outline_tokens.oracle_baseline import _load_fontquant_raw
from hrothgar.outline_tokens.recognizer_audit import (
    _build_task,
    _family_to_font,
    _load_cedars,
)

TERMINALS_CLASSES = ["blunted", "soft", "flared", "angled", "tapered", "rounded"]


def _script_families(family_to_font: dict) -> set[str]:
    """Families tagged ``/Script/*`` (connected scripts).

    Their stems flow into each other, so "stem terminals" are not meaningfully
    evaluable — any cedars label on them is a source of label noise.
    """
    skip: set[str] = set()
    for fam, font in family_to_font.items():
        if any(tag.startswith("/Script/") for tag in font.tags()):
            skip.add(fam)
    return skip


@dataclass
class TerminalsConfig(FeatureRecognizerConfig):
    glyphs: str = "lmn"
    classes: list[str] = field(default_factory=lambda: list(TERMINALS_CLASSES))
    spatial_grid: int = 8
    crop_bottom_frac: float = 0.25
    # Exclude all-caps/small-caps fonts (fontquant ``casing/lowercase_shapes``)
    # from training: their lowercase ``lmn`` slots hold uppercase shapes.
    skip_uppercase: bool = True
    # Exclude ``/Script/*`` families (connected scripts) from training.
    skip_script: bool = False
    expand_axes: bool = False


class TerminalsRecognizer(FeatureRecognizer):
    config_cls = TerminalsConfig

    @staticmethod
    def label_fn(fq: dict | None, cedars_row: dict | None) -> str | None:
        if cedars_row is None:
            return None
        val = cedars_row.get("stem-terminals", "").strip()
        return val if val else None

    @classmethod
    def excluded_families(cls, family_to_font: dict, cedars: dict, fontquant: dict) -> set:
        """All-caps/small-caps fonts, plus connected scripts (no real terminals)."""
        return _all_caps_families(family_to_font, fontquant) | _script_families(family_to_font)

    def build_real(self, repo: str):
        cedars = _load_cedars()
        fontquant = _load_fontquant_raw()
        family_to_font = _family_to_font(repo)
        if self.config.skip_uppercase or self.config.skip_script:
            skip = _all_caps_families(family_to_font, fontquant)
            if self.config.skip_script:
                skip |= _script_families(family_to_font)
            cedars = {k: v for k, v in cedars.items() if k not in skip}
        task = {
            "glyphs": self.config.glyphs,
            "classes": self.class_names,
            "source": "cedars",
            "col": "stem-terminals",
        }
        X, y, class_names, _inj, families = _build_task(
            task, family_to_font, cedars, fontquant, self.config.glyph_size,
            seed=self.config.seed, crop_bottom_frac=self.config.crop_bottom_frac,
            expand_axes=self.config.expand_axes,
        )
        return X, y, class_names, families


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("train", "evaluate", "recognise"), required=True)
    p.add_argument("--repo", default=os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts"))
    p.add_argument("--model-path", default="models/terminals.pth")
    p.add_argument("--ttf", help="TTF to judge (--mode recognise)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.mode == "train":
        config = TerminalsConfig(epochs=args.epochs, glyph_size=args.size)
        TerminalsRecognizer.train_from_repo(args.repo, args.model_path, config=config, device=args.device)
    elif args.mode == "evaluate":
        TerminalsRecognizer.evaluate(args.repo, args.model_path, device=args.device)
    else:
        if not args.ttf:
            p.error("--ttf is required for --mode recognise")
        model = TerminalsRecognizer.from_checkpoint(args.model_path, device=args.device)
        print(model.recognise(args.ttf, device=args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
