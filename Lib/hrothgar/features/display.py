"""display/text discriminator: regresses a continuous display score (0-100).

Unlike the other feature recognisers (which classify into a closed set), this
one is a *regression*: it predicts the tag-derived ``compute_display_score``
(adherence to body-text conventions vs expressive freedom) from the
style-bearing specimen ``hamburgefonts``.  The predicted score is the routing
signal for the text-vs-display similarity paths, so it is deliberately
continuous (semi-display is a real, useful value).

Usage::

    python -m hrothgar.features.display --mode train \
        --repo ~/others-repos/fonts --model-path models/display.pth
    python -m hrothgar.features.display --mode evaluate \
        --repo ~/others-repos/fonts --model-path models/display.pth
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import torch

from hrothgar.features.base import FeatureRecognizer, FeatureRecognizerConfig
from hrothgar.features.build import is_blank, render_prompts
from hrothgar.googlefonts import compute_display_score, families_to_fonts


@dataclass
class DisplayConfig(FeatureRecognizerConfig):
    regression: bool = True
    glyphs: str = "hamburgefonts"
    score_max: float = 100.0
    spatial_grid: int = 1


class DisplayRecognizer(FeatureRecognizer):
    config_cls = DisplayConfig

    def build_real(self, repo: str):
        """Render the specimen and regress to ``compute_display_score`` (0-100)."""
        family_to_font = families_to_fonts(repo)
        X: list[torch.Tensor] = []
        y: list[float] = []
        for fam in sorted(family_to_font):
            font = family_to_font[fam]
            img = render_prompts(font, self.config.glyphs, self.config.glyph_size)
            if img is None or is_blank(img):
                continue
            X.append(img)
            y.append(compute_display_score(font.tags()))
        return torch.stack(X), torch.tensor(y, dtype=torch.float32)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("train", "evaluate", "recognise"), required=True)
    p.add_argument("--repo", default=os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts"))
    p.add_argument("--model-path", default="models/display.pth")
    p.add_argument("--ttf", help="TTF to judge (--mode recognise)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.mode == "train":
        config = DisplayConfig(epochs=args.epochs, glyph_size=args.size)
        DisplayRecognizer.train_from_repo(args.repo, args.model_path, config=config, device=args.device)
    elif args.mode == "evaluate":
        DisplayRecognizer.evaluate(args.repo, args.model_path, device=args.device)
    else:
        if not args.ttf:
            p.error("--ttf is required for --mode recognise")
        model = DisplayRecognizer.from_checkpoint(args.model_path, device=args.device)
        print(model.recognise(args.ttf, device=args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
