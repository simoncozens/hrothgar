"""Per-anchor calibration for the content detector.

The OVR head learns a per-anchor bias that tracks the shared training prevalence
rather than an operating point, so a single 0.5 threshold means a different
trade-off for every anchor (`recall@0.5` is therefore not comparable between
anchors).  Calibrating each anchor's raw logit with its own Platt map makes the
probabilities comparable across anchors, yields a per-anchor expected calibration
error, and supplies an operating point.

What calibration can and cannot do: it is monotone, so it cannot move any ranking
metric (AUC, recall@FAR).  It *does* fix `recall@0.5`, and it gives downstream
consumers (the detection script) an absolute, cross-anchor-comparable probability
to threshold on.

Fit from a checkpoint with::

    python -m hrothgar.content_detector.calibration --model models/content_detector.pth
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from hrothgar.calibration import PlattCalibrator, expected_calibration_error

# FAR levels at which we record a raw-logit threshold (the detector's operating
# point).  The decision itself uses calibrated probabilities; these are recorded
# for reference and for a FAR-driven threshold if we ever want one.
FAR_LEVELS = (0.01, 0.05)


@dataclass
class AnchorCalibration:
    """A Platt map for one anchor, plus its measured quality and thresholds."""

    scale: float
    bias: float
    ece: float
    far_thresholds: dict[str, float]  # FAR level (as str) -> raw-logit threshold
    n_positive: int
    n_negative: int

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Map raw anchor logits to calibrated probabilities."""
        return torch.sigmoid(self.scale * logits + self.bias)

    def to_dict(self) -> dict:
        return {
            "scale": self.scale,
            "bias": self.bias,
            "ece": self.ece,
            "far_thresholds": self.far_thresholds,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AnchorCalibration:
        return cls(
            scale=float(data["scale"]),
            bias=float(data["bias"]),
            ece=float(data["ece"]),
            far_thresholds={
                str(k): float(v) for k, v in data.get("far_thresholds", {}).items()
            },
            n_positive=int(data.get("n_positive", 0)),
            n_negative=int(data.get("n_negative", 0)),
        )


@dataclass
class ContentCalibrator:
    """Per-anchor calibrators for a content-detector checkpoint."""

    anchors: list[int]
    per_anchor: dict[int, AnchorCalibration] = field(default_factory=dict)

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Map a ``(N, K)`` block of anchor logits to calibrated probabilities."""
        probabilities = torch.sigmoid(logits)  # fallback for un-calibrated anchors
        for k, anchor in enumerate(self.anchors):
            calibration = self.per_anchor.get(anchor)
            if calibration is not None:
                probabilities[:, k] = calibration.calibrate(logits[:, k])
        return probabilities

    def anchor(self, codepoint: int) -> AnchorCalibration | None:
        return self.per_anchor.get(codepoint)

    def to_dict(self) -> dict:
        from hrothgar.utils import git_short_sha

        return {
            "anchors": list(self.anchors),
            "git_sha": git_short_sha(),
            "per_anchor": {
                str(cp): calibration.to_dict()
                for cp, calibration in self.per_anchor.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> ContentCalibrator:
        return cls(
            anchors=[int(a) for a in data["anchors"]],
            per_anchor={
                int(cp): AnchorCalibration.from_dict(cal)
                for cp, cal in data["per_anchor"].items()
            },
        )

    @staticmethod
    def _path_for(model_path) -> Path:
        return Path(str(model_path).replace(".pth", ".calibration.json"))

    def save(self, model_path) -> None:
        with self._path_for(model_path).open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            f.write("\n")

    @classmethod
    def load(cls, model_path) -> ContentCalibrator:
        path = cls._path_for(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Calibration sidecar not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def classify_slot(
    probabilities: torch.Tensor,
    claimed_index: int,
    anchors: list[int],
    *,
    ambiguous: float = 0.5,
    reject_below: float = 0.30,
    confident_above: float = 0.70,
    margin: float = 0.0,
) -> tuple[str, int | None]:
    """Label one slot's calibrated probabilities: ``"ok"``/``"wrong"``/``"ambiguous"``.

    Ambiguity is tested first: if no anchor clears ``ambiguous`` the glyph does not
    look like anything the model knows (a symbol/barcode glyph, or a malformed one),
    so there is nothing to accuse it of.

    ``"wrong"`` then requires the model to *reject the claim* — the claimed anchor
    below ``reject_below`` — **and** a different anchor to be confident (at least
    ``confident_above``, and ahead of the claim by ``margin``).  Requiring outright
    rejection, rather than merely "some other anchor ranks higher", is what
    separates a real accent swap from two similar anchors trading places on a
    perfectly good glyph.  Laila's ``á`` slot (which holds an ``à``) reads
    ``claimed≈0.0, competitor≈1.0`` and fires; a legitimate ``a`` that scores
    ``p(a)=0.89, p(à)=0.99`` does not, because the ``a`` anchor accepts it.

    Returns ``(label, best_anchor)`` where ``best_anchor`` is the competing
    codepoint for a ``"wrong"`` verdict and ``None`` otherwise.
    """
    claimed = float(probabilities[claimed_index])
    best_index = int(probabilities.argmax())
    best = float(probabilities[best_index])
    if best < ambiguous:
        return "ambiguous", None
    if (
        best_index != claimed_index
        and claimed < reject_below
        and best >= confident_above
        and best > claimed + margin
    ):
        return "wrong", anchors[best_index]
    return "ok", None


def fit_content_calibrator(
    model,
    panel,
    config,
    device,
    *,
    far_levels: tuple[float, ...] = FAR_LEVELS,
) -> ContentCalibrator:
    """Fit a per-anchor Platt map on a held-out evaluation panel.

    Positives are the anchor's own glyphs; negatives are the anchor's held-out
    *hard* negatives (confusables and diacritic variants) — the same population
    ``recall@FAR`` is measured against, falling back to all off-anchor glyphs when
    a panel carries none.
    """
    from hrothgar.content_detector.dataset import hard_negative_pool
    from hrothgar.content_detector.evaluation import score_panel

    # Scored in chunks (see ``score_panel``): one pass over the whole panel OOMs
    # at 256px.
    _embeddings, logits = score_panel(model, panel, device)

    index = config.anchor_index()
    hard_sets = {
        a: set(hard_negative_pool(a, config.negatives)) for a in config.anchors
    }

    per_anchor: dict[int, AnchorCalibration] = {}
    for anchor in config.anchors:
        k = index[anchor]
        positives = [
            float(logits[n, k])
            for n, cp in enumerate(panel.codepoints)
            if cp == anchor
        ]
        negatives = [
            float(logits[n, k])
            for n, cp in enumerate(panel.codepoints)
            if cp != anchor and cp in hard_sets[anchor]
        ]
        if not negatives:
            negatives = [
                float(logits[n, k])
                for n, cp in enumerate(panel.codepoints)
                if cp != anchor
            ]
        if not positives or not negatives:
            continue

        positive_scores = torch.tensor(positives, dtype=torch.float32)
        negative_scores = torch.tensor(negatives, dtype=torch.float32)
        platt = PlattCalibrator.fit(positive_scores, negative_scores)

        probs = platt.predict_proba(torch.cat([positive_scores, negative_scores]))
        labels = torch.cat(
            [
                torch.ones(positive_scores.numel()),
                torch.zeros(negative_scores.numel()),
            ]
        )
        per_anchor[anchor] = AnchorCalibration(
            scale=platt.scale,
            bias=platt.bias,
            ece=expected_calibration_error(probs, labels),
            far_thresholds={
                str(far): float(torch.quantile(negative_scores, 1.0 - far))
                for far in far_levels
            },
            n_positive=len(positives),
            n_negative=len(negatives),
        )

    return ContentCalibrator(anchors=list(config.anchors), per_anchor=per_anchor)


def main() -> int:
    """Fit and save calibration for a content-detector checkpoint."""
    from hrothgar.content_detector.config import ContentDetectorConfig
    from hrothgar.content_detector.dataset import ContentDatasetMaker
    from hrothgar.content_detector.evaluation import build_content_panel
    from hrothgar.content_detector.model import ContentDetector
    from hrothgar.utils import pick_device

    parser = argparse.ArgumentParser(
        description="Fit per-anchor calibration for a content-detector checkpoint."
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts"),
    )
    parser.add_argument("--model", required=True, help="path to the .pth checkpoint")
    parser.add_argument("--max-fonts", type=int, default=200)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None, help="override the sidecar path")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    config = ContentDetectorConfig.from_sidecar(args.model)
    model = ContentDetector(config)
    model.load(args.model, device)
    model.to(device)
    model.eval()

    maker = ContentDatasetMaker(args.repo, config)
    panel = build_content_panel(maker.test_fonts, config, max_fonts=args.max_fonts)
    calibrator = fit_content_calibrator(model, panel, config, device)

    out_path = args.out or calibrator._path_for(args.model)
    with Path(out_path).open("w", encoding="utf-8") as f:
        json.dump(calibrator.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"panel: {len(panel.codepoints)} glyphs  ->  {out_path}")
    for anchor in calibrator.anchors:
        calibration = calibrator.anchor(anchor)
        if calibration is None:
            print(f"  {chr(anchor)} (U+{anchor:04X}): no calibration (no panel data)")
            continue
        print(
            f"  {chr(anchor)} (U+{anchor:04X}): scale={calibration.scale:+.2f} "
            f"bias={calibration.bias:+.2f} ece={calibration.ece:.3f} "
            f"n_pos={calibration.n_positive} n_neg={calibration.n_negative}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
