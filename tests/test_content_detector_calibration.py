"""Behavioural tests for content-detector calibration."""

from __future__ import annotations

import torch

from hrothgar.content_detector.calibration import (
    ContentCalibrator,
    classify_slot,
    fit_content_calibrator,
)
from hrothgar.content_detector.config import ContentDetectorConfig
from hrothgar.content_detector.dataset import ContentDatasetMaker
from hrothgar.content_detector.evaluation import build_content_panel
from hrothgar.content_detector.model import ContentDetector

DUMMY_REPO = "tests/dummy_repo"


def _config(**overrides) -> ContentDetectorConfig:
    defaults = dict(
        anchors=[ord(c) for c in "aeog"],
        glyph_size=32,
        anchors_per_batch=3,
        fonts_per_anchor=2,
        num_workers=0,
        panel_fonts=4,
    )
    defaults.update(overrides)
    return ContentDetectorConfig(**defaults)


def _fitted() -> tuple[ContentCalibrator, ContentDetector]:
    config = _config()
    maker = ContentDatasetMaker(DUMMY_REPO, config)
    panel = build_content_panel(maker.test_fonts, config, max_fonts=4)
    model = ContentDetector(config).eval()
    return fit_content_calibrator(model, panel, config, "cpu"), model


def test_fit_produces_a_calibrator_per_anchor() -> None:
    calibrator, _model = _fitted()
    config = _config()
    assert calibrator.anchors == list(config.anchors)
    assert set(calibrator.per_anchor) == set(config.anchors)
    for calibration in calibrator.per_anchor.values():
        assert set(calibration.far_thresholds) == {"0.01", "0.05"}
        assert 0.0 <= calibration.ece <= 1.0
        assert calibration.n_positive > 0 and calibration.n_negative > 0


def test_calibrate_is_bounded_and_monotone() -> None:
    calibrator, _model = _fitted()
    logits = torch.linspace(-8, 8, 32).unsqueeze(1).repeat(1, len(calibrator.anchors))
    probabilities = calibrator.calibrate(logits)
    assert probabilities.shape == logits.shape
    assert bool(((probabilities >= 0) & (probabilities <= 1)).all())
    # A Platt map is monotone in the logit (in one direction or the other).
    for k in range(len(calibrator.anchors)):
        diffs = probabilities[:, k].diff()
        assert bool((diffs >= -1e-6).all() or (diffs <= 1e-6).all())


def test_empty_calibrator_is_the_raw_sigmoid() -> None:
    logits = torch.randn(6, 3)
    empty = ContentCalibrator(anchors=[1, 2, 3], per_anchor={})
    assert torch.allclose(empty.calibrate(logits), torch.sigmoid(logits))


def test_classify_slot_probability_rule() -> None:
    anchors = [ord("a"), ord("e")]
    # Confident and correct.
    assert classify_slot(torch.tensor([0.9, 0.1]), 0, anchors) == ("ok", None)
    # The competing anchor wins and the claim is rejected (Laila: à shape in á slot).
    assert classify_slot(torch.tensor([0.08, 0.9]), 0, anchors) == ("wrong", ord("e"))
    # Nothing clears the threshold -> it looks like no anchor at all.
    assert classify_slot(torch.tensor([0.30, 0.20]), 0, anchors) == (
        "ambiguous",
        None,
    )
    # A margin suppresses a marginally-higher wrong anchor.
    assert classify_slot(torch.tensor([0.45, 0.50]), 0, anchors, margin=0.9) == (
        "ok",
        None,
    )


def test_classify_slot_cosine_margin_rule() -> None:
    anchors = [ord("á"), ord("à")]
    # Laila, scored *centred*: the claimed probability is lifted above any useful
    # threshold (0.70), so a probability gate cannot fire, but the cosine ranking
    # still puts the à anchor decisively closer.
    probabilities = torch.tensor([0.70, 1.00])
    assert classify_slot(
        probabilities, 0, anchors, cosine=torch.tensor([0.14, 0.80])
    ) == ("wrong", ord("à"))

    # A legitimate á whose à anchor is a little closer is not accused: the margin
    # (0.10) is under the default 0.35 even though à is the cosine winner.
    assert classify_slot(
        probabilities, 0, anchors, cosine=torch.tensor([0.60, 0.50])
    ) == ("ok", None)

    # A large cosine margin is not enough if the competitor is not confident.
    assert classify_slot(
        torch.tensor([0.60, 0.65]), 0, anchors, cosine=torch.tensor([0.10, 0.60])
    ) == ("ok", None)

    # Ambiguity is still tested first.
    assert classify_slot(
        torch.tensor([0.30, 0.20]), 0, anchors, cosine=torch.tensor([0.10, 0.60])
    ) == ("ambiguous", None)


def test_legitimate_glyph_is_not_flagged_when_a_neighbour_ranks_higher() -> None:
    anchors = [ord("a"), ord("à")]
    # A perfectly good `a` where the à anchor ranks slightly higher: the `a`
    # anchor still *accepts* the glyph, so this must not be accused.
    assert classify_slot(torch.tensor([0.89, 0.99]), 0, anchors) == ("ok", None)
    # A real accent swap: the claim is rejected and the competitor is confident.
    assert classify_slot(torch.tensor([0.00, 1.00]), 0, anchors) == ("wrong", ord("à"))
    # The reject threshold is the knob that separates them.
    assert classify_slot(
        torch.tensor([0.89, 0.99]), 0, anchors, reject_below=0.95
    ) == ("wrong", ord("à"))

    # The centred cosine rule divides the same two cases the same way.
    assert classify_slot(
        torch.tensor([0.89, 0.99]), 0, anchors, cosine=torch.tensor([0.55, 0.60])
    ) == ("ok", None)
    assert classify_slot(
        torch.tensor([0.89, 0.99]), 0, anchors, cosine=torch.tensor([0.10, 0.80])
    ) == ("wrong", ord("à"))


def test_calibrator_roundtrip() -> None:
    calibrator, _model = _fitted()
    restored = ContentCalibrator.from_dict(calibrator.to_dict())
    assert restored.anchors == calibrator.anchors
    assert set(restored.per_anchor) == set(calibrator.per_anchor)


def test_calibrator_save_load(tmp_path) -> None:
    calibrator, _model = _fitted()
    model_path = str(tmp_path / "cd.pth")
    calibrator.save(model_path)
    loaded = ContentCalibrator.load(model_path)
    assert loaded.anchors == calibrator.anchors
    assert set(loaded.per_anchor) == set(calibrator.per_anchor)
