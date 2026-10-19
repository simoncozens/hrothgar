"""Behavioural tests for the calibrated in-font confidence deliverable."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hrothgar.set_membership.calibration import (
    ConfidenceScorer,
    PlattCalibrator,
    expected_calibration_error,
    operating_points,
    reference_loo_scores,
)


def _clusters(
    n: int, per: int, dim: int, seed: int, scale: float = 5.0
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    centres = torch.eye(n, dim) * scale
    rows = []
    for i in range(n):
        for _ in range(per):
            rows.append(
                F.normalize(centres[i] + torch.randn(dim, generator=g) * 0.05, dim=-1)
            )
    return torch.stack(rows)


def test_reference_loo_scores_are_cosines() -> None:
    refs = _clusters(1, 6, 16, seed=0)
    scores = reference_loo_scores(refs)
    assert scores.shape == (6,)
    assert torch.isfinite(scores).all()
    assert (scores >= -1).all() and (scores <= 1).all()


def test_reference_loo_scores_needs_two_references() -> None:
    import pytest

    with pytest.raises(ValueError):
        reference_loo_scores(F.normalize(torch.randn(1, 8), dim=-1))


def test_platt_calibrator_maps_higher_to_higher() -> None:
    in_scores = torch.linspace(1.0, 3.0, 100)
    out_scores = torch.linspace(-3.0, -1.0, 100)
    calibrator = PlattCalibrator.fit(in_scores, out_scores)
    assert calibrator.scale > 0
    assert calibrator.predict_proba(torch.tensor(3.0)) > calibrator.predict_proba(
        torch.tensor(-3.0)
    )


def test_expected_calibration_error_rewards_calibration() -> None:
    good = expected_calibration_error(
        torch.tensor([0.1, 0.9]), torch.tensor([0.0, 1.0]), n_bins=10
    )
    bad = expected_calibration_error(
        torch.tensor([0.9, 0.9]), torch.tensor([0.0, 1.0]), n_bins=10
    )
    assert good < bad


def test_operating_points_recall_monotone_in_far() -> None:
    in_probs = torch.tensor([0.9, 0.8, 0.7, 0.6])
    out_probs = torch.tensor([0.4, 0.3, 0.2, 0.1])
    points = operating_points(in_probs, out_probs, far_levels=(0.25, 0.75))
    assert points[0]["recall"] <= points[1]["recall"]


def test_confidence_scorer_bounds_and_orders() -> None:
    g = torch.Generator().manual_seed(0)
    dim = 32
    centre = F.normalize(torch.randn(dim, generator=g), dim=-1)
    refs = F.normalize(centre + torch.randn(6, dim, generator=g) * 0.01, dim=-1)
    in_query = F.normalize(centre + torch.randn(dim, generator=g) * 0.01, dim=-1)
    out_query = F.normalize(torch.randn(dim, generator=g), dim=-1)

    scorer = ConfidenceScorer(PlattCalibrator(scale=10.0, bias=0.1), std_floor=1e-3)
    p_in = scorer(in_query, refs)
    p_out = scorer(out_query, refs)
    assert 0.0 <= p_in <= 1.0 and 0.0 <= p_out <= 1.0
    assert p_in > 0.5 > p_out
    assert p_in > p_out


def test_confidence_scorer_accepts_batch_queries() -> None:
    g = torch.Generator().manual_seed(1)
    dim = 16
    centre = F.normalize(torch.randn(dim, generator=g), dim=-1)
    refs = F.normalize(centre + torch.randn(4, dim, generator=g) * 0.01, dim=-1)
    queries = F.normalize(centre + torch.randn(3, dim, generator=g) * 0.01, dim=-1)

    scorer = ConfidenceScorer(PlattCalibrator(scale=10.0, bias=0.1))
    probs = scorer(queries, refs)
    assert probs.shape == (3,)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_confidence_scorer_roundtrips_calibration_file(tmp_path) -> None:
    import json

    path = tmp_path / "cal.json"
    path.write_text(json.dumps({"scale": 2.0, "bias": -0.5, "std_floor": 1e-3}))
    scorer = ConfidenceScorer.from_calibration_file(str(path))
    assert scorer.calibrator.scale == 2.0
    assert scorer.calibrator.bias == -0.5
    assert scorer.std_floor == 1e-3


def test_score_leave_one_out_flags_outlier() -> None:
    g = torch.Generator().manual_seed(2)
    dim = 32
    centre = F.normalize(torch.randn(dim, generator=g), dim=-1)
    good = F.normalize(centre + torch.randn(6, dim, generator=g) * 0.01, dim=-1)
    outlier = F.normalize(torch.randn(dim, generator=g), dim=-1)
    emb = torch.cat([good, outlier.unsqueeze(0)])

    scorer = ConfidenceScorer(PlattCalibrator(scale=10.0, bias=0.1))
    scores = scorer.score_leave_one_out(emb)
    assert scores.shape == (7,)
    assert scores[-1] < scores[:-1].min()
    assert ((scores >= 0) & (scores <= 1)).all()
