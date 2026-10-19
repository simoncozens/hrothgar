"""Shared calibration primitives.

Model-agnostic helpers for turning a raw score into a probability and reporting
how well calibrated it is: a logistic (Platt) map, expected calibration error,
reliability curves and recall-at-FAR operating points.  Used by the
set-membership model and the content detector.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

DEFAULT_FAR_LEVELS = (0.01, 0.05, 0.10, 0.20)
DEFAULT_N_BINS = 10


@dataclass
class PlattCalibrator:
    """Logistic map ``sigmoid(scale * score + bias)`` onto a probability.

    ``scale`` and ``bias`` are plain floats, so the calibrator can be serialized
    as JSON and applied with pure torch at runtime (no scikit-learn needed to
    *evaluate*).
    """

    scale: float
    bias: float

    def predict_proba(self, scores: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.scale * scores + self.bias)

    @classmethod
    def fit(cls, in_scores: torch.Tensor, out_scores: torch.Tensor) -> PlattCalibrator:
        """Fit a logistic on positive/negative scores.

        ``class_weight="balanced"`` reweights the loss to a 50/50 prior, so the
        returned probability reads "P(in-style | score) under equal base rates"
        regardless of how imbalanced the supplied scores are.
        """
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        x = np.concatenate([in_scores, out_scores]).reshape(-1, 1).astype(np.float64)
        y = np.concatenate([np.ones(in_scores.numel()), np.zeros(out_scores.numel())])
        clf = LogisticRegression(
            class_weight="balanced", solver="lbfgs", max_iter=1000
        ).fit(x, y)
        return cls(scale=float(clf.coef_[0, 0]), bias=float(clf.intercept_[0]))

    def to_dict(self) -> dict:
        return {"scale": self.scale, "bias": self.bias}

    @classmethod
    def from_dict(cls, data: dict) -> PlattCalibrator:
        return cls(scale=float(data["scale"]), bias=float(data["bias"]))


def expected_calibration_error(
    probs: torch.Tensor, labels: torch.Tensor, n_bins: int = DEFAULT_N_BINS
) -> float:
    """Expected calibration error (ECE) over equal-width confidence bins."""
    probs = probs.float().clamp(0, 1)
    labels = labels.float()
    if probs.numel() == 0:
        return 0.0
    edges = torch.linspace(0, 1, n_bins + 1)
    total = probs.numel()
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (probs >= lo) & (probs < hi)
        if i == n_bins - 1:
            in_bin = (probs >= lo) & (probs <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        ece += (count / total) * abs(
            (probs[in_bin].mean() - labels[in_bin].mean()).item()
        )
    return float(ece)


def reliability_curve(
    probs: torch.Tensor, labels: torch.Tensor, n_bins: int = DEFAULT_N_BINS
) -> dict:
    """Per-bin confidence vs accuracy for a reliability diagram."""
    probs = probs.float().clamp(0, 1)
    labels = labels.float()
    edges = torch.linspace(0, 1, n_bins + 1)
    confidences, accuracies, counts = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (probs >= lo) & (probs < hi)
        if i == n_bins - 1:
            in_bin = (probs >= lo) & (probs <= hi)
        count = int(in_bin.sum())
        counts.append(count)
        confidences.append(float(probs[in_bin].mean()) if count else 0.0)
        accuracies.append(float(labels[in_bin].mean()) if count else 0.0)
    return {
        "edges": [float(e) for e in edges],
        "confidence": confidences,
        "accuracy": accuracies,
        "counts": counts,
    }


def operating_points(
    in_probs: torch.Tensor,
    out_probs: torch.Tensor,
    far_levels: tuple[float, ...] = DEFAULT_FAR_LEVELS,
) -> list[dict]:
    """Recall (true-accept) at selected false-accept rates."""
    out_probs = out_probs.float()
    points = []
    for far in far_levels:
        threshold = torch.quantile(out_probs, 1.0 - far).item()
        points.append(
            {
                "far_target": far,
                "threshold": threshold,
                "recall": float((in_probs > threshold).float().mean()),
                "far_actual": float((out_probs > threshold).float().mean()),
            }
        )
    return points
