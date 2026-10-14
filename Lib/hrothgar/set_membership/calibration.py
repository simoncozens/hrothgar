"""Calibrated in-font confidence for the instance-membership model.

The raw membership score is the cosine between a query glyph's embedding and an
instance's reference prototype.  Two things stop that cosine from being a useful
``[0, 1]`` confidence on its own:

1. It is not a probability — a cosine of 0.7 says "fairly similar", not "70%
   sure".  A logistic (Platt) calibration maps it onto a probability scale.
2. Its scale varies across fonts — a tight font's in-style glyphs sit at cosine
   0.95 while a loose font's sit at 0.6, so a single global threshold would be
   wrong for one of them.  We therefore standardize each score against the
   target instance's own reference-glyph distribution (per-font normalization)
   before the global logistic is applied.

The deliverable is :class:`ConfidenceScorer`, which maps
``(query_embedding, reference_embeddings)`` to a probability in ``[0, 1]`` that
is comparable across fonts, so a per-font mean is meaningful.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from hrothgar.set_membership.config import SetMembershipConfig
from hrothgar.set_membership.evaluation import (
    MembershipEvalPanel,
    build_membership_panel,
    embed_glyphs,
)

DEFAULT_STD_FLOOR = 1e-3
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


def reference_loo_scores(references: torch.Tensor) -> torch.Tensor:
    """Leave-one-out cosine of each reference glyph against the others' prototype.

    ``references`` is ``(R, D)`` (L2-normalized).  Each glyph is scored against
    the mean of the *other* reference glyphs, so no glyph inflates its own score.
    """
    references = F.normalize(references, dim=-1)
    r = references.shape[0]
    if r < 2:
        raise ValueError("at least two reference glyphs are required")
    sums = references.sum(dim=0)
    loo = (sums.unsqueeze(0) - references) / (r - 1)
    loo = F.normalize(loo, dim=-1)
    return (references * loo).sum(dim=-1)


@dataclass
class CalibrationScores:
    """Raw cosine and per-font-normalized scores for a held-out panel."""

    in_cosine: torch.Tensor  # (P,)  query vs its own instance prototype
    out_cosine: torch.Tensor  # (N,)  query vs every other instance prototype
    in_score: torch.Tensor  # (P,)  per-font standardized
    out_score: torch.Tensor  # (N,)  per-font standardized
    reference_sigma: torch.Tensor  # (I,)  per-instance std of LOO cosines


def collect_scores(
    model, panel: MembershipEvalPanel, device, std_floor: float = DEFAULT_STD_FLOOR
) -> CalibrationScores:
    """Compute in/out cosine and per-font-standardized scores for the panel.

    Positives are query glyphs scored against their own instance's prototype.
    Negatives are the same query glyphs scored against every *other* instance's
    prototype (so same-family weights and same-category families are included
    alongside unrelated fonts).  Every score is standardized by the target
    instance's own reference distribution so tight and loose fonts become
    comparable.
    """
    ref = embed_glyphs(
        model,
        panel.reference_images.flatten(0, 1),
        panel.reference_geometry.flatten(0, 1),
        device,
    )
    n_instances = panel.reference_images.shape[0]
    n_ref = panel.reference_images.shape[1]
    ref = ref.view(n_instances, n_ref, -1)
    prototypes = F.normalize(ref.mean(dim=1), dim=-1)  # (I, D)

    mus, sigmas = [], []
    for i in range(n_instances):
        cos = reference_loo_scores(ref[i])
        mus.append(cos.mean())
        sigmas.append(cos.std())
    mus = torch.stack(mus)
    sigmas = torch.stack(sigmas)

    query = embed_glyphs(
        model,
        panel.query_images.flatten(0, 1),
        panel.query_geometry.flatten(0, 1),
        device,
    )
    n_query = panel.query_images.shape[1]
    flat_query = query.view(n_instances * n_query, -1)  # (I*Q, D)
    sim = flat_query @ prototypes.T  # (I*Q, I)

    own = (
        torch.arange(n_instances).unsqueeze(1).expand(n_instances, n_query).reshape(-1)
    )  # (I*Q,)
    rows = torch.arange(n_instances * n_query)
    off_diag = torch.ones_like(sim, dtype=torch.bool)
    off_diag[rows, own] = False

    col_mu = mus.unsqueeze(0).expand(n_instances * n_query, n_instances)
    col_sigma = sigmas.unsqueeze(0).expand(n_instances * n_query, n_instances)
    standardized = (sim - col_mu) / torch.clamp(col_sigma, min=std_floor)

    return CalibrationScores(
        in_cosine=sim[rows, own],
        out_cosine=sim[off_diag],
        in_score=standardized[rows, own],
        out_score=standardized[off_diag],
        reference_sigma=sigmas,
    )


@dataclass
class ConfidenceScorer:
    """Score a query glyph's coherence with a set of reference glyphs.

    The query and references are embeddings from :class:`SetMembershipModel`.
    The query's cosine to the reference prototype is standardized against the
    references' own leave-one-out distribution, then mapped to a probability by
    the fitted logistic.  Only the target font's references are needed at call
    time, so this works on fonts the model has never seen.
    """

    calibrator: PlattCalibrator
    std_floor: float = DEFAULT_STD_FLOOR

    def score(self, query: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
        """Return ``P(in-style)`` for ``query`` (D,) or (B,D) vs ``references`` (R,D)."""
        query = F.normalize(query, dim=-1)
        references = F.normalize(references, dim=-1)
        prototype = F.normalize(references.mean(dim=0), dim=-1)
        cosine = query @ prototype

        ref_cos = reference_loo_scores(references)
        mean = ref_cos.mean()
        std = ref_cos.std()
        standardized = (cosine - mean) / torch.clamp(std, min=self.std_floor)
        return self.calibrator.predict_proba(standardized)

    def __call__(self, query: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
        return self.score(query, references)

    @classmethod
    def from_calibration_file(cls, path: str) -> ConfidenceScorer:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            calibrator=PlattCalibrator.from_dict(data),
            std_floor=float(data.get("std_floor", DEFAULT_STD_FLOOR)),
        )


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


def _quantiles(t: torch.Tensor) -> dict:
    t = t.float()
    qs = torch.quantile(
        t, torch.tensor([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0], device=t.device)
    )
    return {
        "min": float(qs[0]),
        "q05": float(qs[1]),
        "q25": float(qs[2]),
        "median": float(qs[3]),
        "q75": float(qs[4]),
        "q95": float(qs[5]),
        "max": float(qs[6]),
        "mean": float(t.mean()),
        "std": float(t.std()),
        "n": int(t.numel()),
    }


def _calibration_summary(
    in_scores: torch.Tensor,
    out_scores: torch.Tensor,
    calibrator: PlattCalibrator,
    n_bins: int,
    far_levels: tuple[float, ...],
) -> dict:
    in_probs = calibrator.predict_proba(in_scores)
    out_probs = calibrator.predict_proba(out_scores)
    probs = torch.cat([in_probs, out_probs])
    labels = torch.cat([torch.ones_like(in_probs), torch.zeros_like(out_probs)])
    return {
        "scale": calibrator.scale,
        "bias": calibrator.bias,
        "ece": expected_calibration_error(probs, labels, n_bins),
        "operating_points": operating_points(in_probs, out_probs, far_levels),
        "reliability": reliability_curve(probs, labels, n_bins),
        "in_prob": _quantiles(in_probs),
        "out_prob": _quantiles(out_probs),
    }


def calibration_report(
    scores: CalibrationScores,
    *,
    n_bins: int = DEFAULT_N_BINS,
    far_levels: tuple[float, ...] = DEFAULT_FAR_LEVELS,
    std_floor: float = DEFAULT_STD_FLOOR,
) -> dict:
    """Full calibration report: per-font-normalized + raw-cosine baselines.

    The raw-cosine baseline is included so the benefit of per-font normalization
    is visible directly (compare the two ECEs).
    """
    normalized = PlattCalibrator.fit(scores.in_score, scores.out_score)
    raw = PlattCalibrator.fit(scores.in_cosine, scores.out_cosine)
    return {
        "std_floor": std_floor,
        "per_font_normalized": _calibration_summary(
            scores.in_score, scores.out_score, normalized, n_bins, far_levels
        ),
        "raw_cosine": _calibration_summary(
            scores.in_cosine, scores.out_cosine, raw, n_bins, far_levels
        ),
        "distributions": {
            "in_cosine": _quantiles(scores.in_cosine),
            "out_cosine": _quantiles(scores.out_cosine),
            "in_score": _quantiles(scores.in_score),
            "out_score": _quantiles(scores.out_score),
            "reference_sigma": _quantiles(scores.reference_sigma),
        },
    }


def main() -> int:
    """Fit and report calibration for an instance-membership checkpoint."""
    import torch

    from hrothgar.set_membership.dataset import SetMembershipDatasetMaker
    from hrothgar.set_membership.model import SetMembershipModel
    from hrothgar.utils import pick_device

    parser = argparse.ArgumentParser(
        description="Fit and report in-font confidence calibration."
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts"),
    )
    parser.add_argument("--model", required=True, help="path to the .pth checkpoint")
    parser.add_argument(
        "--max-instances",
        type=int,
        default=400,
        help="calibration panel size (0 = all test instances)",
    )
    parser.add_argument("--std-floor", type=float, default=DEFAULT_STD_FLOOR)
    parser.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None, help="write the report as JSON")
    parser.add_argument(
        "--save-calibrator",
        action="store_true",
        help="write scale/bias/std_floor as a sidecar next to the model",
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    config = SetMembershipConfig.from_sidecar(args.model)
    model = SetMembershipModel(config)
    model.load(args.model, device)
    model.to(device)
    model.eval()

    maker = SetMembershipDatasetMaker(args.repo, batch_size=64, config=config)
    max_instances = args.max_instances or len(maker.test_instances)
    panel = build_membership_panel(
        maker.test_instances, config, max_instances=max_instances
    )
    scores = collect_scores(model, panel, device, std_floor=args.std_floor)
    report = calibration_report(scores, n_bins=args.n_bins, std_floor=args.std_floor)

    print(f"model: {args.model}  panel: {len(panel.families)} instances")
    for scope in ("per_font_normalized", "raw_cosine"):
        c = report[scope]
        print(
            f"  {scope:20} ece={c['ece']:.4f}  scale={c['scale']:.3f} "
            f"bias={c['bias']:.3f}"
        )
        for p in c["operating_points"]:
            print(
                f"    far={p['far_target']:.2f}  threshold={p['threshold']:.3f}  "
                f"recall={p['recall']:.3f}"
            )
    print(
        "  reference_sigma      "
        f"min={report['distributions']['reference_sigma']['min']:.4f}  "
        f"median={report['distributions']['reference_sigma']['median']:.4f}  "
        f"max={report['distributions']['reference_sigma']['max']:.4f}"
    )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"wrote {args.out}")

    if args.save_calibrator:
        path = args.model.replace(".pth", ".calibration.json")
        data = report["per_font_normalized"]
        data = {
            "scale": data["scale"],
            "bias": data["bias"],
            "std_floor": args.std_floor,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
