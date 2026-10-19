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

from hrothgar.calibration import (
    DEFAULT_FAR_LEVELS,
    DEFAULT_N_BINS,
    PlattCalibrator,
    expected_calibration_error,
    operating_points,
    reliability_curve,
)
from hrothgar.set_membership.config import SetMembershipConfig
from hrothgar.set_membership.evaluation import (
    MembershipEvalPanel,
    build_membership_panel,
    embed_glyphs,
)

DEFAULT_STD_FLOOR = 1e-3


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

    def score_leave_one_out(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Score each glyph against the mean of the *other* glyphs.

        Used for whole-font QA: given all of a font instance's glyph
        embeddings ``(N, D)``, return each glyph's in-style probability with
        itself left out of its own prototype, so a single out-of-style glyph
        cannot hide behind its own contribution.  The per-font reference
        distribution is the same leave-one-out cosine array, so self-inclusion
        is a 1/N effect (negligible once a font has a handful of glyphs).
        """
        loo = reference_loo_scores(embeddings)
        mean = loo.mean()
        std = loo.std()
        standardized = (loo - mean) / torch.clamp(std, min=self.std_floor)
        return self.calibrator.predict_proba(standardized)

    @classmethod
    def from_calibration_file(cls, path: str) -> ConfidenceScorer:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            calibrator=PlattCalibrator.from_dict(data),
            std_floor=float(data.get("std_floor", DEFAULT_STD_FLOOR)),
        )


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
