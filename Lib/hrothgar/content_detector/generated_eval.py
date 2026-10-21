"""Score the content detector against the hand-labelled generated glyphs.

``scripts/labeling/build_critic_dataset.py`` re-frames generated glyphs into the
critic window and emits a manifest of hand labels, split into ``train`` and
``eval``.  Training consumes only the ``train`` half (see
``dataset.ContentDatasetMaker._load_generated``); this module scores the ``eval``
half, which is the hold-out.

The two labels this exists to separate are:

* **ambiguous** — "I can't even tell what this is supposed to be".  It should be
  rejected by *every* anchor, i.e. the audit's ``ambiguous`` verdict should fire.
* **bad-skeleton** — "obviously an ``e``, badly built".  It should **not** be
  ambiguous: it must still claim its anchor.  Whether the model additionally
  ranks it *below* a correct glyph is what ``quality_margin`` was meant to buy,
  so that is reported as an AUC on the claimed anchor's cosine.

Glyphs are scored *uncentred*, because that is how the training loop feeds them
(``_generated_batch`` reads the PNG as-is); the audit's font centring needs a
font's other glyphs and is a separate inference-time choice.

Usage::

    python -m hrothgar.content_detector.generated_eval --model models/content_detector.pth
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from hrothgar.content_detector.calibration import ContentCalibrator
from hrothgar.content_detector.config import ContentDetectorConfig
from hrothgar.content_detector.evaluation import _auc, _mean
from hrothgar.content_detector.model import ContentDetector
from hrothgar.utils import pick_device

# The audit's thresholds, kept in step with ``detect_content_errors``'s defaults:
# a slot is *ambiguous* when no anchor reaches ``AMBIGUOUS_BELOW``, and *wrong* when
# the claim is under ``REJECT_BELOW`` while a different anchor is over
# ``CONFIDENT_ABOVE``.
AMBIGUOUS_BELOW = 0.5
REJECT_BELOW = 0.30
CONFIDENT_ABOVE = 0.70

# The labels the review UI produces, in the order reported.
LABELS = (
    "acceptable",
    "bad-skeleton",
    "bad-execution",
    "wrong-style",
    "wrong-content",
    "ambiguous",
)


def load_manifest(manifest_path: str | Path, split: str) -> list[dict]:
    """Labelled records for ``split`` (records without a label are skipped)."""
    records = []
    with Path(manifest_path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("split") != split or not record.get("label"):
                continue
            records.append(record)
    return records


def score_records(model, records, root, device, *, batch_size: int = 32):
    """Return ``(cosine, probabilities)`` for the manifest's PNGs.

    Images are read as-is (ink on white, already in the critic frame), so a file
    that is not ``config.glyph_size`` is skipped rather than silently rescaled —
    rescaling would change the framing and invalidate the comparison.
    """
    anchors = model.normalized_anchors().detach()
    scale = model.logit_scale.detach()
    bias = model.anchor_bias.detach()
    size = model.config.glyph_size

    cosines: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    kept: list[dict] = []
    from PIL import Image

    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            chunk, kept_chunk = [], []
            for record in records[start : start + batch_size]:
                path = Path(root) / record["file"]
                image = Image.open(path).convert("L")
                if image.size != (size, size):
                    continue
                chunk.append(
                    torch.from_numpy(
                        np.asarray(image, dtype=np.float32) / 255.0
                    ).unsqueeze(0)
                )
                kept_chunk.append(record)
            if not chunk:
                continue
            embedding = model.embed(torch.stack(chunk).to(device))
            cosine = embedding @ anchors.T
            logits = scale * cosine + bias
            cosines.append(cosine.cpu())
            probabilities.append(torch.sigmoid(logits).cpu())
            kept.extend(kept_chunk)

    if not cosines:
        return kept, torch.empty(0), torch.empty(0)
    return kept, torch.cat(cosines), torch.cat(probabilities)


def _calibrated(model, calibrator, cosines: torch.Tensor) -> torch.Tensor:
    """Calibrated probabilities for already-computed (CPU) cosines."""
    scale = model.logit_scale.detach().cpu()
    bias = model.anchor_bias.detach().cpu()
    return calibrator.calibrate(scale * cosines + bias)


def _fmt(value: float | None) -> str:
    return "  -  " if value is None else f"{value:5.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--manifest",
        default=None,
        help="review manifest (default: the checkpoint's generated_manifest)",
    )
    parser.add_argument("--split", default="eval", help="manifest split to score")
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None, help="write the per-label summary as JSON")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    config = ContentDetectorConfig.from_sidecar(args.model)
    manifest = args.manifest or config.generated_manifest
    if not manifest:
        raise SystemExit(
            "no manifest given and the checkpoint's config has no generated_manifest"
        )

    model = ContentDetector(config)
    model.load(args.model, device)
    model.to(device)
    model.eval()

    try:
        calibrator = ContentCalibrator.load(args.calibration or args.model)
    except FileNotFoundError:
        print("warning: no calibration sidecar; using uncalibrated sigmoid")
        calibrator = ContentCalibrator(anchors=list(config.anchors), per_anchor={})

    records = load_manifest(manifest, args.split)
    kept, cosines, _raw = score_records(
        model, records, Path(manifest).parent, device, batch_size=args.batch_size
    )
    if not kept:
        raise SystemExit("no scorable records")
    probabilities = _calibrated(model, calibrator, cosines)

    index = config.anchor_index()
    best_prob, best_index = probabilities.max(dim=1)

    def blank_bucket() -> dict:
        return {
            "n": 0,
            "claimed_cos": [],
            "claimed_prob": [],
            "best_prob": [],
            "best_prob_anchor": [],
            "top_is_claimed": [],
            "rejected": [],
            "wrong_verdict": [],
            "rows": [],
        }

    by_label: dict[str, dict] = defaultdict(blank_bucket)
    for row, record in enumerate(kept):
        bucket = by_label[record["label"]]
        bucket["n"] += 1
        bucket["best_prob"].append(float(best_prob[row]))
        k = index.get(int(record["codepoint"]))
        if k is None:
            # No anchor for this codepoint, so no anchor *should* fire — whether
            # one does says nothing about the label.  Only anchor codepoints give
            # an apples-to-apples comparison, so every rate below uses them.
            continue
        claimed_probability = float(probabilities[row, k])
        top = int(best_index[row])
        bucket["claimed_cos"].append(float(cosines[row, k]))
        bucket["claimed_prob"].append(claimed_probability)
        bucket["best_prob_anchor"].append(float(best_prob[row]))
        bucket["top_is_claimed"].append(top == k)
        bucket["rejected"].append(float(best_prob[row]) < AMBIGUOUS_BELOW)
        bucket["wrong_verdict"].append(
            claimed_probability < REJECT_BELOW
            and top != k
            and float(probabilities[row, top]) >= CONFIDENT_ABOVE
        )
        bucket["rows"].append(
            (record, claimed_probability, config.anchors[top], float(best_prob[row]))
        )

    print(
        f"model: {args.model}\nmanifest: {manifest}  split: {args.split}\n"
        f"records scored: {len(kept)} of {len(records)} labelled\n"
        f"(claimed-anchor metrics only apply where the codepoint is an anchor)"
    )
    header = (
        f"\n{'label':<16}{'n':>6}{'n_anchor':>9}{'rejected':>10}"
        f"{'top=claimed':>12}{'wrong':>7}{'claimed cos':>12}{'claimed p':>10}"
    )
    print(header)
    summary: dict[str, dict] = {}
    for label in LABELS:
        bucket = by_label.get(label)
        if not bucket:
            continue
        n_anchor = len(bucket["claimed_cos"])
        summary[label] = {
            "n": bucket["n"],
            "n_anchor": n_anchor,
            "rejected": _mean([float(x) for x in bucket["rejected"]]),
            "top_is_claimed": _mean([float(x) for x in bucket["top_is_claimed"]]),
            "wrong_verdict": _mean([float(x) for x in bucket["wrong_verdict"]]),
            "claimed_cos_median": float(np.median(bucket["claimed_cos"]))
            if n_anchor
            else None,
            "claimed_prob_median": float(np.median(bucket["claimed_prob"]))
            if n_anchor
            else None,
            "best_prob_median": float(np.median(bucket["best_prob"])),
        }
        row_values = summary[label]
        print(
            f"{label:<16}{bucket['n']:>6}{n_anchor:>9}"
            f"{_fmt(row_values['rejected']):>10}"
            f"{_fmt(row_values['top_is_claimed']):>12}"
            f"{_fmt(row_values['wrong_verdict']):>7}"
            f"{_fmt(row_values['claimed_cos_median']):>12}"
            f"{_fmt(row_values['claimed_prob_median']):>10}"
        )

    # --- the two questions ------------------------------------------------- #
    print("\nwhat we want:")
    ambiguous = by_label.get("ambiguous")
    acceptable = by_label.get("acceptable")
    if ambiguous and acceptable:
        print(
            f"  truly ambiguous glyphs rejected by every anchor: "
            f"{_fmt(_mean([float(x) for x in ambiguous['rejected']]))} "
            f"of only {len(ambiguous['rejected'])} anchor-codepoint records; "
            f"acceptable glyphs wrongly rejected: "
            f"{_fmt(_mean([float(x) for x in acceptable['rejected']]))} "
            f"of {len(acceptable['rejected'])}"
        )
    skeleton = by_label.get("bad-skeleton")
    if skeleton and acceptable:
        print(
            f"  bad-skeleton still claims its anchor: "
            f"{_fmt(_mean([float(x) for x in skeleton['top_is_claimed']]))} "
            f"of {len(skeleton['top_is_claimed'])};  wrongly called ambiguous: "
            f"{_fmt(_mean([float(x) for x in skeleton['rejected']]))}"
        )
        auc = _auc(acceptable["claimed_cos"], skeleton["claimed_cos"])
        print(
            f"  quality separation AUC(acceptable > bad-skeleton) on the "
            f"claimed-anchor cosine: {_fmt(auc)}  "
            f"(0.5 = the model cannot tell a correct glyph from a broken one)"
        )
        if auc is not None:
            summary["quality_auc_acceptable_vs_bad_skeleton"] = auc

    # Is there an operating point at all?  An AUC near 0.5 on *every* score means
    # no threshold can separate the two, however it is chosen.
    if ambiguous and acceptable and ambiguous["best_prob_anchor"]:
        print("  can any threshold separate them?")
        for name, key in (
            ("best-anchor probability", "best_prob_anchor"),
            ("claimed-anchor cosine", "claimed_cos"),
        ):
            auc = _auc(acceptable[key], ambiguous[key])
            print(
                f"    AUC(acceptable > ambiguous) on {name:<26} {_fmt(auc)}"
            )
            if auc is not None:
                summary[f"ambiguous_auc_{key}"] = auc
        print("    what it substitutes for the ambiguous glyphs:")
        for record, claimed_p, top, best_p in sorted(
            ambiguous["rows"], key=lambda r: -r[3]
        )[:10]:
            print(
                f"      {record['family'][:20]:20} {record['codepoint_hex']:>7} "
                f"claimed p={claimed_p:.2f}  ->  {chr(top)} ({top:04X}) p={best_p:.2f}"
            )

    if args.out:
        with Path(args.out).open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
