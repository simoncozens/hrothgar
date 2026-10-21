"""Audit the Google Fonts library for glyphs whose *content* looks wrong.

For every font file, and every anchor codepoint that font carries, ask the content
detector whether the glyph is what it claims to be.  Three outcomes per slot:

**(1) wrong content** — a *different* anchor is closer to the glyph than the one
    it is encoded as, by at least ``--wrong-margin`` of cosine, and that
    competitor's calibrated probability is confident.  This is the QA signal, and
    it is deliberately conservative: it does **not** fire merely because a
    neighbouring anchor out-ranks the claim on an otherwise good glyph (a
    legitimate ``a`` whose ``à`` anchor is a little closer is not an error;
    Laila's ``á`` slot holding an ``à`` — cosine margin +0.7 — is).  The margin is
    a cosine, not a probability threshold, because centring lifts every claimed
    probability above any usable cut-off while leaving the ranking intact.

**(2) ambiguous** — no anchor is more likely than not: "this doesn't look like any
    letter I know".  A malformed generated glyph should fire this.

**(3) blank** — the slot renders no ink at all (Adobe Blank, or a broken glyph).
    Detected structurally, before the model.

Nonsense inputs are not run through the model at all.  Skipped by default, via
``fontquant.csv`` and the library's own tag policy:

* **all-caps / small-caps** fonts — lowercase slots hold uppercase shapes;
* **blank** fonts — every slot is empty;
* **symbol / barcode / redaction** fonts and **guide-line** fonts (tagged
  ``/Special use/*``, or ``SC``/``guide`` in the path) — not text.

Pass ``--include-excluded`` to audit everything anyway.

Scoring is **font-centred**: each glyph is compared with the anchors *relative to
the font it came from* (that font's mean embedding is subtracted first), which
removes the shared style offset that otherwise drags an unfamiliar font onto a
single anchor.  Over 123 library fonts that lifts the fraction of anchors
identified correctly from 0.79 to 0.90 and makes the median font perfect.

A font whose style the encoder has no handle on fails *wholesale*: the detector
disagrees with the claim on most of its anchors. That is reported once, as
**unrecognised — too weird to test**, rather than as a flood of confident
per-slot errors; ``--include-collapsed`` emits the slots anyway.  Note the
calibration map is fitted on *uncentred* scores, so its absolute probabilities
are only indicative here; the ``wrong`` verdict therefore gates on the cosine
margin and uses probability only for the competitor's confidence.  Use
``--sweep`` to choose the margin in this space.

Run from the repo root::

    python -m hrothgar.content_detector.calibration --model models/content_detector.pth
    python scripts/content_detector/detect_content_errors.py --model models/content_detector.pth
    # pick operating thresholds from one pass:
    python scripts/content_detector/detect_content_errors.py --model models/content_detector.pth --sweep
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

from hrothgar.content_detector.calibration import (
    DEFAULT_WRONG_MARGIN,
    ContentCalibrator,
    classify_slot,
    cosine_margin,
)
from hrothgar.content_detector.config import ContentDetectorConfig
from hrothgar.content_detector.dataset import content_frame, is_blank_render
from hrothgar.content_detector.model import ContentDetector
from hrothgar.fontquant import load_default_or_none
from hrothgar.googlefonts import GoogleFont, load_family_tags
from hrothgar.render import render
from hrothgar.utils import pick_device

# The library's font roots.  We deliberately do NOT apply ``GoogleFonts``'s
# ``having`` filter, and only its *skip* policy where it marks a font as nonsense.
FONT_ROOTS = ("ofl", "apache", "ufl")

# Tags that mark a font as not-a-text-font for our purposes.
NONSENSE_SPECIAL_TAGS = (
    "/Special use/Symbols",
    "/Special use/Barcode",
    "/Special use/Redaction",
)

# A font is "unrecognised" when the detector disagrees with the claim on most of
# its anchors: that is a style the encoder has no handle on, so auditing it slot by
# slot produces floods of confident false positives.  Measured on the library:
# ~8% of fonts fail this way.  How the losing winners distribute is reported
# alongside, because piling onto one foreign anchor is the classic collapse
# signature (`Stick` sent 6 of 11 to `E`, `Gugi` 7 to `G`) — but it is a
# diagnostic, not the gate, since centring usually spreads the winners out.
COLLAPSE_MIN_SLOTS = 4
COLLAPSE_DISAGREEMENT = 0.5

# Threshold grids for ``--sweep``.
SWEEP_MARGIN = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
SWEEP_CONFIDENT = (0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass
class SlotRecord:
    """Raw per-slot scores, kept so the ``wrong`` rule can be re-swept offline."""

    claimed_probability: float
    top_probability: float  # highest calibrated probability of any anchor
    top_is_claimed: bool
    competitor_probability: float  # p of the best *other* anchor by cosine
    cosine_excess: float  # cos(competitor) - cos(claimed); >0 means the claim lost
    blank: bool


def iter_fonts(repo: str, *, family: str | None = None, limit: int | None = None):
    """Yield every loadable ``GoogleFont`` in the library."""
    scanned = 0
    for root in FONT_ROOTS:
        for path in sorted(Path(repo).glob(f"{root}/*/*.ttf")):
            if family and family.lower() not in str(path).lower():
                continue
            try:
                font = GoogleFont(path)
            except Exception:
                continue  # no METADATA.pb, unreadable, etc.
            yield font
            scanned += 1
            if limit is not None and scanned >= limit:
                return


def is_excluded_font(font, family_tags: dict) -> str | None:
    """Why the library's policy would keep this font out of a text dataset.

    Returns a reason string, or ``None`` for an ordinary text font.
    """
    path = str(font.path)
    if "SC" in path or "guide" in path.lower():
        return "small-caps/guide"
    tags = family_tags.get(font.family, {})
    if any(tags.get(tag, 0) for tag in NONSENSE_SPECIAL_TAGS):
        return "special-use"
    return None


def _load_tags(repo: str) -> dict:
    try:
        return load_family_tags(repo)
    except OSError:
        print(
            "warning: tags/all/families.csv not found; "
            "not skipping special-use fonts",
            file=sys.stderr,
        )
        return {}


def sweep_counts(records, margin_grid, confident_grid) -> list[tuple[float, float, int]]:
    """Count ``wrong`` verdicts for each ``(wrong_margin, confident_above)`` pair.

    ``records`` is a list of ``SlotRecord``.  One pass over the library covers the
    whole grid, so the operating point can be chosen without re-scanning.
    """
    table = []
    for margin in margin_grid:
        for confident in confident_grid:
            count = sum(
                1
                for record in records
                if not record.blank
                and record.cosine_excess >= margin
                and record.competitor_probability >= confident
            )
            table.append((margin, confident, count))
    return table


def font_scores(model, images, device, *, centre: bool):
    """``(cosine, logits)`` for one font's glyphs, computed on the CPU.

    Centring subtracts the font's mean embedding before scoring, so the anchors
    are compared *relative to this font*: that is what removes the shared style
    offset which otherwise drags an unfamiliar font onto whichever anchor is
    nearest.  A single glyph has nothing to centre on, so it is a no-op there.
    """
    with torch.no_grad():
        embedding = model.embed(images.to(device))
        if centre and len(embedding) > 1:
            embedding = torch.nn.functional.normalize(
                embedding - embedding.mean(dim=0, keepdim=True), p=2, dim=-1
            )
        cosine = embedding @ model.normalized_anchors().T
        logits = model.logit_scale * cosine + model.anchor_bias
    return cosine.cpu(), logits.cpu()


def collapsed_font(cosine, present, anchors) -> dict | None:
    """Describe a font the detector cannot recognise, or ``None`` if it can.

    The gate is that most anchors are beaten by a *different* anchor — an
    ordinary font with a few odd slots is not swept up.  The concentration of the
    losing winners is reported with it, because piling onto one foreign anchor is
    the classic collapse signature even though it is not required.
    """
    if len(present) < COLLAPSE_MIN_SLOTS:
        return None

    index = {codepoint: k for k, codepoint in enumerate(anchors)}
    winners = [int(row.argmax()) for row in cosine]
    disagreements = [
        (codepoint, k) for codepoint, k in zip(present, winners) if k != index[codepoint]
    ]
    if len(disagreements) < COLLAPSE_DISAGREEMENT * len(present):
        return None

    foreign = Counter(k for _codepoint, k in disagreements)
    top_k, top_n = foreign.most_common(1)[0]
    own = sorted(float(row[index[codepoint]]) for codepoint, row in zip(present, cosine))
    return {
        "n_slots": len(present),
        "n_disagreeing": len(disagreements),
        "top_foreign": anchors[top_k],
        "top_foreign_count": top_n,
        "median_own_cosine": own[len(own) // 2],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit the library for wrong / ambiguous / blank glyph content."
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts"),
    )
    parser.add_argument("--model", required=True, help="path to the .pth checkpoint")
    parser.add_argument(
        "--calibration",
        default=None,
        help="calibration sidecar (default: <model>.calibration.json)",
    )
    parser.add_argument("--family", default=None, help="only scan paths containing this")
    parser.add_argument("--limit", type=int, default=None, help="stop after N fonts")
    parser.add_argument(
        "--ambiguous",
        type=float,
        default=0.5,
        help="flag a slot as ambiguous when its best anchor is below this "
        "calibrated probability (default 0.50)",
    )
    parser.add_argument(
        "--wrong-margin",
        type=float,
        default=DEFAULT_WRONG_MARGIN,
        help="a slot is 'wrong' when a different anchor beats the claimed one "
        "by at least this much *cosine* (default 0.35)",
    )
    parser.add_argument(
        "--confident-above",
        type=float,
        default=0.70,
        help="…and that competing anchor is at least this confident (default 0.70)",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="audit all-caps/blank/symbol/guide fonts too (off by default)",
    )
    parser.add_argument(
        "--no-centre",
        dest="centre",
        action="store_false",
        help="score against the global anchors instead of the font's own centre",
    )
    parser.add_argument(
        "--include-collapsed",
        action="store_true",
        help="emit per-slot findings for fonts the detector cannot recognise",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="print wrong-counts over a grid of (wrong_margin, confident_above)",
    )
    parser.add_argument("--max-print", type=int, default=50, help="rows per section")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None, help="write findings + slot scores as JSON")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    config = ContentDetectorConfig.from_sidecar(args.model)
    model = ContentDetector(config)
    model.load(args.model, device)
    model.to(device)
    model.eval()

    try:
        calibrator = ContentCalibrator.load(args.calibration or args.model)
    except FileNotFoundError:
        print(
            "warning: no calibration sidecar found; using uncalibrated probabilities",
            file=sys.stderr,
        )
        calibrator = ContentCalibrator(anchors=list(config.anchors), per_anchor={})

    metadata = None if args.include_excluded else load_default_or_none()
    family_tags = {} if args.include_excluded else _load_tags(args.repo)

    index = config.anchor_index()
    findings: list[dict] = []
    records: list[tuple[float, float, bool, bool]] = []
    stats: dict = {
        "fonts_scanned": 0,
        "slots_checked": 0,
        "skipped_caps": [],
        "skipped_blank": [],
        "skipped_special": [],
        "skipped_collapsed": [],
        "tallies": Counter(),
    }

    frame = content_frame(config)
    for font in iter_fonts(args.repo, family=args.family, limit=args.limit):
        stats["fonts_scanned"] += 1
        if stats["fonts_scanned"] % 500 == 0:
            print(f"  ... {stats['fonts_scanned']} fonts", file=sys.stderr)

        if metadata is not None:
            if metadata.is_blank(font.path):
                stats["skipped_blank"].append(font.family)
                continue
            if not metadata.has_lowercase(font.path):
                stats["skipped_caps"].append(font.family)
                continue
            if is_excluded_font(font, family_tags):
                stats["skipped_special"].append(font.family)
                continue

        present = [a for a in config.anchors if a in font.codepoints]
        if not present:
            continue

        images = torch.stack(
            [
                render(
                    font, codepoint=anchor, size=config.glyph_size, frame=frame
                )[0].unsqueeze(0)
                for anchor in present
            ]
        )
        cosine, logits = font_scores(model, images, device, centre=args.centre)
        probabilities = calibrator.calibrate(logits)

        collapse = collapsed_font(cosine, present, config.anchors)
        if collapse is not None and not args.include_collapsed:
            stats["skipped_collapsed"].append(
                {"family": font.family, "path": str(font.path), **collapse}
            )
            continue

        for row_index, claimed in enumerate(present):
            stats["slots_checked"] += 1
            row = probabilities[row_index]
            claimed_k = index[claimed]
            top_k = int(row.argmax())
            competitor_k, cosine_excess = cosine_margin(cosine[row_index], claimed_k)
            blank = is_blank_render(images[row_index])

            if blank:
                label, competitor = "blank", None
            else:
                label, competitor = classify_slot(
                    row,
                    claimed_k,
                    config.anchors,
                    cosine=cosine[row_index],
                    ambiguous=args.ambiguous,
                    confident_above=args.confident_above,
                    wrong_margin=args.wrong_margin,
                )
            records.append(
                SlotRecord(
                    claimed_probability=float(row[claimed_k]),
                    top_probability=float(row[top_k]),
                    top_is_claimed=top_k == claimed_k,
                    competitor_probability=float(row[competitor_k]),
                    cosine_excess=cosine_excess,
                    blank=blank,
                )
            )
            if blank or label != "ok":
                stats["tallies"][f"{label}:{chr(claimed)}"] += 1
                # For a wrong verdict the anchor worth reporting is the one the
                # model thinks the glyph *is* — the cosine winner — which need not
                # be the highest-probability anchor, since each anchor has its own
                # calibration map.
                reported_k = (
                    index[competitor]
                    if label == "wrong" and competitor is not None
                    else top_k
                )
                findings.append(
                    {
                        "label": label,
                        "family": font.family,
                        "path": str(font.path),
                        "claimed": claimed,
                        "claimed_probability": float(row[claimed_k]),
                        "top": config.anchors[reported_k],
                        "top_probability": float(row[reported_k]),
                        "cosine_excess": cosine_excess,
                    }
                )

    _print_findings(findings, args, stats)
    if args.sweep:
        _print_sweep(records, args)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": args.model,
                    "fonts_scanned": stats["fonts_scanned"],
                    "slots_checked": stats["slots_checked"],
                    "skipped_caps": stats["skipped_caps"],
                    "skipped_blank": stats["skipped_blank"],
                    "skipped_special": stats["skipped_special"],
                    "skipped_collapsed": stats["skipped_collapsed"],
                    "centred": args.centre,
                    "ambiguous_threshold": args.ambiguous,
                    "wrong_margin": args.wrong_margin,
                    "confident_above": args.confident_above,
                    "findings": findings,
                    # Raw per-slot scores, so the rule can be re-swept offline.
                    "slots": [
                        {
                            "claimed_probability": record.claimed_probability,
                            "top_probability": record.top_probability,
                            "top_is_claimed": record.top_is_claimed,
                            "competitor_probability": record.competitor_probability,
                            "cosine_excess": record.cosine_excess,
                            "blank": record.blank,
                        }
                        for record in records
                    ],
                },
                f,
                indent=2,
                sort_keys=True,
            )
            f.write("\n")
        print(f"wrote {args.out}")

    return 0


def _print_findings(findings, args, stats: dict) -> None:
    wrong = [f for f in findings if f["label"] == "wrong"]
    ambiguous = [f for f in findings if f["label"] == "ambiguous"]
    blank = [f for f in findings if f["label"] == "blank"]

    for key, label in (
        ("skipped_caps", "all-caps/small-caps"),
        ("skipped_blank", "blank"),
        ("skipped_special", "symbol/barcode/guide"),
    ):
        skipped = stats.get(key) or []
        if skipped:
            examples = ", ".join(sorted(set(skipped))[:4])
            print(f"(skipped {len(skipped)} {label} fonts; e.g. {examples})")

    collapsed_fonts = stats.get("skipped_collapsed") or []
    if collapsed_fonts:
        print(
            f"(skipped {len(collapsed_fonts)} unrecognised fonts: the detector "
            f"disagrees with most of their slots, so per-slot verdicts say nothing)"
        )

    print(
        f"Scanned {stats['fonts_scanned']} fonts, {stats['slots_checked']} anchor slots.\n"
        f"  wrong content : {len(wrong)}\n"
        f"  ambiguous     : {len(ambiguous)}\n"
        f"  blank         : {len(blank)}\n"
        f"  unrecognised  : {len(collapsed_fonts)} fonts"
    )

    if collapsed_fonts:
        print(
            "\nunrecognised fonts (too weird to test — the detector disagrees with "
            "most of their claims):"
        )
        for entry in sorted(
            collapsed_fonts, key=lambda e: -e["n_disagreeing"]
        )[: args.max_print]:
            print(
                f"  {entry['family'][:24]:24} "
                f"{entry['n_disagreeing']}/{entry['n_slots']} slots disagree, "
                f"median own cos {entry['median_own_cosine']:+.2f}, "
                f"mostly -> {chr(entry['top_foreign'])} "
                f"({entry['top_foreign_count']}x)  {entry['path']}"
            )
        if len(collapsed_fonts) > args.max_print:
            print(f"  ... {len(collapsed_fonts) - args.max_print} more")

    print(
        f"\nwrong content (competitor's cosine lead >= {args.wrong_margin:.2f}, "
        f"competitor p >= {args.confident_above:.2f}):"
    )
    for finding in sorted(wrong, key=lambda f: -f["cosine_excess"])[: args.max_print]:
        print(
            f"  {finding['family'][:24]:24} "
            f"{chr(finding['claimed'])} (U+{finding['claimed']:04X}) looks like "
            f"{chr(finding['top'])} (U+{finding['top']:04X}) "
            f"lead={finding['cosine_excess']:+.2f} "
            f"p={finding['top_probability']:.2f} "
            f"(claimed {finding['claimed_probability']:.2f})  {finding['path']}"
        )
    if len(wrong) > args.max_print:
        print(f"  ... {len(wrong) - args.max_print} more")

    print(f"\nambiguous (best anchor < {args.ambiguous:.2f}):")
    for finding in sorted(ambiguous, key=lambda f: f["top_probability"])[
        : args.max_print
    ]:
        print(
            f"  {finding['family'][:24]:24} "
            f"{chr(finding['claimed'])} (U+{finding['claimed']:04X}) "
            f"max p={finding['top_probability']:.2f} "
            f"(best {chr(finding['top'])} U+{finding['top']:04X})  "
            f"{finding['path']}"
        )
    if len(ambiguous) > args.max_print:
        print(f"  ... {len(ambiguous) - args.max_print} more")

    if blank:
        print("\nblank slots (no ink — structural, not a model verdict):")
        for finding in blank[: args.max_print]:
            print(
                f"  {finding['family'][:24]:24} "
                f"{chr(finding['claimed'])} (U+{finding['claimed']:04X})  "
                f"{finding['path']}"
            )
        if len(blank) > args.max_print:
            print(f"  ... {len(blank) - args.max_print} more")

    tallies: Counter = stats["tallies"]
    if tallies:
        print("\nper anchor (wrong / ambiguous / blank):")
        for anchor in sorted({key.split(":")[1] for key in tallies}):
            print(
                f"  {anchor}: wrong={tallies.get(f'wrong:{anchor}', 0)} "
                f"ambiguous={tallies.get(f'ambiguous:{anchor}', 0)} "
                f"blank={tallies.get(f'blank:{anchor}', 0)}"
            )


def _print_sweep(records, args) -> None:
    table = sweep_counts(records, SWEEP_MARGIN, SWEEP_CONFIDENT)
    print(
        f"\nwrong-count sweep ({len(records)} slots; rows wrong_margin, "
        f"cols confident_above):"
    )
    print("    margin\\conf " + "".join(f"{c:>8.2f}" for c in SWEEP_CONFIDENT))
    for margin in SWEEP_MARGIN:
        cells = "".join(
            f"{count:>8d}" for r, c, count in table if r == margin
        )
        print(f"    {margin:>11.2f} " + cells)


if __name__ == "__main__":
    raise SystemExit(main())
