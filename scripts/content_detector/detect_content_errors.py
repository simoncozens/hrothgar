"""Audit the Google Fonts library for glyphs whose *content* looks wrong.

For every font file, and every anchor codepoint that font carries, ask the content
detector whether the glyph is what it claims to be.  Three outcomes per slot:

**(1) wrong content** — the model *rejects the claim* (the claimed anchor's
    calibrated probability is low) and a *different* anchor is confident.  This is
    the QA signal, and it is deliberately conservative: it does **not** fire merely
    because a neighbouring anchor out-ranks the claim on an otherwise good glyph
    (a legitimate ``a`` scoring ``p(a)=0.89, p(à)=0.99`` is not an error; Laila's
    ``á`` slot holding an ``à`` — ``claimed≈0.0, competitor≈1.0`` — is).

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
from pathlib import Path

import torch

from hrothgar.content_detector.calibration import ContentCalibrator, classify_slot
from hrothgar.content_detector.config import ContentDetectorConfig
from hrothgar.content_detector.dataset import content_frame, is_blank_render
from hrothgar.content_detector.model import ContentDetector
from hrothgar.fontquant import load_default_or_none
from hrothgar.googlefonts import GoogleFont, load_family_tags
from hrothgar.render import render_glyph_uniform
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

# Threshold grids for ``--sweep``.
SWEEP_REJECT = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
SWEEP_CONFIDENT = (0.5, 0.6, 0.7, 0.8, 0.9)


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


def sweep_counts(records, reject_grid, confident_grid) -> list[tuple[float, float, int]]:
    """Count ``wrong`` verdicts for each ``(reject_below, confident_above)`` pair.

    ``records`` is a list of ``(claimed_probability, top_probability,
    top_is_claimed, is_blank)``.  One pass over the library covers the whole grid,
    so the operating point can be chosen without re-scanning.
    """
    table = []
    for reject in reject_grid:
        for confident in confident_grid:
            count = sum(
                1
                for claimed, top, same, blank in records
                if not blank and not same and claimed < reject and top >= confident
            )
            table.append((reject, confident, count))
    return table


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
        "--reject-below",
        type=float,
        default=0.30,
        help="a slot is 'wrong' only when the claimed anchor is below this "
        "calibrated probability (default 0.30)",
    )
    parser.add_argument(
        "--confident-above",
        type=float,
        default=0.70,
        help="…and a different anchor is at least this confident (default 0.70)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="…and beats the claimed anchor by this much (default 0.00)",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="audit all-caps/blank/symbol/guide fonts too (off by default)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="print wrong-counts over a grid of (reject_below, confident_above)",
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
        "tallies": Counter(),
    }

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

        frame = content_frame(config)
        images = torch.stack(
            [
                render_glyph_uniform(
                    font, anchor, config.glyph_size, frame=frame
                )[0].unsqueeze(0)
                for anchor in present
            ]
        )
        with torch.no_grad():
            logits = model(images.to(device)).cpu()
        probabilities = calibrator.calibrate(logits)

        for row_index, claimed in enumerate(present):
            stats["slots_checked"] += 1
            row = probabilities[row_index]
            claimed_k = index[claimed]
            top_k = int(row.argmax())
            blank = is_blank_render(images[row_index])

            if blank:
                label = "blank"
            else:
                label, _ = classify_slot(
                    row,
                    claimed_k,
                    config.anchors,
                    ambiguous=args.ambiguous,
                    reject_below=args.reject_below,
                    confident_above=args.confident_above,
                    margin=args.margin,
                )
            records.append(
                (float(row[claimed_k]), float(row[top_k]), top_k == claimed_k, blank)
            )
            if blank or label != "ok":
                stats["tallies"][f"{label}:{chr(claimed)}"] += 1
                findings.append(
                    {
                        "label": label,
                        "family": font.family,
                        "path": str(font.path),
                        "claimed": claimed,
                        "claimed_probability": float(row[claimed_k]),
                        "top": config.anchors[top_k],
                        "top_probability": float(row[top_k]),
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
                    "ambiguous_threshold": args.ambiguous,
                    "reject_below": args.reject_below,
                    "confident_above": args.confident_above,
                    "margin": args.margin,
                    "findings": findings,
                    # Raw per-slot scores, so the rule can be re-swept offline.
                    "slots": [
                        {
                            "claimed_probability": claimed,
                            "top_probability": top,
                            "top_is_claimed": same,
                            "blank": blank,
                        }
                        for claimed, top, same, blank in records
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

    print(
        f"Scanned {stats['fonts_scanned']} fonts, {stats['slots_checked']} anchor slots.\n"
        f"  wrong content : {len(wrong)}\n"
        f"  ambiguous     : {len(ambiguous)}\n"
        f"  blank         : {len(blank)}"
    )

    print(
        f"\nwrong content (claimed < {args.reject_below:.2f}, competitor "
        f">= {args.confident_above:.2f}, margin {args.margin:.2f}):"
    )
    for finding in sorted(wrong, key=lambda f: -f["top_probability"])[: args.max_print]:
        print(
            f"  {finding['family'][:24]:24} "
            f"{chr(finding['claimed'])} (U+{finding['claimed']:04X}) looks like "
            f"{chr(finding['top'])} (U+{finding['top']:04X}) "
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
    table = sweep_counts(records, SWEEP_REJECT, SWEEP_CONFIDENT)
    print(
        f"\nwrong-count sweep ({len(records)} slots; rows reject_below, "
        f"cols confident_above):"
    )
    print("    reject\\conf " + "".join(f"{c:>8.2f}" for c in SWEEP_CONFIDENT))
    for reject in SWEEP_REJECT:
        cells = "".join(
            f"{count:>8d}" for r, c, count in table if r == reject
        )
        print(f"    {reject:>11.2f} " + cells)


if __name__ == "__main__":
    raise SystemExit(main())
