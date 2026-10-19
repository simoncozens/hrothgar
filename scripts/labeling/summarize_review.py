#!/usr/bin/env python
"""Summarise the human labels written by the review rig.

Reads ``review_state.json`` (``"<family>/U+XXXX" -> category``) plus the rig's
``review_manifest.json`` and prints:

* coverage (how many glyphs / families are labelled) and the sampled family
  range (an alphabetical prefix is *not* a random sample -- the script says so),
* the overall category distribution, with ``1 in N`` rarity,
* the same broken down by glyph class (uppercase/lowercase/digit/symbol),
  style (normal/italic) and font stratum (sans/serif/fancy),
* a per-codepoint table (which characters are systematically hard, and which
  category they fail into),
* per-family defect-rate distribution and the hardest families,
* triage agreement on the labelled subset (precision of the auto-accept, and
  what the human called the triage's ``review`` bucket),
* the mean triage score for each human label (useful for setting thresholds),
* the rare-category examples (e.g. every ``wrong-content`` glyph).

The output is meant to answer "how common is each failure mode, and where does
it concentrate?" so critic models can be weighted sensibly.

Usage::

    python3 scripts/labeling/summarize_review.py
    python3 scripts/labeling/summarize_review.py --top 30 --examples 50
    python3 scripts/labeling/summarize_review.py --json > review_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

CATEGORIES = [
    "acceptable",
    "ambiguous",
    "bad-skeleton",
    "wrong-style",
    "bad-execution",
    "wrong-content",
]
NON_ACCEPTABLE = CATEGORIES[1:]


# ── Loading ───────────────────────────────────────────────────────────────


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_triage(path: Path | None) -> dict[str, dict]:
    """``file -> triage record`` from the triage script's JSONL manifest."""
    records: dict[str, dict] = {}
    if path is None or not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            rel = record.get("file")
            if rel:
                records[rel] = record
    return records


def sanitize(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")
    return cleaned or "family"


def instance_keys(path: Path) -> dict[str, dict]:
    """``family key -> instance`` for stratum lookup (path/style/weight)."""
    if not path.exists():
        return {}
    mapping: dict[str, dict] = {}
    for inst in load_json(path):
        style_suffix = "" if inst.get("style", "normal") == "normal" else "-italic"
        mapping[sanitize(inst["family"]) + style_suffix] = inst
    return mapping


def glyph_class(cp: int) -> str:
    if 0x41 <= cp <= 0x5A:
        return "uppercase"
    if 0x61 <= cp <= 0x7A:
        return "lowercase"
    if 0x30 <= cp <= 0x39:
        return "digit"
    return "symbol"


def stratum_of(metadata_pb: Path) -> str:
    """Coarse sans/serif/fancy/mono bucket, mirroring ``dataset.style_bucket``."""
    try:
        from gftools.util.google_fonts import Metadata
    except Exception:
        return "unknown"
    try:
        metadata = Metadata(str(metadata_pb))
        cls = metadata.classifications or metadata.category
    except Exception:
        return "unknown"
    if isinstance(cls, str):
        text = cls
    elif cls is None:
        return "unknown"
    else:
        text = "/".join(str(x) for x in cls)
    if "MONOSPACE" in text:
        return "mono"
    if "HANDWRITING" in text or "DISPLAY" in text:
        return "fancy"
    if "SANS_SERIF" in text:
        return "sans"
    if "SERIF" in text:
        return "serif"
    return "other"


# ── Build the row set ─────────────────────────────────────────────────────


def build_rows(state, manifest, triage):
    info = {}
    for family in manifest["families"]:
        for glyph in family["glyphs"]:
            key = f"{family['key']}/U+{glyph['cp']:04X}"
            info[key] = (family, glyph)

    rows = []
    orphans = []
    for key, label in state.items():
        if key not in info:
            orphans.append(key)
            continue
        family, glyph = info[key]
        record = triage.get(glyph["file"], {})
        rows.append(
            {
                "key": key,
                "family": family["key"],
                "family_name": family["family"],
                "style": family["style"],
                "weight": family["weight"],
                "cp": glyph["cp"],
                "char": chr(glyph["cp"]),
                "has_gt": glyph["has_gt"],
                "label": label,
                "triage": record.get("decision"),
                "iou": record.get("iou"),
                "hausdorff": record.get("hausdorff"),
            }
        )
    return rows, orphans


def attach_strata(rows, instances, repo: Path) -> None:
    cache: dict[str, str] = {}
    for row in rows:
        family = row["family"]
        if family not in cache:
            inst = instances.get(family)
            if inst is None:
                cache[family] = "unknown"
            else:
                metadata_pb = (repo / inst["path"]).parent / "METADATA.pb"
                cache[family] = stratum_of(metadata_pb)
        row["stratum"] = cache[family]


# ── Formatting helpers ────────────────────────────────────────────────────


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def bar(fraction: float, width: int = 32) -> str:
    return "#" * round(width * fraction)


def distribution(counter: Counter, total: int) -> list[tuple[str, int, float]]:
    return [(cat, counter[cat], pct(counter[cat], total)) for cat in CATEGORIES]


def one_in(n: int, total: int) -> str:
    if n == 0:
        return "—"
    return f"1 in {total / n:.0f}"


def grouped_distribution(rows, key_name):
    groups: dict[str, list] = defaultdict(list)
    for row in rows:
        groups[row[key_name]].append(row)
    return {
        key: (len(sub), Counter(r["label"] for r in sub))
        for key, sub in sorted(groups.items())
    }


# ── Report ────────────────────────────────────────────────────────────────


def build_summary(rows, orphans, args) -> dict:
    total = len(rows)
    families = sorted({r["family"] for r in rows})
    for row in rows:
        row["_class"] = glyph_class(row["cp"])
    overall = Counter(r["label"] for r in rows)

    summary: dict = {
        "coverage": {
            "labelled": total,
            "families": len(families),
            "family_first": families[0] if families else None,
            "family_last": families[-1] if families else None,
            "orphan_labels": len(orphans),
        },
        "overall": {
            cat: {"count": overall[cat], "pct": round(pct(overall[cat], total), 2)}
            for cat in CATEGORIES
        },
        "by_class": {},
        "by_style": {},
        "by_stratum": {},
        "by_codepoint": [],
        "by_family": [],
        "triage": {},
        "rare_examples": {},
    }

    for cls, (n, counter) in grouped_distribution(rows, "_class").items():
        summary["by_class"][cls] = {
            "n": n,
            **{cat: {"count": counter[cat], "pct": round(pct(counter[cat], n), 2)} for cat in CATEGORIES},
        }
    for style, (n, counter) in grouped_distribution(rows, "style").items():
        summary["by_style"][style] = {
            "n": n,
            **{cat: {"count": counter[cat], "pct": round(pct(counter[cat], n), 2)} for cat in CATEGORIES},
        }
    if any("stratum" in r for r in rows):
        for stratum, (n, counter) in grouped_distribution(rows, "stratum").items():
            summary["by_stratum"][stratum] = {
                "n": n,
                **{cat: {"count": counter[cat], "pct": round(pct(counter[cat], n), 2)} for cat in CATEGORIES},
            }

    # Per-codepoint.
    by_cp: dict[int, list] = defaultdict(list)
    for row in rows:
        by_cp[row["cp"]].append(row)
    cp_stats = []
    for cp, sub in by_cp.items():
        counter = Counter(r["label"] for r in sub)
        n = len(sub)
        nonacc = sum(counter[c] for c in NON_ACCEPTABLE)
        cp_stats.append(
            {
                "cp": cp,
                "char": chr(cp),
                "n": n,
                "class": glyph_class(cp),
                "counts": {c: counter[c] for c in CATEGORIES},
                "nonacceptable_pct": round(pct(nonacc, n), 2),
            }
        )
    cp_stats.sort(key=lambda x: x["nonacceptable_pct"], reverse=True)
    summary["by_codepoint"] = cp_stats

    # Per-family.
    by_family: dict[str, list] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    fam_stats = []
    for family, sub in by_family.items():
        counter = Counter(r["label"] for r in sub)
        n = len(sub)
        nonacc = sum(counter[c] for c in NON_ACCEPTABLE)
        fam_stats.append(
            {
                "family": family,
                "stratum": sub[0].get("stratum"),
                "style": sub[0]["style"],
                "n": n,
                "counts": {c: counter[c] for c in CATEGORIES},
                "defect_pct": round(pct(nonacc, n), 2),
            }
        )
    fam_stats.sort(key=lambda x: x["defect_pct"], reverse=True)
    summary["by_family"] = fam_stats

    # Triage agreement.
    triaged = [r for r in rows if r["triage"] in ("acceptable", "review", "no-gt")]
    if triaged:
        triage_summary = {}
        for decision in ("acceptable", "review", "no-gt"):
            sub = [r for r in triaged if r["triage"] == decision]
            if not sub:
                continue
            counter = Counter(r["label"] for r in sub)
            triage_summary[decision] = {
                "n": len(sub),
                "labels": {
                    c: {"count": counter[c], "pct": round(pct(counter[c], len(sub)), 2)}
                    for c in CATEGORIES
                },
            }
        accepted = [r for r in triaged if r["triage"] == "acceptable"]
        tp = sum(1 for r in accepted if r["label"] == "acceptable")
        triage_summary["accept_precision"] = {
            "n": len(accepted),
            "true_positive": tp,
            "pct": round(pct(tp, len(accepted)), 2) if accepted else None,
        }
        summary["triage"] = triage_summary

    # Mean triage score per human label.
    score_stats = {}
    for cat in CATEGORIES:
        sub = [
            r
            for r in rows
            if r["label"] == cat
            and r["iou"] is not None
            and r["hausdorff"] is not None
        ]
        if not sub:
            continue
        score_stats[cat] = {
            "n": len(sub),
            "mean_iou": round(statistics.mean(r["iou"] for r in sub), 4),
            "mean_hausdorff": round(
                statistics.mean(r["hausdorff"] for r in sub), 2
            ),
        }
    summary["scores_by_label"] = score_stats

    # Rare examples.
    for cat in NON_ACCEPTABLE:
        examples = [r for r in rows if r["label"] == cat]
        summary["rare_examples"][cat] = [
            {
                "family": r["family"],
                "char": r["char"],
                "cp": f"U+{r['cp']:04X}",
                "iou": r["iou"],
                "hausdorff": r["hausdorff"],
            }
            for r in examples[: args.examples]
        ]
        summary["rare_examples"][cat + "_count"] = len(examples)

    return summary


def print_report(rows, orphans, args, summary) -> None:
    total = len(rows)
    overall = Counter(r["label"] for r in rows)
    families = sorted({r["family"] for r in rows})

    print("=" * 78)
    print(f"REVIEW SUMMARY — {total} glyphs, {len(families)} families")
    if families:
        print(
            f"Sampled families: {families[0]} … {families[-1]} "
            f"(alphabetical prefix, not a random sample)"
        )
    if orphans:
        print(f"⚠ {len(orphans)} state labels had no manifest entry (ignored)")

    print("\nOVERALL CATEGORY DISTRIBUTION")
    for cat, count, percentage in distribution(overall, total):
        rarity = one_in(count, total) if cat != "acceptable" else ""
        print(
            f"  {cat:14s} {count:6d}  {percentage:5.1f}%  "
            f"{bar(count / total if total else 0):32s}  {rarity}"
        )
    nonacc = sum(overall[c] for c in NON_ACCEPTABLE)
    print(
        f"  {'non-acceptable':14s} {nonacc:6d}  {pct(nonacc, total):5.1f}%"
    )

    print("\nBY GLYPH CLASS")
    header = f"{'class':10s} {'n':>6s} " + " ".join(f"{c[:10]:>10s}" for c in CATEGORIES)
    print(header)
    for cls, (n, counter) in grouped_distribution(rows, "_class").items():
        print(
            f"{cls:10s} {n:6d} "
            + " ".join(f"{pct(counter[c], n):9.1f}%" for c in CATEGORIES)
        )

    print("\nBY STYLE")
    for style, (n, counter) in grouped_distribution(rows, "style").items():
        print(
            f"  {style:8s} n={n:5d}  "
            + "  ".join(f"{c}={pct(counter[c], n):.1f}%" for c in CATEGORIES)
        )

    if any("stratum" in r for r in rows):
        print("\nBY STRATUM")
        for stratum, (n, counter) in grouped_distribution(rows, "stratum").items():
            print(
                f"  {stratum:8s} n={n:5d}  "
                + "  ".join(f"{c}={pct(counter[c], n):.1f}%" for c in CATEGORIES)
            )

    print(
        f"\nPER-CODEPOINT — top {args.top} by non-acceptable rate "
        f"(n>={args.min_glyphs})"
    )
    print(
        f"{'char':>4s} {'cp':>8s} {'n':>4s} {'nonacc':>7s}  "
        + " ".join(f"{c[:8]:>9s}" for c in NON_ACCEPTABLE)
    )
    shown = 0
    for stat in summary["by_codepoint"]:
        if stat["n"] < args.min_glyphs:
            continue
        if shown >= args.top:
            break
        shown += 1
        counts = stat["counts"]
        print(
            f"{stat['char']:>4s} {('U+%04X' % stat['cp']):>8s} "
            f"{stat['n']:4d} {stat['nonacceptable_pct']:6.1f}%  "
            + " ".join(
                f"{pct(counts[c], stat['n']):8.1f}%" for c in NON_ACCEPTABLE
            )
        )

    print("\nMOST-AFFECTED CODEPOINTS PER CATEGORY (rate, n>={})".format(args.min_glyphs))
    for cat in NON_ACCEPTABLE:
        ranked = [
            (s["counts"][cat] / s["n"], s["char"], s["cp"], s["n"], s["counts"][cat])
            for s in summary["by_codepoint"]
            if s["n"] >= args.min_glyphs
        ]
        ranked.sort(reverse=True)
        top = ", ".join(
            f"{char!r} ({rate * 100:.0f}%, {count}/{n})"
            for rate, char, cp, n, count in ranked[:5]
            if count
        )
        print(f"  {cat:14s} {top or '—'}")

    print("\nPER-FAMILY DEFECT RATE")
    rates = [f["defect_pct"] for f in summary["by_family"]]
    if rates:
        rates_sorted = sorted(rates)
        def q(p):
            return rates_sorted[min(len(rates_sorted) - 1, int(p * len(rates_sorted)))]
        print(
            f"  families={len(rates)}  min={min(rates):.1f}%  "
            f"p25={q(0.25):.1f}%  median={statistics.median(rates):.1f}%  "
            f"p75={q(0.75):.1f}%  max={max(rates):.1f}%  "
            f"mean={statistics.mean(rates):.1f}%"
        )
        print(f"  hardest (n>={args.min_glyphs}):")
        shown = 0
        for fam in summary["by_family"]:
            if fam["n"] < args.min_glyphs:
                continue
            if shown >= args.top_families:
                break
            shown += 1
            counts = fam["counts"]
            detail = " ".join(
                f"{c[:4]}={counts[c]}" for c in NON_ACCEPTABLE if counts[c]
            )
            print(
                f"    {fam['defect_pct']:5.1f}%  {fam['family']:32s} "
                f"n={fam['n']:3d}  {detail}"
            )

    if summary["triage"]:
        print("\nTRIAGE AGREEMENT (on labelled glyphs)")
        for decision in ("acceptable", "review", "no-gt"):
            info = summary["triage"].get(decision)
            if not info:
                continue
            labels = info["labels"]
            print(
                f"  triage={decision:10s} n={info['n']:5d}  "
                + "  ".join(f"{c}={labels[c]['pct']:.1f}%" for c in CATEGORIES)
            )
        precision = summary["triage"]["accept_precision"]
        if precision["pct"] is not None:
            print(
                f"  auto-accept precision: {precision['true_positive']}/"
                f"{precision['n']} = {precision['pct']:.1f}%"
            )

    if summary["scores_by_label"]:
        print("\nMEAN TRIAGE SCORE BY HUMAN LABEL")
        for cat, stats in summary["scores_by_label"].items():
            print(
                f"  {cat:14s} n={stats['n']:5d}  "
                f"mean iou={stats['mean_iou']:.3f}  "
                f"mean haus={stats['mean_hausdorff']:.2f}"
            )

    print(
        f"\nRARE-CATEGORY EXAMPLES "
        f"(categories with <= {args.example_threshold} labels; "
        f"up to {args.examples} each)"
    )
    printed = False
    for cat in NON_ACCEPTABLE:
        count = summary["rare_examples"][cat + "_count"]
        if count > args.example_threshold:
            continue
        printed = True
        examples = summary["rare_examples"][cat]
        print(f"  {cat} ({count}):")
        for example in examples:
            print(
                f"    {example['family']:32s} {example['char']!r} "
                f"{example['cp']}  iou={example['iou']} haus={example['hausdorff']}"
            )
        if not examples:
            print("    —")
    if not printed:
        print("  (none below the threshold; raise --example-threshold)")



def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--state",
        type=Path,
        default=Path("outputs/generated/review_state.json"),
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/generated/review_manifest.json"),
    )
    p.add_argument(
        "--triage-manifest",
        type=Path,
        default=Path("outputs/labels/manifest.jsonl"),
        help="Triage JSONL; omit/rename to skip triage comparisons",
    )
    p.add_argument(
        "--instances",
        type=Path,
        default=Path("models/fontid_diffusion.pth.instances.json"),
        help="Instance sidecar, for the family stratum lookup",
    )
    p.add_argument(
        "--repo",
        type=Path,
        default=Path(
            os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts")
        ),
        help="Google Fonts checkout, for METADATA-based strata",
    )
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--top-families", type=int, default=15)
    p.add_argument("--min-glyphs", type=int, default=10)
    p.add_argument("--examples", type=int, default=20)
    p.add_argument(
        "--example-threshold",
        type=int,
        default=400,
        help="Only list per-category examples for categories with at most "
        "this many labels",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit the machine-readable summary instead of the text report",
    )
    args = p.parse_args()

    state = load_json(args.state)
    manifest = load_json(args.manifest)
    triage = load_triage(args.triage_manifest)
    rows, orphans = build_rows(state, manifest, triage)

    instances = instance_keys(args.instances)
    if instances and args.repo.exists():
        attach_strata(rows, instances, args.repo)

    summary = build_summary(rows, orphans, args)

    if args.json:
        json.dump(summary, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        print_report(rows, orphans, args, summary)


if __name__ == "__main__":
    main()
