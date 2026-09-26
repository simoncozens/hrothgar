#!/usr/bin/env python
"""Stratified subset sampler CLI for the factorized font-ID diffusion dataset.

Thin command-line wrapper over :mod:`hrothgar.diffusion.dataset_fontid`.  Selects
``n`` training instances balanced across text/fancy strata, with in-family weight
contrast, variable-font locations, and a target covreage fraction of families
containing U+20B9 ₹.  Unit records are cached to JSON for fast iteration; pass
``--rebuild-cache`` after changing the font repo or the needed codepoints.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from hrothgar.dataset_constants import LATIN_KERNEL
from hrothgar.diffusion.dataset_fontid import (
    DEFAULT_STRATA,
    RUPEE,
    STRATA,
    load_or_build_units,
    select_subset,
)


def parse_strata(args) -> dict[str, float]:
    if args.strata:
        return {
            k.strip(): float(v)
            for k, v in (part.split(":") for part in args.strata.split(","))
        }
    text = args.text_frac
    fancy = 1.0 - text - args.mono_frac
    return {
        "sans": text * (1 - args.serif_share),
        "serif": text * args.serif_share,
        "display": fancy * args.fancy_split[0],
        "script": fancy * args.fancy_split[1],
        "handwriting": fancy * args.fancy_split[2],
        "mono": args.mono_frac,
    }


def print_report(rep: dict) -> None:
    print(
        f"\nRequested {rep['requested']} instances -> got {rep['n_instances']} "
        f"({rep['distinct_instances']} distinct + "
        f"{rep['oversampled_instances']} oversampled) from "
        f"{rep['families']} families / {rep['units']} units (seed {rep['seed']})"
    )
    print(f"{'stratum':<12}{'instances':>10}{'share':>8}")
    for st in STRATA:
        ni = rep["instances_per_stratum"].get(st, 0)
        if ni:
            print(f"{st:<12}{ni:>10}" f"{ni / max(rep['n_instances'], 1):>8.1%}")
    print(f"\nFancy vs text: {rep['fancy_instances']} / {rep['text_instances']}")
    print(
        f"  sans/serif: {rep['instances_per_stratum'].get('sans', 0)} / "
        f"{rep['instances_per_stratum'].get('serif', 0)}"
    )
    print(
        f"  display/script/handwriting: "
        f"{rep['instances_per_stratum'].get('display', 0)} / "
        f"{rep['instances_per_stratum'].get('script', 0)} / "
        f"{rep['instances_per_stratum'].get('handwriting', 0)}"
    )
    print(
        f"\nPer unit {rep['unit_k_histogram']} | per family "
        f"{rep['family_k_histogram']}"
    )
    print(
        f"Styles: {rep['style_counts']} | variable instances: "
        f"{rep['variable_instances']}"
    )
    print(
        f"Multi-instance units: {rep['multi_instance_units']} "
        f"(mean span {rep['mean_multi_unit_weight_span']:.0f}) | "
        f"families with both styles: {rep['families_with_both_styles']}"
    )
    print(
        f"Families with target: {rep['families_with_target']}/{rep['families']} "
        f"({rep['target_family_frac']:.0%})"
    )


def print_examples(selected, k: int = 10) -> None:
    by_unit: dict[str, list] = defaultdict(list)
    for s in selected:
        by_unit[s.unit].append(s)
    multi = sorted(
        (u for u, ss in by_unit.items() if len(ss) >= 2), key=lambda u: -len(by_unit[u])
    )[:k]
    print("\nSample multi-instance units (contrast points):")
    for u in multi:
        ss = sorted(by_unit[u], key=lambda s: s.weight)
        pts = ", ".join(f"{s.weight}{'(var)' if s.variable else ''}" for s in ss)
        print(f"  {ss[0].stratum:<11} {u:<34} {pts}")
    print("\nSample single-instance units by stratum:")
    singles = sorted(u for u, ss in by_unit.items() if len(ss) == 1)
    by_st: dict[str, list[str]] = defaultdict(list)
    for u in singles:
        by_st[by_unit[u][0].stratum].append(u)
    for st in STRATA:
        if by_st.get(st):
            print(f"  {st:<11} {', '.join(by_st[st][:6])}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--repo", default=os.environ.get("GOOGLE_FONTS_REPO"))
    p.add_argument(
        "--num-instances",
        type=int,
        required=True,
        help="target stratified instance count",
    )
    p.add_argument(
        "--strata",
        default=None,
        help="override, e.g. 'sans:0.25,serif:0.25,display:0.2,"
        "script:0.2,handwriting:0.1'",
    )
    p.add_argument("--text-frac", type=float, default=0.5)
    p.add_argument("--serif-share", type=float, default=0.5)
    p.add_argument(
        "--fancy-split",
        type=float,
        nargs=3,
        default=(0.4, 0.4, 0.2),
        metavar=("DISPLAY", "SCRIPT", "HANDWRITING"),
    )
    p.add_argument("--mono-frac", type=float, default=0.0)
    p.add_argument("--max-per-unit", type=int, default=3)
    p.add_argument("--avg-instances", type=float, default=1.6)
    p.add_argument("--target-frac", type=float, default=0.7)
    p.add_argument(
        "--min-coverage",
        type=int,
        default=21,
        help="Min needed-codepoint coverage for a unit to be eligible",
    )
    p.add_argument(
        "--no-prefer-multi-target", dest="prefer_multi_target", action="store_false"
    )
    p.add_argument("--no-replacement", dest="replacement", action="store_false")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--cache",
        type=Path,
        default=Path(os.environ.get("FONT_DB_CACHE", "/tmp/hrothgar_units.json")),
    )
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--examples", type=int, default=10)
    args = p.parse_args()

    if not args.repo:
        raise SystemExit("Provide --repo or set GOOGLE_FONTS_REPO")

    needed = set(LATIN_KERNEL) | {RUPEE}
    units = load_or_build_units(args.repo, needed, args.cache, args.rebuild_cache)
    strata_fracs = parse_strata(args)
    print("Strata: " + ", ".join(f"{k}={v:.3f}" for k, v in strata_fracs.items()))

    selected, rep = select_subset(
        units,
        args.num_instances,
        strata_fracs,
        max_per_unit=args.max_per_unit,
        avg_instances=args.avg_instances,
        target_frac=args.target_frac,
        prefer_multi_target=args.prefer_multi_target,
        replacement=args.replacement,
        min_coverage=args.min_coverage,
        seed=args.seed,
    )
    print_report(rep)
    if args.examples:
        print_examples(selected, args.examples)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {"report": rep, "instances": [asdict(s) for s in selected]}, indent=2
            )
            + "\n"
        )
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
