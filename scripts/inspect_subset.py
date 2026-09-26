#!/usr/bin/env python
"""Inspect the stylistic composition of a Google Fonts checkout.

Reports per-category family/font counts, variable-font availability, and how
many families can supply the target glyph (and at how many contrasting
weights).  Useful for sanity-checking the pools the stratified sampler draws
from; see ``scripts/stratified_subset.py`` for the sampler itself.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from hrothgar.googlefonts import GoogleFonts

NTYPES = ("Serif", "Sans", "Handwriting", "Script", "Monospace", "Display", "Other")


def parse_weights(metadata_pb: Path) -> dict[str, int]:
    """filename -> weight from a family METADATA.pb."""
    if not metadata_pb.exists():
        return {}
    txt = metadata_pb.read_text(encoding="utf-8", errors="replace")
    out: dict[str, int] = {}
    for block in re.findall(r"fonts\s*\{(.*?)\n\s*\}", txt, re.S):
        fn = re.search(r'filename:\s*"([^"]+)"', block)
        wt = re.search(r"weight:\s*(\d+)", block)
        if fn:
            out[fn.group(1)] = int(wt.group(1)) if wt else 400
    return out


def is_variable(path: Path) -> bool:
    return "[" in path.name


def fvar_axes(path: Path) -> set[str]:
    """Design axes present in a variable font (empty for static)."""
    if not is_variable(path):
        return set()
    from fontTools.ttLib import TTFont

    try:
        return {a.axisTag for a in TTFont(path, lazy=True)["fvar"].axes}
    except Exception:
        return set()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--json-out", default=None)
    p.add_argument("--max-print-multi", type=int, default=30)
    args = p.parse_args()

    gf = GoogleFonts(args.repo, max_fonts=args.limit)
    fonts = gf.fonts
    print(f"Loaded fonts (max_fonts={args.limit}): {len(fonts)}")
    print(
        f"Path range: {fonts[0].path.relative_to(args.repo)} -> "
        f"{fonts[-1].path.relative_to(args.repo)}"
    )

    # --- family grouping -------------------------------------------------
    by_family: dict[str, list] = defaultdict(list)
    for f in fonts:
        by_family[f.family].append(f)
    sizes = Counter(len(v) for v in by_family.values())

    print(f"\nFamilies: {len(by_family)}")
    print("Fonts-per-family histogram:", dict(sorted(sizes.items())))

    # --- category (from GF tags) ---
    def cat(f) -> str:
        c = f.category()
        return c if c in NTYPES else "Other"

    fonts_per_cat = Counter(cat(f) for f in fonts)
    fams_per_cat: dict[str, int] = defaultdict(int)
    for fam, fs in by_family.items():
        fams_per_cat[cat(fs[0])] += 1

    print("\n=== By category ===")
    print(
        f"{'category':<12}{'families':>9}{'fonts':>8}{'fonts/fam':>10}{'var fonts':>10}"
    )
    for c in NTYPES:
        fams = fams_per_cat.get(c, 0)
        nf = fonts_per_cat.get(c, 0)
        var = sum(1 for f in fonts if cat(f) == c and is_variable(f.path))
        ratio = f"{nf / fams:.2f}" if fams else "-"
        if nf or fams:
            print(f"{c:<12}{fams:>9}{nf:>8}{ratio:>10}{var:>10}")

    # --- variable vs static, overall and by category ---------------------
    print("\n=== Variable vs static ===")
    var_wght = 0
    for c in NTYPES:
        sub = [f for f in fonts if cat(f) == c]
        if not sub:
            continue
        var = [f for f in sub if is_variable(f.path)]
        wght = [f for f in var if "wght" in fvar_axes(f.path)]
        var_wght += len(wght)
        print(
            f"{c:<12} static={len(sub) - len(var):>4}  variable={len(var):>4}"
            f"  of which wght-axis={len(wght):>3}"
        )
    print(f"Total variable files with a wght axis: {var_wght}")

    # --- weight-multiplicity within families -----------------------------
    multi_fams = 0
    fam_rows = []
    for fam, fs in sorted(by_family.items()):
        meta = parse_weights(fs[0].path.parent / "METADATA.pb")
        ws = sorted({meta.get(f.path.name, 400) for f in fs})
        has_var = any(is_variable(f.path) for f in fs)
        if len(ws) > 1:
            multi_fams += 1
        fam_rows.append(
            {
                "family": fam,
                "n_fonts": len(fs),
                "weights": ws,
                "category": cat(fs[0]),
                "variable": has_var,
            }
        )
    print(f"\nFamilies with >1 distinct weight: {multi_fams} / {len(by_family)}")
    print(f"Multi-weight families (showing up to {args.max_print_multi}):")
    shown = 0
    for r in fam_rows:
        if len(r["weights"]) > 1:
            if shown < args.max_print_multi:
                print(
                    f"  {r['family']:<22} {r['category']:<12} "
                    f"n={r['n_fonts']:<2} weights={r['weights']} var={r['variable']}"
                )
            shown += 1

    # --- instance-per-family concentration by category -------------------
    # A training row is a (font, codepoint) pair, so a family with k weight
    # files contributes k x the rows of a single-weight family: this ratio is
    # the stylistic oversampling factor.
    print("\n=== Stylistic oversampling (instances per family) ===")
    for c in NTYPES:
        fs = [r for r in fam_rows if r["category"] == c]
        if not fs:
            continue
        inst = sum(r["n_fonts"] for r in fs)
        maxf = max(r["n_fonts"] for r in fs)
        print(
            f"{c:<12} families={len(fs):>3} instances={inst:>4} "
            f"mean={inst / len(fs):.2f}  max_in_family={maxf}"
        )

    # --- projection: expand variable fonts across K wght instances --------
    print("\n=== Projected instances if variable fonts are wght-expanded ===")
    has_wght = {f.path.as_posix(): ("wght" in fvar_axes(f.path)) for f in fonts}
    for k in (3, 5):
        print(f"  K={k} wght instances per variable file:")
        for c in NTYPES:
            fs = [f for f in fonts if cat(f) == c]
            if not fs:
                continue
            n_var = sum(1 for f in fs if has_wght[f.path.as_posix()])
            proj = len(fs) - n_var + k * n_var
            print(f"    {c:<12} {len(fs):>4} -> {proj:>4}")
        var_tot = sum(has_wght.values())
        print(
            f"    {'TOTAL':<12} {len(fonts):>4} -> "
            f"{len(fonts) - var_tot + k * var_tot:>4}"
        )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(fam_rows, indent=2) + "\n")
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
