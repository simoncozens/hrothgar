#!/usr/bin/env python
"""Measure how much the diffusion generator's output collapses toward a shared
"average glyph" relative to the fonts' own ground truth.

Given the pre-generated tree under ``outputs/generated/`` (produced by sampling
the factorized model and placing each crop-to-ink output back on the baseline
with :func:`hrothgar.render.place_glyph`), this compares, for each codepoint,
the *spread* of the generated glyphs across families against the spread of the
ground-truth glyphs across those same families.

Method
------

For every ``<family>/<family>-U+XXXX.png`` we mirror ``triage_identical.py``'s
data conventions:

1. Crop the 384x256 placed canvas to its left 256 columns -- the 2em
   ``UNIFORM_FRAME`` that :func:`~hrothgar.render.render` draws at ``size=256``
   -- so generated and ground truth share a scale and coordinate system.
2. Render the ground truth with ``render(font, codepoint=cp, size=256)``.  A
   codepoint the font lacks (e.g. the rupee) has no GT and is skipped; so is a
   blank glyph.
3. Downsample both to ``--size`` and accumulate, per ``(codepoint, stratum)``,
   the running sums needed for the *dispersion* (mean squared distance to the
   set's own centroid):

       disp = mean_i ||x_i - xbar||^2  =  sum_i||x_i||^2 / N  -  ||xbar||^2

   Accumulating sums (not the glyphs) keeps memory O(codepoints), not
   O(codepoints x families).

The headline per codepoint is ``ratio = disp_generated / disp_ground_truth``.
``ratio < 1`` means the model's glyphs are *less diverse* than the fonts' own --
i.e. the generator is collapsing toward a shared mean.  ``ratio ~ 1`` means the
spread is preserved; ``ratio > 1`` means the model is over-dispersed (noisy).
The centroid shift ``||xbar_gen - xbar_gt||^2`` is reported separately as a bias
term, so "collapsed spread" and "shifted mean" are not conflated.

Dispersion is computed on raw (downsampled) ink, so it deliberately conflates
weight and shape variation -- both are legitimate style dimensions whose
collapse we want to see.  Results are broken down by font stratum (sans / serif
/ display / script / handwriting / mono / other), because the interesting
question is whether collapse is uniform or worse in e.g. display.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image

from hrothgar.dataset import style_bucket
from hrothgar.googlefonts import GoogleFont, GoogleFonts, StandaloneFont
from hrothgar.render import render

STRATA = ("sans", "serif", "display", "script", "handwriting", "mono", "other")
_INK_THRESHOLD = 0.5


def sanitize(name: str) -> str:
    """Replicate the filename sanitization used to write the generated tree."""
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")
    return cleaned or "family"


def key_for(inst: dict) -> str:
    """The directory name an instance was written under."""
    style_suffix = "" if inst.get("style", "normal") == "normal" else "-italic"
    return sanitize(inst["family"]) + style_suffix


def load_instance_map(path: Path) -> dict[str, dict]:
    """``key -> instance``, from the training loop's ``.instances.json`` sidecar."""
    with path.open("r", encoding="utf-8") as f:
        instances = json.load(f)
    mapping: dict[str, dict] = {}
    for inst in instances:
        key = key_for(inst)
        if key in mapping:
            raise SystemExit(f"Instance key collision for {key!r}")
        mapping[key] = inst
    return mapping


def _parse_codepoints(spec: str | None) -> set[int] | None:
    """Parse a ``--codepoints`` spec into a set of Unicode ords, or ``None``."""
    if spec is None or spec.strip() == "" or spec.strip().lower() == "all":
        return None
    out: set[int] = set()
    for part in spec.split(","):
        for ch in part:
            if not ch.isspace():
                out.add(ord(ch))
    return out


def _stratum(gfont: GoogleFont | None) -> str:
    """Replicate :meth:`hrothgar.dataset.Unit.stratum` for a single font."""
    if gfont is None:
        return "other"
    bucket = style_bucket(gfont)  # mono / fancy / sans / serif / other
    if bucket != "fancy":
        return bucket
    if gfont.category() == "Script":
        return "script"
    if "HANDWRITING" in gfont.classification():
        return "handwriting"
    return "display"


def _small(arr: np.ndarray, size: int) -> np.ndarray:
    """Downsample a [0, 1] ink=0 / white=1 array to ``(size, size)``."""
    im = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def _blank(arr: np.ndarray) -> bool:
    return not bool((arr < _INK_THRESHOLD).any())


def render_gt(font: StandaloneFont, codepoint: int) -> np.ndarray | None:
    """Ground truth in the uniform 2em frame, or ``None`` if absent."""
    if not font.has_codepoint(codepoint):
        return None
    img, _ = render(font, codepoint=codepoint, size=256)
    return img.numpy()


def _dispersion(sum_x: np.ndarray, sum2_x: float, n: int) -> tuple[np.ndarray, float]:
    """Return ``(mean, dispersion)`` from running sums."""
    mean = sum_x / n
    disp = sum2_x / n - float((mean * mean).sum())
    return mean, max(disp, 0.0)  # clamp tiny negative round-off


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--instances",
        type=Path,
        default=Path("models/fontid_diffusion.pth.instances.json"),
        help=".instances.json sidecar mapping each generated directory to its "
        "source font instance",
    )
    p.add_argument(
        "--repo",
        type=str,
        default=os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts"),
        help="Google Fonts repo root, to resolve repo-relative instance paths",
    )
    p.add_argument(
        "--input-dir", type=Path, default=Path("outputs/generated"),
        help="Directory tree of generated glyphs (family dirs)",
    )
    p.add_argument(
        "--codepoints", type=str, default="all",
        help="Comma/string of characters to analyse, or 'all' (default)",
    )
    p.add_argument(
        "--size", type=int, default=64,
        help="Downsample glyphs to (size x size) before measuring spread",
    )
    p.add_argument(
        "--min-families", type=int, default=20,
        help="Only report a codepoint with at least this many (family, GT) pairs",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Only process this many glyphs (for a smoke test)",
    )
    p.add_argument(
        "--families", type=int, default=None,
        help="Only process this many family directories (for a smoke test)",
    )
    p.add_argument("--output-json", type=Path, default=None,
                   help="Write the full per-(codepoint, stratum) report here")
    args = p.parse_args()

    repo = Path(args.repo)
    mapping = load_instance_map(args.instances)
    cp_filter = _parse_codepoints(args.codepoints)
    d = args.size * args.size

    # Family name -> stratum, computed once from Google Fonts metadata.
    try:
        gf = GoogleFonts(repo)
    except Exception as exc:  # missing tags csv on a partial checkout
        print(f"Warning: could not load GoogleFonts metadata ({exc}); "
              "stratum will fall back to 'other'.")
        gf = None

    stratum_cache: dict[str, str] = {}

    def stratum_for(family: str, path: str) -> str:
        if family in stratum_cache:
            return stratum_cache[family]
        gfont = None
        if gf is not None:
            gfont = GoogleFonts.families_by_name.get(family)
        if gfont is None:
            try:
                gfont = GoogleFont(repo / path, gf)
            except Exception:
                gfont = None
        stratum = _stratum(gfont)
        stratum_cache[family] = stratum
        return stratum

    # (codepoint, stratum) -> running sums.
    acc: dict[tuple[int, str], dict] = {}

    def slot(cp: int, stratum: str) -> dict:
        k = (cp, stratum)
        a = acc.get(k)
        if a is None:
            a = {"sum_gen": np.zeros(d, dtype=np.float64), "sum2_gen": 0.0,
                 "sum_gt": np.zeros(d, dtype=np.float64), "sum2_gt": 0.0, "n": 0}
            acc[k] = a
        return a

    counts = {"total": 0, "no_gt": 0, "blank": 0, "filtered": 0, "unknown_family": 0}

    family_dirs = sorted(p for p in args.input_dir.iterdir() if p.is_dir())
    for fi, family_dir in enumerate(tqdm.tqdm(family_dirs, total=len(family_dirs))):
        if args.families is not None and fi >= args.families:
            break

        key = family_dir.name
        inst = mapping.get(key)
        if inst is None:
            counts["unknown_family"] += sum(1 for _ in family_dir.glob("*.png"))
            continue

        font = StandaloneFont(repo / inst["path"])
        stratum = stratum_for(inst["family"], inst["path"])

        for png in sorted(family_dir.glob("*.png")):
            if args.limit is not None and counts["total"] >= args.limit:
                break

            m = re.match(r".*-U\+([0-9A-Fa-f]{4,6})\.png$", png.name)
            if m is None:
                continue
            codepoint = int(m.group(1), 16)
            counts["total"] += 1
            if cp_filter is not None and codepoint not in cp_filter:
                counts["filtered"] += 1
                continue

            gt = render_gt(font, codepoint)
            if gt is None:
                counts["no_gt"] += 1
                continue

            gen = np.asarray(Image.open(png).convert("L"), dtype=np.float32) / 255.0
            gen_small = _small(gen[:, :256], args.size)
            gt_small = _small(gt, args.size)
            if _blank(gen_small) or _blank(gt_small):
                counts["blank"] += 1
                continue

            a = slot(codepoint, stratum)
            g = gen_small.ravel().astype(np.float64)
            t = gt_small.ravel().astype(np.float64)
            a["sum_gen"] += g
            a["sum2_gen"] += float((g * g).sum())
            a["sum_gt"] += t
            a["sum2_gt"] += float((t * t).sum())
            a["n"] += 1

    # Per-(codepoint, stratum) records (only the ones with enough families).
    records = []
    for (cp, stratum), a in acc.items():
        if a["n"] < args.min_families:
            continue
        _, disp_gen = _dispersion(a["sum_gen"], a["sum2_gen"], a["n"])
        _, disp_gt = _dispersion(a["sum_gt"], a["sum2_gt"], a["n"])
        mean_gen = a["sum_gen"] / a["n"]
        mean_gt = a["sum_gt"] / a["n"]
        bias = float(((mean_gen - mean_gt) ** 2).sum())
        ratio = disp_gen / disp_gt if disp_gt > 1e-12 else float("nan")
        records.append({
            "codepoint": cp,
            "stratum": stratum,
            "n": a["n"],
            "disp_gen": disp_gen,
            "disp_gt": disp_gt,
            "rms_gen": float(np.sqrt(disp_gen)),
            "rms_gt": float(np.sqrt(disp_gt)),
            "ratio": ratio,
            "bias": bias,
        })

    if not records:
        raise SystemExit("No (codepoint, stratum) groups passed --min-families; "
                         "check --repo, --codepoints, and --min-families.")

    # Aggregate per codepoint (combine the per-stratum running sums).
    by_cp: dict[int, dict] = {}
    for (cp, stratum), a in acc.items():
        b = by_cp.get(cp)
        if b is None:
            b = {"sum_gen": np.zeros(d, dtype=np.float64), "sum2_gen": 0.0,
                 "sum_gt": np.zeros(d, dtype=np.float64), "sum2_gt": 0.0, "n": 0}
            by_cp[cp] = b
        b["sum_gen"] += a["sum_gen"]
        b["sum2_gen"] += a["sum2_gen"]
        b["sum_gt"] += a["sum_gt"]
        b["sum2_gt"] += a["sum2_gt"]
        b["n"] += a["n"]

    cp_rows = []
    for cp, b in by_cp.items():
        if b["n"] < args.min_families:
            continue
        _, disp_gen = _dispersion(b["sum_gen"], b["sum2_gen"], b["n"])
        _, disp_gt = _dispersion(b["sum_gt"], b["sum2_gt"], b["n"])
        ratio = disp_gen / disp_gt if disp_gt > 1e-12 else float("nan")
        cp_rows.append({"codepoint": cp, "n": b["n"],
                        "disp_gen": disp_gen, "disp_gt": disp_gt, "ratio": ratio})
    cp_rows.sort(key=lambda r: r["ratio"])

    # Per-stratum summary (median/mean ratio across codepoints).
    stratum_rows = {}
    for s in STRATA:
        ratios = [r["ratio"] for r in records if r["stratum"] == s]
        if not ratios:
            continue
        n_families = sum(r["n"] for r in records if r["stratum"] == s)
        stratum_rows[s] = {
            "codepoints": len(ratios),
            "families": n_families,
            "median_ratio": float(np.median(ratios)),
            "mean_ratio": float(np.mean(ratios)),
        }

    def _label(cp: int) -> str:
        ch = chr(cp)
        return f"{ch!r} U+{cp:04X}" if ch.isprintable() and not ch.isspace() else f"U+{cp:04X}"

    print("\n=== diversity collapse (ratio = generated spread / ground-truth spread) ===")
    print(f"processed {counts['total']} glyphs: "
          f"{counts['no_gt']} no-gt, {counts['blank']} blank, "
          f"{counts['filtered']} filtered, {counts['unknown_family']} unknown-family")
    print(f"{'codepoint':>18}  {'N':>5}  {'disp_gen':>10}  {'disp_gt':>10}  {'ratio':>6}")
    for r in cp_rows:
        print(f"{_label(r['codepoint']):>18}  {r['n']:>5}  "
              f"{r['disp_gen']:>10.5f}  {r['disp_gt']:>10.5f}  {r['ratio']:>6.3f}")

    print("\n=== per stratum (median ratio across codepoints; < 1 = collapse) ===")
    print(f"{'stratum':>12}  {'codepoints':>10}  {'families':>8}  {'median':>7}  {'mean':>7}")
    for s in STRATA:
        row = stratum_rows.get(s)
        if row is None:
            continue
        print(f"{s:>12}  {row['codepoints']:>10}  {row['families']:>8}  "
              f"{row['median_ratio']:>7.3f}  {row['mean_ratio']:>7.3f}")

    if args.output_json:
        report = {
            "counts": counts,
            "per_codepoint": cp_rows,
            "per_stratum": stratum_rows,
            "detail": [
                {**r, "codepoint_hex": f"U+{r['codepoint']:04X}"} for r in records
            ],
        }
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        print(f"\nreport -> {args.output_json}")


if __name__ == "__main__":
    main()
