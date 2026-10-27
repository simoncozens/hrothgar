 #!/usr/bin/env python3
"""Probe the shared GlyphEncoder's spatial resolution with a variable-axis sweep.

Runs weight-free experiments that answer, independently of the trained
contrastive model, whether the GlyphEncoder *can carry* the fine typographic
detail that distinguishes styles:

* **Architecture resolution** — the receptive field and effective stride of the
  encoder in input pixels (the encoder's strided convs downsample the input).
* **Axis sweep** — render a parametric font at the same glyph set and per-font
  frame the similarity dataset uses while sweeping one design axis, then measure
  (a) how much the input image changes, (b) how much of that change survives the
  encoder's output resolution, and (c) how much a *freshly-initialised* encoder's
  output feature map changes.

Usage::

    PYTHONPATH=Lib python scripts/similarity/glyph_encoder_probe.py \
        --glyph-size 128 --downsample 2 --coarse-grid-size 32

Note: the ``FontStyleEmbedderConfig`` class and the training CLI (``train.py``)
both default ``glyph_size`` to 128, ``encoder_downsample`` to 2 and
``coarse_grid_size`` to 32, so the defaults above match a model trained with
default args.  The receptive field (9px) and stride (2px) are the same either
way, but running at ``--glyph-size 64`` or ``--downsample 4`` halves the *em*
resolution — exactly the regime that hid terminal rounding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hrothgar.glyph_encoder_probe import run_probe
from hrothgar.similarity.config import DEFAULT_INPUT_CODEPOINTS

_REPO_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_SPECS = [
    {
        "path": str(_REPO_ROOT / "GoogleSansFlex[ROND,opsz,slnt,wdth,wght].ttf"),
        "axis": "ROND",
        "to": 100.0,
    },
    {
        "path": str(_REPO_ROOT / "RobotoDelta-Roman-VF.ttf"),
        "axis": "YOPQ",
        "to": 2.0,
    },
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--glyph-size", type=int, default=128)
    p.add_argument("--downsample", type=int, choices=(2, 4, 8), default=2)
    p.add_argument("--coarse-grid-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--encoder-seeds", type=int, default=5)
    p.add_argument(
        "--glyphs",
        default=None,
        help="override the input glyphs (a literal string, e.g. 'aogAO')",
    )
    p.add_argument("--json", default=None, help="write full results here")
    return p.parse_args()


def _print_geometry(geometry: dict) -> None:
    print("\n[1] GlyphEncoder spatial resolution")
    print(
        f"  input {geometry['input_size']}px -> feature map "
        f"{geometry['output_size']}px  (downsample {geometry['effective_stride']}x)"
    )
    print(
        f"  receptive field {geometry['receptive_field']}px  "
        f"effective stride {geometry['effective_stride']}px"
    )
    print(
        "  => features finer than ~"
        f"{geometry['effective_stride']}px are sub-sampled; an isolated "
        f"1px change is diluted ~{geometry['effective_stride']**2}x across an "
        "output cell."
    )


def _print_sweep(sweep: dict) -> None:
    print(f"\n[2] {sweep['axis']} sweep  ({Path(sweep['font']).name})")
    print(
        f"  default {sweep['default']:g} -> {sweep['to']:g} "
        f"(axis range [{sweep['lo']:g}..{sweep['hi']:g}], {len(sweep['steps'])} steps)"
    )
    print("  input difference vs. default:")
    for v, d in zip(sweep["steps"], sweep["input_diffs"].values()):
        print(
            f"    {v:8.2f}  mean|d|={d['mean_abs']:.5f}  max|d|={d['max_abs']:.2f}  "
            f"frac>0.1={d['fraction_changed']:.4f}  changed_px={d['changed_pixels']}"
        )

    surv = sweep["downsampling_survival"]
    print(
        f"  downsampling survival (full swing through a {surv['factor']}x box pool):"
    )
    print(
        f"    energy {surv['energy_full']:.4g} -> {surv['energy_pooled']:.4g}  "
        f"(ratio {surv['energy_ratio']:.4f})"
    )
    print(
        f"    mean|d| {surv['mean_abs_full']:.5f} -> {surv['mean_abs_pooled']:.5f}"
    )
    print(
        f"    output cells moved {surv['changed_output_cells']} / "
        f"{surv['output_cells_total']} "
        f"({surv['changed_output_cells'] / surv['output_cells_total']:.2%})"
    )

    summ = sweep["summary_survival"]
    print(
        f"  summary survival (input pooled to the {summ['grid_size']}x"
        f"{summ['grid_size']} coarse grid the summary embedding sees):"
    )
    print(
        f"    energy {summ['energy_full']:.4g} -> {summ['energy_pooled']:.4g}  "
        f"(ratio {summ['energy_ratio']:.4f})"
    )
    print(
        f"    output cells moved {summ['changed_output_cells']} / "
        f"{summ['output_cells_total']} "
        f"({summ['changed_output_cells'] / summ['output_cells_total']:.2%})"
    )

    sens = sweep["sensitivity"]
    print("  sensitivity:")
    print(
        f"    mean|d| per axis unit {sens['mean_abs_per_unit']:.5f}  "
        f"(span {sens['span']:g})"
    )
    md = sens["min_detectable_delta"]
    if md is None:
        print(
            f"    no step down to {sens['span'] / sens['fine_steps']:g} axis-units "
            "moved a single output cell"
        )
    else:
        print(
            f"    smallest detectable axis delta ~{md:.3f} "
            f"(mean|d|={sens['min_detectable_mean_abs']:.5f}) "
            f"[fine step {abs(sens['span']) / sens['fine_steps']:.3f}]"
        )

    out = sweep["output_diff"]
    print(f"  untrained-encoder output difference vs default:")
    print(
        f"    frobenius {out['frobenius']:.3f}  relative {out['relative']:.4f}  "
        f"cosine {out['cosine']:.5f}"
    )


def main() -> int:
    args = _parse_args()
    codepoints = (
        [ord(c) for c in args.glyphs]
        if args.glyphs is not None
        else list(DEFAULT_INPUT_CODEPOINTS)
    )

    report = run_probe(
        _DEFAULT_SPECS,
        codepoints=codepoints,
        glyph_size=args.glyph_size,
        downsample=args.downsample,
        coarse_grid_size=args.coarse_grid_size,
        steps=args.steps,
        encoder_seeds=args.encoder_seeds,
    )

    _print_geometry(report["geometry"])
    for sweep in report["sweeps"]:
        _print_sweep(sweep)

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
