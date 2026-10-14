"""Empirically characterize how variable-font bolding displaces outline nodes.

Answers three questions that decide the Tier 0 model's functional form:

1. Is the regular -> bold displacement purely along each node's outward normal
   (pure thickening), or is there a substantial tangential component?
2. Is the normal offset a smooth function of local stroke width and stroke
   orientation (i.e. is it "proportional expansion", and is there contrast)?
3. Do corner nodes behave differently from smooth nodes?

Run with::

    PYTHONPATH=Lib python scripts/analyze_weight_decomposition.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import uharfbuzz as hb

from hrothgar.weight_adjust.outlines import (
    NodeDecomposition,
    decompose,
    extract_glyph,
)

# (label, font path, regular wght, bold wght)
FONTS = [
    ("roboto (sans)", "tests/dummy_repo/ofl/roboto/Roboto[wdth,wght].ttf", 400.0, 700.0),
    ("playfair (serif)", "tests/dummy_repo/ofl/playfairdisplay/PlayfairDisplay[wght].ttf", 400.0, 700.0),
    ("shantell (handwriting)", "tests/dummy_repo/ofl/shantellsans/ShantellSans[BNCE,INFM,SPAC,wght].ttf", 300.0, 700.0),
]

# A spread of structural types: stems, bowls, counters, diagonals, terminals.
GLYPHS = "HOSDBogaeemnl"


@dataclass
class _Agg:
    nodes: int = 0
    sum_off2: float = 0.0
    sum_tan2: float = 0.0
    sum_off: float = 0.0
    sum_abs_off: float = 0.0
    sum_abs_tan: float = 0.0
    n_pos_off: int = 0

    corner_nodes: int = 0
    corner_off2: float = 0.0
    corner_tan2: float = 0.0
    corner_abs_off: float = 0.0
    corner_abs_tan: float = 0.0

    smooth_nodes: int = 0
    smooth_off2: float = 0.0
    smooth_tan2: float = 0.0
    smooth_abs_off: float = 0.0
    smooth_abs_tan: float = 0.0

    # normal-angle buckets in [0, 90) degrees -> (sum offset, count)
    angle_off: dict[int, float] = field(default_factory=dict)
    angle_count: dict[int, int] = field(default_factory=dict)

    # width buckets -> (sum offset, count)
    width_off: dict[int, float] = field(default_factory=dict)
    width_count: dict[int, int] = field(default_factory=dict)


def _fold_angle(nx: float, ny: float) -> float:
    """Fold a normal direction into [0, 90) degrees (0 = vertical stem)."""
    deg = math.degrees(math.atan2(ny, nx)) % 180.0
    if deg > 90.0:
        deg = 180.0 - deg
    return deg


def _add(agg: _Agg, d: NodeDecomposition) -> None:
    agg.nodes += 1
    agg.sum_off2 += d.offset * d.offset
    agg.sum_tan2 += d.tangential * d.tangential
    agg.sum_off += d.offset
    agg.sum_abs_off += abs(d.offset)
    agg.sum_abs_tan += abs(d.tangential)
    if d.offset > 0:
        agg.n_pos_off += 1

    if d.is_corner:
        agg.corner_nodes += 1
        agg.corner_off2 += d.offset * d.offset
        agg.corner_tan2 += d.tangential * d.tangential
        agg.corner_abs_off += abs(d.offset)
        agg.corner_abs_tan += abs(d.tangential)
    else:
        agg.smooth_nodes += 1
        agg.smooth_off2 += d.offset * d.offset
        agg.smooth_tan2 += d.tangential * d.tangential
        agg.smooth_abs_off += abs(d.offset)
        agg.smooth_abs_tan += abs(d.tangential)

    ang = _fold_angle(d.nx, d.ny)
    bucket = int(ang // 15) * 15
    agg.angle_off[bucket] = agg.angle_off.get(bucket, 0.0) + d.offset
    agg.angle_count[bucket] = agg.angle_count.get(bucket, 0) + 1

    if d.width is not None:
        wbucket = int(d.width // 20) * 20
        agg.width_off[wbucket] = agg.width_off.get(wbucket, 0.0) + d.offset
        agg.width_count[wbucket] = agg.width_count.get(wbucket, 0) + 1


def _report(label: str, agg: _Agg) -> None:
    n = agg.nodes
    rms_off = math.sqrt(agg.sum_off2 / n)
    rms_tan = math.sqrt(agg.sum_tan2 / n)
    # Fraction of displacement energy that is tangential (0 = pure normal).
    tan_energy = agg.sum_tan2 / (agg.sum_off2 + agg.sum_tan2)
    mean_off = agg.sum_off / n
    frac_pos = agg.n_pos_off / n

    print(f"\n=== {label} ===")
    print(f"nodes={n}  rms_offset={rms_off:.1f}  rms_tangential={rms_tan:.1f}")
    print(f"tangential energy fraction = {tan_energy:.3f}  (0 = pure normal offset)")
    print(f"mean offset = {mean_off:+.1f}  fraction positive (thickening) = {frac_pos:.3f}")

    if agg.corner_nodes:
        cn = agg.corner_nodes
        co = math.sqrt(agg.corner_off2 / cn)
        ct = math.sqrt(agg.corner_tan2 / cn)
        ce = agg.corner_tan2 / (agg.corner_off2 + agg.corner_tan2)
        print(f"corners (n={cn}): rms_offset={co:.1f}  rms_tangential={ct:.1f}  tang_energy={ce:.3f}")
    if agg.smooth_nodes:
        sn = agg.smooth_nodes
        so = math.sqrt(agg.smooth_off2 / sn)
        st = math.sqrt(agg.smooth_tan2 / sn)
        se = agg.smooth_tan2 / (agg.smooth_off2 + agg.smooth_tan2)
        print(f"smooth   (n={sn}): rms_offset={so:.1f}  rms_tangential={st:.1f}  tang_energy={se:.3f}")

    # Offset vs stroke orientation (normal angle in [0,90); 0 = vertical stem).
    print("offset vs normal angle (deg):")
    for b in sorted(agg.angle_off):
        mean = agg.angle_off[b] / agg.angle_count[b]
        bar = "#" * max(0, int(abs(mean)))
        print(f"  {b:3d} deg  n={agg.angle_count[b]:4d}  mean_offset={mean:+7.1f}  {bar}")

    if agg.width_count:
        print("offset vs local width (font units):")
        for b in sorted(agg.width_off):
            mean = agg.width_off[b] / agg.width_count[b]
            print(f"  width {b:4d}  n={agg.width_count[b]:4d}  mean_offset={mean:+7.1f}")


def _analyze_font(label: str, path: str, regular: float, bold: float) -> None:
    face = hb.Face(hb.Blob.from_file_path(path))
    font = hb.Font(face)
    agg = _Agg()
    for ch in GLYPHS:
        gid = font.get_nominal_glyph(ord(ch))
        if gid == 0:
            continue
        reg = extract_glyph(face, gid, {"wght": regular})
        bld = extract_glyph(face, gid, {"wght": bold})
        if len(reg.contours) != len(bld.contours):
            continue
        if any(
            len(list(a.segments())) != len(list(b.segments()))
            for a, b in zip(reg.contours, bld.contours)
        ):
            print(f"  !! {ch}: incompatible point counts, skipped")
            continue
        for d in decompose(reg, bld, with_width=True):
            _add(agg, d)
    _report(label, agg)


def main() -> None:
    for label, path, regular, bold in FONTS:
        if not Path(path).exists():
            print(f"missing font: {path}")
            continue
        _analyze_font(label, path, regular, bold)


if __name__ == "__main__":
    main()
