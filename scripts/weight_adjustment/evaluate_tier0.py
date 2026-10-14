"""Evaluate the Tier 0 scalar bolding model on held-out glyphs.

For each variable font we do leave-one-out: fit ``vertical``/``horizontal`` on
every glyph except the held-out one, then measure how well the two-parameter
normal-offset model predicts the held-out glyph's bold displacement.

The headline numbers are:

* the fitted ``vertical``/``horizontal`` (and their stability across folds),
* the normal-offset R^2 on held-out nodes (how much of the thickening a
  two-parameter model captures), and
* the tangential residual (what a pure-normal model leaves behind -- expected to
  concentrate at corners).

Run with::

    PYTHONPATH=Lib python scripts/evaluate_tier0.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import uharfbuzz as hb

from hrothgar.weight_adjust.outlines import (
    Glyph,
    extract_glyph,
    is_corner,
    node_frame,
    stroke_widths,
)
from hrothgar.weight_adjust.tier0 import BoldParams, fit, predict_offset

# (label, font path, regular wght, bold wght)
FONTS = [
    ("roboto (sans)", "tests/dummy_repo/ofl/roboto/Roboto[wdth,wght].ttf", 400.0, 700.0),
    ("playfair (serif)", "tests/dummy_repo/ofl/playfairdisplay/PlayfairDisplay[wght].ttf", 400.0, 700.0),
    ("shantell (handwriting)", "tests/dummy_repo/ofl/shantellsans/ShantellSans[BNCE,INFM,SPAC,wght].ttf", 300.0, 700.0),
]

GLYPHS = "HOSDBogaeemnlpuy"


@dataclass
class _Acc:
    params_v: list[float] = field(default_factory=list)
    params_h: list[float] = field(default_factory=list)

    ss_offset_corner: float = 0.0
    ss_offset_smooth: float = 0.0
    ss_normal_resid_corner: float = 0.0
    ss_normal_resid_smooth: float = 0.0
    ss_tang_corner: float = 0.0
    ss_tang_smooth: float = 0.0
    n_corner: int = 0
    n_smooth: int = 0


def _evaluate_pair(
    regular: Glyph, bold: Glyph, params: BoldParams, acc: _Acc
) -> None:
    widths = stroke_widths(regular)
    for ci, (cr, cb) in enumerate(zip(regular.contours, bold.contours)):
        segs_r = list(cr.segments())
        segs_b = list(cb.segments())
        for i, (sr, sb) in enumerate(zip(segs_r, segs_b)):
            width = widths[ci][i]
            if width is None:
                continue
            tangent, normal = node_frame(cr, i)
            dx = sb.p0.x - sr.p0.x
            dy = sb.p0.y - sr.p0.y
            o_true = dx * normal.x + dy * normal.y
            t_true = dx * tangent.x + dy * tangent.y
            o_pred = predict_offset(params, normal.x, normal.y, width)
            o_resid = o_true - o_pred

            corner = is_corner(cr, i)
            if corner:
                acc.ss_offset_corner += o_true * o_true
                acc.ss_normal_resid_corner += o_resid * o_resid
                acc.ss_tang_corner += t_true * t_true
                acc.n_corner += 1
            else:
                acc.ss_offset_smooth += o_true * o_true
                acc.ss_normal_resid_smooth += o_resid * o_resid
                acc.ss_tang_smooth += t_true * t_true
                acc.n_smooth += 1


def _rms(ss: float, n: int) -> float:
    return math.sqrt(ss / n) if n else 0.0


def _report(label: str, acc: _Acc) -> None:
    n = acc.n_corner + acc.n_smooth
    ss_off = acc.ss_offset_corner + acc.ss_offset_smooth
    ss_nres = acc.ss_normal_resid_corner + acc.ss_normal_resid_smooth
    ss_tang = acc.ss_tang_corner + acc.ss_tang_smooth
    r2 = 1.0 - ss_nres / ss_off if ss_off > 0.0 else 0.0

    v = sum(acc.params_v) / len(acc.params_v)
    h = sum(acc.params_h) / len(acc.params_h)

    print(f"\n=== {label} ===")
    print(f"fitted: vertical={v:.3f}  horizontal={h:.3f}  contrast={v/h if h else float('inf'):.2f}:1")
    print(f"held-out nodes: {n}  ({acc.n_corner} corners, {acc.n_smooth} smooth)")
    print(f"normal-offset R^2 = {r2:.3f}   (1.0 = scalar model captures all thickening)")
    print(f"rms true offset      = {_rms(ss_off, n):5.1f}")
    print(f"rms normal residual  = {_rms(ss_nres, n):5.1f}")
    print(f"rms tangential resid = {_rms(ss_tang, n):5.1f}")
    print(f"  corners: normal_resid={_rms(acc.ss_normal_resid_corner, acc.n_corner):5.1f}  "
          f"tang_resid={_rms(acc.ss_tang_corner, acc.n_corner):5.1f}")
    print(f"  smooth : normal_resid={_rms(acc.ss_normal_resid_smooth, acc.n_smooth):5.1f}  "
          f"tang_resid={_rms(acc.ss_tang_smooth, acc.n_smooth):5.1f}")


def _load_pairs(
    path: str, regular: float, bold: float
) -> list[tuple[Glyph, Glyph]]:
    face = hb.Face(hb.Blob.from_file_path(path))
    font = hb.Font(face)
    pairs: list[tuple[Glyph, Glyph]] = []
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
            continue
        pairs.append((reg, bld))
    return pairs


def main() -> None:
    for label, path, regular, bold in FONTS:
        if not Path(path).exists():
            print(f"missing font: {path}")
            continue
        pairs = _load_pairs(path, regular, bold)
        if len(pairs) < 2:
            print(f"{label}: not enough usable glyphs")
            continue
        acc = _Acc()
        for held_i in range(len(pairs)):
            fit_set = pairs[:held_i] + pairs[held_i + 1 :]
            params = fit(fit_set)
            acc.params_v.append(params.vertical)
            acc.params_h.append(params.horizontal)
            _evaluate_pair(*pairs[held_i], params, acc)
        _report(label, acc)


if __name__ == "__main__":
    main()
