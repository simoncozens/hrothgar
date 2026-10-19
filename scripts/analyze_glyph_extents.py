"""Measure the em-space ink extents of GF Latin Core glyphs across the library.

Motivation
----------
``render_glyph_uniform`` frames every glyph into a fixed 2em x 2em window — the
X origin 0.5em from the left and the baseline 1.5em from the top — so the window
covers ``x in [-0.5, 1.5]`` and ``y in [-0.5, 1.5]`` relative to the origin and
baseline.  If the ink actually occupies much less than that, the framing is
throwing away resolution: a tighter window makes every glyph *larger* at the same
128x128, which is exactly what the content detector's capital/lowercase problem
needs — its cue is an ink-height difference that falls below the encoder's ~8px
acuity in small-x-height designs.

What it measures
----------------
For every ``*.ttf`` in the checkout it reads the ``glyf`` bbox
(``xMin/xMax/yMin/yMax`` / ``unitsPerEm``) of every glyph cmap'd to a GF Latin
Core codepoint, and reports

* the **pooled glyph** quantiles,
* the **per-font envelope** quantiles and the *joint* containment of candidate
  windows (marginal quantiles are not enough: a font can sit at p80 on x and p99
  on y, and a per-font frame must contain the whole envelope), and
* the resulting **o/O cue in pixels** under each candidate framing, which is the
  quantity the content detector actually cares about.

Only the default instance is measured: ``glyf`` stores one outline set, so axis
positions (``opsz``/``GRAD``/``wght``) are not represented here.

Usage
-----
    python3 scripts/analyze_glyph_extents.py --repo /home/simon/others-repos/fonts
    python3 scripts/analyze_glyph_extents.py --pilot-anchors
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# The window ``render_gid_uniform`` frames into, in em relative to the glyph
# origin (x = 0) and the baseline (y = 0, positive up).  Keep in sync with the
# framing in ``hrothgar/render.py``; these are the assumptions under test.
CURRENT_WINDOW = dict(left=-0.5, right=1.5, bottom=-0.5, top=1.5)

# The encoder is at chance on a case pair whose cue is under ~8px at 128x128
# (measured on the content detector): scale the best candidate framings by this.
ACUITY_PX = 8.0

# Canonical framing candidate: the canvas is always 128px, so px/em == 128 / side.
CANVAS_PX = 128.0


def _codepoints(pilot_only: bool) -> list[int]:
    if pilot_only:
        from hrothgar.content_detector.constants import PILOT_ANCHORS

        return sorted(PILOT_ANCHORS)
    from hrothgar.dataset_constants import LATIN_CORE

    return sorted(set(LATIN_CORE))


def glyph_extents(path: str, codepoints: tuple[int, ...]) -> dict | None:
    """Em-space bboxes of every requested glyph in ``path``, plus the o/O cue.

    Returns ``{"boxes": [(xmin, ymin, xmax, ymax), ...], "cue": float | None}``
    where ``cue`` is ``yMax(O) - yMax(o)`` in em — the ink-height difference that
    separates the case pair — or ``None`` if the font lacks either glyph.
    """
    from fontTools.ttLib import TTFont

    try:
        font = TTFont(path, lazy=True)
    except Exception:
        return None
    try:
        if "glyf" not in font or "head" not in font:
            return None
        upem = font["head"].unitsPerEm
        if not upem:
            return None
        cmap = font.getBestCmap()
        if not cmap:
            return None
        glyf = font["glyf"]
        boxes: list[tuple] = []
        ymax_by_codepoint: dict[int, float] = {}
        for codepoint in codepoints:
            name = cmap.get(codepoint)
            if name is None or name not in glyf:
                continue
            glyph = glyf[name]
            box = (glyph.xMin, glyph.yMin, glyph.xMax, glyph.yMax)
            if any(value is None for value in box):
                continue  # blank glyph: no ink, no bbox
            xmin, ymin, xmax, ymax = (value / upem for value in box)
            # Guard against the occasional corrupt bbox (a few fonts write
            # absurd values); these would otherwise set the tail.
            if max(abs(xmin), abs(ymin), abs(xmax), abs(ymax)) > 4.0:
                continue
            boxes.append((xmin, ymin, xmax, ymax))
            ymax_by_codepoint[codepoint] = ymax
        cue = None
        if 0x6F in ymax_by_codepoint and 0x4F in ymax_by_codepoint:
            cue = ymax_by_codepoint[0x4F] - ymax_by_codepoint[0x6F]
        return {"boxes": boxes, "cue": cue}
    except Exception:
        return None
    finally:
        font.close()


def _extents_call(call: tuple[str, tuple[int, ...]]) -> dict | None:
    """``ProcessPoolExecutor.map`` takes a single argument; unpack there."""
    return glyph_extents(*call)


def _quantiles(values: list[float], qs: tuple[float, ...]) -> list[float]:
    return list(np.percentile(np.asarray(values, dtype=float), [q * 100 for q in qs]))


def _print_quantile_table(
    title: str, columns: dict[str, list[float]], qs: tuple[float, ...]
) -> None:
    header = "".join(f"{'p' + str(int(q * 100)):>9}" for q in qs)
    print(f"\n{title}")
    print(f"{'':<10}{header}")
    for name, values in columns.items():
        row = "".join(f"{v:>9.3f}" for v in _quantiles(values, qs))
        print(f"{name:<10}{row}")


def _cue_table(title: str, cues: list[float], sides: list[float] | None) -> None:
    """Print cue-in-pixels for a fixed window and/or per-font framing."""
    arr = np.asarray(cues, dtype=float)
    print(f"\n{title}")
    print(
        f"  cue in em: p1 {np.percentile(arr, 1):.3f}  p5 {np.percentile(arr, 5):.3f}  "
        f"p50 {np.percentile(arr, 50):.3f}  |  <=0.02em: {(arr <= 0.02).mean():.1%}  "
        f"<=0.05em: {(arr <= 0.05).mean():.1%}   (framing cannot change these)"
    )
    print(f"{'framing':<38}{'px/em':>7}{'p5':>7}{'p50':>7}{'min':>7}{'<8px':>8}")
    if sides is None:
        for side in (2.00, 1.80, 1.60, 1.50, 1.40, 1.30, 1.20):
            px = arr * CANVAS_PX / side
            print(
                f"{'fixed window side %.2f em' % side:<38}{CANVAS_PX / side:>7.0f}"
                f"{np.percentile(px, 5):>7.1f}{np.median(px):>7.1f}{px.min():>7.1f}"
                f"{(px < ACUITY_PX).mean():>8.1%}"
            )
    else:
        for margin in (0.0, 0.08):
            denom = np.maximum(np.asarray(sides, dtype=float) + margin, 0.2)
            px = arr * CANVAS_PX / denom
            print(
                f"{'per-font envelope (+%.0f%% margin)' % (margin * 100):<38}"
                f"{'varies':>7}{np.percentile(px, 5):>7.1f}{np.median(px):>7.1f}"
                f"{px.min():>7.1f}{(px < ACUITY_PX).mean():>8.1%}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts"),
    )
    parser.add_argument(
        "--glob",
        default="ofl/*/*.ttf",
        help="Glob under --repo. GoogleFonts (and training) only sees ofl/*/*.ttf.",
    )
    parser.add_argument("--limit", type=int, default=None, help="stop after N fonts")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--pilot-anchors",
        action="store_true",
        help="measure only the content detector's current anchors",
    )
    parser.add_argument(
        "--quantiles",
        default="0.5,0.8,0.9,0.95,0.99",
        help="comma-separated quantiles to report",
    )
    args = parser.parse_args()

    qs = tuple(float(q) for q in args.quantiles.split(","))
    codepoints = _codepoints(args.pilot_anchors)
    label = "pilot anchors" if args.pilot_anchors else "GF Latin Core"
    paths = sorted(Path(args.repo).glob(args.glob))
    if args.limit:
        paths = paths[: args.limit]
    print(
        f"repo: {args.repo}\ncodepoints: {len(codepoints)} ({label})\n"
        f"fonts found: {len(paths)}"
    )

    jobs = max(1, args.jobs)
    calls = [(str(p), tuple(codepoints)) for p in paths]
    if jobs == 1:
        results = [_extents_call(call) for call in calls]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(_extents_call, calls))

    pooled: dict[str, list[float]] = {k: [] for k in ("xMin", "xMax", "yMin", "yMax")}
    envelopes: dict[str, list[float]] = {k: [] for k in ("xMin", "xMax", "yMin", "yMax")}
    cues: list[float] = []
    envelope_sides: list[float] = []
    n_ok = 0
    worst: list[tuple[float, str]] = []
    for path, result in zip(paths, results):
        if not result or not result["boxes"]:
            continue
        n_ok += 1
        arr = np.asarray(result["boxes"], dtype=float)  # (N, 4) xmin, ymin, xmax, ymax
        pooled["xMin"].extend(arr[:, 0])
        pooled["yMin"].extend(arr[:, 1])
        pooled["xMax"].extend(arr[:, 2])
        pooled["yMax"].extend(arr[:, 3])
        env = (arr[:, 0].min(), arr[:, 1].min(), arr[:, 2].max(), arr[:, 3].max())
        envelopes["xMin"].append(env[0])
        envelopes["yMin"].append(env[1])
        envelopes["xMax"].append(env[2])
        envelopes["yMax"].append(env[3])
        if result["cue"] is not None:
            cues.append(result["cue"])
            envelope_sides.append(max(env[2] - env[0], env[3] - env[1]))
        worst.append((float(env[2] - env[0]), path.name))

    print(f"fonts measured: {n_ok}   glyphs: {len(pooled['xMin']):,}")

    _print_quantile_table(f"pooled glyph bboxes (em), {label}", pooled, qs)
    _print_quantile_table(
        f"per-font {label} envelope (em), {n_ok} fonts", envelopes, qs
    )

    # --- comparison with the current window --------------------------------- #
    w = CURRENT_WINDOW
    print(
        f"\ncurrent window: x [{w['left']:+.2f}, {w['right']:+.2f}]  "
        f"y [{w['bottom']:+.2f}, {w['top']:+.2f}]  "
        f"({w['right'] - w['left']:.2f} x {w['top'] - w['bottom']:.2f} em)"
    )
    fits = sum(
        1
        for i in range(n_ok)
        if envelopes["xMin"][i] >= w["left"]
        and envelopes["xMax"][i] <= w["right"]
        and envelopes["yMin"][i] >= w["bottom"]
        and envelopes["yMax"][i] <= w["top"]
    )
    print(f"whole envelopes inside it: {fits}/{n_ok} = {fits / n_ok:.3f}")

    def window_for(keep: float):
        """Marginal quantile window: covers ``keep`` of the envelopes per dimension."""
        low = 1.0 - keep
        left = float(np.percentile(envelopes["xMin"], low * 100))
        bottom = float(np.percentile(envelopes["yMin"], low * 100))
        right = float(np.percentile(envelopes["xMax"], keep * 100))
        top = float(np.percentile(envelopes["yMax"], keep * 100))
        return left, right, bottom, top, max(right - left, top - bottom)

    print("\nmarginal quantile windows (per dimension -- NOT joint containment):")
    print(f"{'keep':>6}  {'x needed':>24}  {'y needed':>24}  {'side':>6}  {'gain':>6}")
    for keep in (0.80, 0.90, 0.95, 0.99):
        left, right, bottom, top, side = window_for(keep)
        print(
            f"{keep:>5.0%}  [{left:+.3f}, {right:+.3f}] {right - left:>6.3f}  "
            f"[{bottom:+.3f}, {top:+.3f}] {top - bottom:>6.3f}  "
            f"{side:>6.3f}  {2.0 / side:>5.2f}x"
        )

    env = np.column_stack(
        [envelopes["xMin"], envelopes["yMin"], envelopes["xMax"], envelopes["yMax"]]
    )

    def best_containment(side: float) -> tuple[int, float, float]:
        """(count, x0, y0): the square window of ``side`` holding most envelopes.

        Each font's envelope constrains the window origin ``(x0, y0)`` to a
        rectangle, so scanning a coarse origin grid finds (near) the best
        placement — a per-font frame is *font-constant*, but where it sits in the
        2em canvas is free.
        """
        xmin, ymin, xmax, ymax = env[:, 0], env[:, 1], env[:, 2], env[:, 3]
        best = (0, 0.0, 0.0)
        for x0 in np.arange(-1.6, 0.6, 0.04):
            fits_x = (xmax - x0 <= side) & (xmin >= x0)
            if not fits_x.any():
                continue
            low_y, high_y = ymin[fits_x], ymax[fits_x]
            for y0 in np.arange(-1.6, 0.6, 0.04):
                count = int(((high_y - y0 <= side) & (low_y >= y0)).sum())
                if count > best[0]:
                    best = (count, float(x0), float(y0))
        return best

    print("\njoint containment vs square window size (best placement):")
    print(f"{'side':>6}  {'gain':>6}  {'fonts inside':>13}   placement (x0, y0)")
    sides = (1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.80, 2.00)
    measured = {}
    for side in sides:
        count, x0, y0 = best_containment(side)
        measured[side] = (count, x0, y0)
        print(
            f"{side:>6.2f}  {2.0 / side:>5.2f}x  {count / len(env):>13.3f}   "
            f"({x0:+.2f}, {y0:+.2f})"
        )

    target = 0.90
    side = next((s for s in sides if measured[s][0] / len(env) >= target), max(sides))
    count, x0, y0 = measured[side]
    print(
        f"\nsmallest fixed window keeping >={target:.0%} of fonts: {side:.2f} em  "
        f"({2.0 / side:.2f}x more px/em), {count / len(env):.1%} inside\n"
        f"  window x [{x0:+.3f}, {x0 + side:+.3f}]   y [{y0:+.3f}, {y0 + side:+.3f}]\n"
        f"  origin (x=0) {(-x0) / side:.1%} from the left;  "
        f"baseline {(y0 + side) / side:.1%} from the top"
    )

    if cues:
        _cue_table(
            f"o/O cue in pixels at 128x128 ({len(cues)} fonts with both glyphs)",
            cues,
            None,
        )
        _cue_table("same cue, per-font envelope framing", cues, envelope_sides)

    print("\nwidest-envelope fonts (the ones any tightening deliberately clips):")
    for span, name in sorted(worst, reverse=True)[:8]:
        print(f"   {span:5.2f} em  {name}")


if __name__ == "__main__":
    main()
