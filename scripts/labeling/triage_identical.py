#!/usr/bin/env python
"""Auto-triage generated glyphs: promote the ones that are a faithful copy of
the font's own ground-truth glyph straight into ``acceptable``.

The generated glyphs under ``outputs/generated/`` were produced by sampling the
factorized diffusion model and placing each crop-to-ink output back on the
baseline with :func:`hrothgar.render.place_glyph` at ``ppm=128`` (a 384x256
canvas, ink=0 / white=1).  For any codepoint the font *already has*, we can
render the ground truth and ask "is the generated glyph essentially the font's
own outline, up to the small shift/scale the geometry predictor introduces?"

The filter is deliberately **one-way**: it only ever promotes a glyph to
``acceptable``, never demotes one, so precision (don't auto-accept a bad glyph)
matters far more than recall (don't auto-accept every good glyph).  Its job is
to lift the unambiguous cases out of the manual queue, not to replace the
labels.

Method
------

For each ``<family>/<family>-U+XXXX.png``:

1. Crop the 384x256 placed canvas to its left 256 columns -- exactly the 2em
   ``UNIFORM_FRAME`` that :func:`~hrothgar.render.render_glyph_uniform` draws at
   ``size=256`` (1 em == 128 px), so generated and ground truth share a scale
   and coordinate system.
2. Render the ground truth with ``render_glyph_uniform(font, cp, 256,
   axis_position=...)`` at the same instance (static file, or the variable
   font's synthesised weight-400 design coords from ``.instances.json``).
3. Binarise both to ink masks, crop each to its ink bounding box, and stretch
   the ground-truth mask to the generated mask's bbox (this absorbs the small
   -- possibly anisotropic -- scale error the geometry head introduces, while a
   separate aspect-ratio gate still rejects a grossly distorted glyph).
4. Find the best integer shift between the two and score two things:
   * **IoU** -- how much of the ink overlaps (global shape agreement), and
   * **Hausdorff** (max symmetric chamfer) -- the largest local disagreement.
     This is what catches a locally-wrong construction that global IoU is blind
     to: a missing counter-dot on ``0``, a differently-shaped ``q`` tail, a
     wrong serif.
5. Tag ``acceptable`` only if *all* of ``iou >= --iou-threshold``,
   ``hausdorff <= --hausdorff-threshold`` and ``|aspect - 1| <=
   --aspect-threshold`` hold.  Codepoints the font lacks (e.g. the rupee) have
   no ground truth and are reported as ``no-gt``.

With ``--calibrate <mine.jsonl> --family <key>`` the script instead re-scores a
single family and prints a precision/recall table over a threshold grid, so you
can pick a boundary from your own hand labels before running the full pass.

The script writes ``manifest.jsonl`` (one decision per glyph, with scores for
auditing) and ``acceptable.txt`` (the promoted paths).  ``--move`` physically
moves the promoted files into ``--accepted-dir/<family>/`` so the remaining
``outputs/generated`` tree is exactly the review queue.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import tqdm
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, zoom
from scipy.signal import correlate2d

from hrothgar.googlefonts import StandaloneFont
from hrothgar.render import render_glyph_uniform

# Ink is rendered near 0.0 on a white (1.0) background.
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
    """``key -> instance``, from the training loop's ``.instances.json`` sidecar.

    Every generated directory corresponds to exactly one sidecar entry (the
    sanitize + style-suffix naming is deterministic), so this mapping is
    lossless -- unlike trying to recover a family name from a sanitised stem.
    """
    with path.open("r", encoding="utf-8") as f:
        instances = json.load(f)
    mapping: dict[str, dict] = {}
    for inst in instances:
        key = key_for(inst)
        if key in mapping:
            raise SystemExit(f"Instance key collision for {key!r}")
        mapping[key] = inst
    return mapping


def ink_mask(img: np.ndarray) -> np.ndarray:
    """Binarise a [0, 1] ink=0 / white=1 image to a boolean ink mask."""
    return img < _INK_THRESHOLD


def _bbox(mask: np.ndarray, pad: int = 2) -> np.ndarray | None:
    """Crop ``mask`` to its ink bounding box (with ``pad`` px of slack)."""
    ys, xs = mask.nonzero()
    if len(xs) == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    h, w = mask.shape
    y0 = max(0, y0 - pad)
    y1 = min(h, y1 + pad + 1)
    x0 = max(0, x0 - pad)
    x1 = min(w, x1 + pad + 1)
    return mask[y0:y1, x0:x1]


def _pad_to(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """Centre ``mask`` in a ``(h, w)`` canvas (assumes mask fits)."""
    out = np.zeros((h, w), dtype=bool)
    y0 = (h - mask.shape[0]) // 2
    x0 = (w - mask.shape[1]) // 2
    out[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]] = mask
    return out


def _shift_into(canvas: np.ndarray, mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Paste ``mask`` into a copy of ``canvas`` shifted by ``(dy, dx)``."""
    out = np.zeros_like(canvas)
    ch, cw = canvas.shape
    mh, mw = mask.shape
    sy0 = max(0, dy)
    sy1 = min(mh, mh + dy)
    dy0 = max(0, -dy)
    sx0 = max(0, dx)
    sx1 = min(mw, mw + dx)
    dx0 = max(0, -dx)
    dy1 = dy0 + (sy1 - sy0)
    dx1 = dx0 + (sx1 - sx0)
    if dy1 > dy0 and dx1 > dx0:
        out[dy0:dy1, dx0:dx1] = mask[sy0:sy1, sx0:sx1]
    return out


def compare(
    gen_arr: np.ndarray, gt_arr: np.ndarray
) -> dict | None:
    """Score a placed generated glyph against its ground-truth uniform render.

    ``gen_arr`` is the full ``(256, 384)`` placed canvas; ``gt_arr`` is the
    ``(256, 256)`` uniform render.  Returns ``None`` if either side is blank.
    """
    gen_ink = ink_mask(gen_arr[:, :256])
    gt_ink = ink_mask(gt_arr)

    gen_bbox = _bbox(gen_ink)
    gt_bbox = _bbox(gt_ink)
    if gen_bbox is None or gt_bbox is None:
        return None

    gh, gw = gen_bbox.shape
    th, tw = gt_bbox.shape
    aspect = (gw / gh) / (tw / th) if tw and th else 1.0
    # Uniform-ish scale (area) between the two ink boxes, for diagnostics.
    area_scale = float(np.sqrt((gw * gh) / (tw * th))) if tw and th else 1.0

    # Stretch the ground-truth mask to the generated mask's bbox.  This absorbs
    # the (possibly anisotropic) scale error the geometry head introduces, so
    # IoU/Hausdorff measure *shape* agreement, not placement scale.
    gt_scaled = (
        zoom(gt_bbox.astype(np.float32), (gh / th, gw / tw), order=1)
        >= _INK_THRESHOLD
    )

    h = max(gh, gt_scaled.shape[0])
    w = max(gw, gt_scaled.shape[1])
    g = _pad_to(gen_bbox, h, w)
    t = _pad_to(gt_scaled, h, w)

    corr = correlate2d(g.astype(np.float32), t.astype(np.float32), mode="full")
    iy, ix = np.unravel_index(np.argmax(corr), corr.shape)
    dy = int(iy - (t.shape[0] - 1))
    dx = int(ix - (t.shape[1] - 1))
    aligned = _shift_into(g, t, dy, dx)

    inter = int((g & aligned).sum())
    union = int((g | aligned).sum())
    iou = inter / union if union else 0.0

    dist_g = distance_transform_edt(~g)
    dist_t = distance_transform_edt(~aligned)
    to_gt = dist_t[g]  # each gen ink pixel -> nearest GT ink pixel
    to_gen = dist_g[aligned]  # each GT ink pixel -> nearest gen ink pixel
    mean_chamfer = float((to_gt.mean() + to_gen.mean()) / 2.0)
    hausdorff = float(max(to_gt.max(), to_gen.max()))

    return {
        "iou": round(iou, 4),
        "hausdorff": round(hausdorff, 2),
        "mean_chamfer": round(mean_chamfer, 2),
        "aspect_ratio": round(aspect, 4),
        "scale": round(area_scale, 4),
        "shift": [dy, dx],
    }


def decide(
    scores: dict | None,
    *,
    iou_threshold: float,
    hausdorff_threshold: float,
    aspect_threshold: float,
) -> str:
    """``acceptable`` only when *every* gate clears (otherwise ``review``)."""
    if scores is None:
        return "review"
    if scores["iou"] < iou_threshold:
        return "review"
    if scores["hausdorff"] > hausdorff_threshold:
        return "review"
    if abs(scores["aspect_ratio"] - 1.0) > aspect_threshold:
        return "review"
    return "acceptable"


def render_gt(font: StandaloneFont, codepoint: int) -> np.ndarray | None:
    """Render the ground truth for ``codepoint`` in the uniform 2em frame.

    Returns the ``(256, 256)`` ink=0 / white=1 array, or ``None`` if the font
    does not contain the codepoint.
    """
    if not font.has_codepoint(codepoint):
        return None
    img, _ = render_glyph_uniform(font, codepoint, 256)
    return img.numpy()


def _load_labels(path: Path) -> dict[str, str]:
    """Read a ``mine.jsonl``-style file (a stream of JSON objects)."""
    text = path.read_text()
    decoder = json.JSONDecoder()
    labels: dict[str, str] = {}
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        obj, i = decoder.raw_decode(text, i)
        labels[obj["codepoint_hex"]] = obj["final"]
    return labels


def calibrate(
    family_key: str,
    inst: dict,
    repo: Path,
    labels: dict[str, str],
    input_dir: Path,
    axis_position,
) -> None:
    """Re-score one family and print precision/recall over a threshold grid."""
    font = StandaloneFont(repo / inst["path"])
    rows = []
    for png in sorted((input_dir / family_key).glob("*.png")):
        m = re.match(r".*-U\+([0-9A-Fa-f]{4,6})\.png$", png.name)
        if m is None:
            continue
        cp = int(m.group(1), 16)
        gt = render_gt(font, cp)
        if gt is None:
            continue
        gen = np.asarray(Image.open(png).convert("L"), dtype=np.float32) / 255.0
        scores = compare(gen, gt)
        if scores is None:
            continue
        hexcp = f"U+{cp:04X}"
        if hexcp not in labels:
            continue
        rows.append((hexcp, labels[hexcp], scores))

    total_acc = sum(1 for r in rows if r[1] == "acceptable")
    print(f"{family_key}: {len(rows)} labelled glyphs ({total_acc} acceptable)\n")
    print("iou>= | haus<= | aspect<= | accept | tp | fp | precision | recall")
    for iou_t in (0.80, 0.85, 0.90, 0.92):
        for haus_t in (2.0, 2.5, 3.0, 4.0):
            for asp_t in (0.10, 0.15, 0.20):
                sel = [
                    r
                    for r in rows
                    if r[2]["iou"] >= iou_t
                    and r[2]["hausdorff"] <= haus_t
                    and abs(r[2]["aspect_ratio"] - 1.0) <= asp_t
                ]
                tp = sum(1 for r in sel if r[1] == "acceptable")
                fp = sum(1 for r in sel if r[1] != "acceptable")
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec = tp / total_acc if total_acc else 0.0
                print(
                    f"  {iou_t:.2f}  | {haus_t:.1f}   | {asp_t:.2f}    | "
                    f"{len(sel):6d} | {tp:2d} | {fp:2d} | {prec:.2f}     | {rec:.2f}"
                )


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
        "--manifest", type=Path, default=Path("outputs/labels/manifest.jsonl"),
        help="Per-glyph decision manifest (JSONL) written here",
    )
    p.add_argument(
        "--acceptable-list", type=Path, default=Path("outputs/labels/acceptable.txt"),
        help="Plain list of auto-accepted glyph paths written here",
    )
    p.add_argument("--iou-threshold", type=float, default=0.85)
    p.add_argument("--hausdorff-threshold", type=float, default=3.0)
    p.add_argument("--aspect-threshold", type=float, default=0.20)
    p.add_argument(
        "--move", action="store_true",
        help="Move auto-accepted files into --accepted-dir instead of only listing them",
    )
    p.add_argument(
        "--accepted-dir", type=Path, default=Path("outputs/acceptable"),
        help="Destination tree for --move",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Only process this many glyphs (for a smoke test)",
    )
    p.add_argument(
        "--families", type=int, default=None,
        help="Only process this many family directories (for a smoke test)",
    )
    p.add_argument(
        "--calibrate", type=Path, default=None,
        help="Re-score a single labelled family (mine.jsonl) and print a "
        "precision/recall grid; use with --family",
    )
    p.add_argument(
        "--family", type=str, default=None,
        help="Family directory key to calibrate against",
    )
    args = p.parse_args()

    repo = Path(args.repo)
    mapping = load_instance_map(args.instances)

    if args.calibrate is not None:
        if args.family is None:
            raise SystemExit("--calibrate requires --family")
        inst = mapping[args.family]
        labels = _load_labels(args.calibrate)
        calibrate(
            args.family,
            inst,
            repo,
            labels,
            args.input_dir,
            axis_position=inst.get("axis_position"),
        )
        return

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.acceptable_list.parent.mkdir(parents=True, exist_ok=True)

    counts = {
        "total": 0,
        "acceptable": 0,
        "review": 0,
        "no-gt": 0,
        "blank": 0,
        "unknown-family": 0,
    }

    family_dirs = sorted(d for d in args.input_dir.iterdir() if d.is_dir())
    manifest_f = args.manifest.open("w", encoding="utf-8")
    acceptable_f = args.acceptable_list.open("w", encoding="utf-8")

    try:
        for fi, family_dir in tqdm.tqdm(enumerate(family_dirs), total=len(family_dirs)):
            if args.families is not None and fi >= args.families:
                break

            key = family_dir.name
            inst = mapping.get(key)
            if inst is None:
                counts["unknown-family"] += sum(1 for _ in family_dir.glob("*.png"))
                continue

            font = StandaloneFont(repo / inst["path"])

            for png in sorted(family_dir.glob("*.png")):
                if args.limit is not None and counts["total"] >= args.limit:
                    break

                m = re.match(r".*-U\+([0-9A-Fa-f]{4,6})\.png$", png.name)
                if m is None:
                    continue
                codepoint = int(m.group(1), 16)

                gt_arr = render_gt(font, codepoint)
                if gt_arr is None:
                    decision = "no-gt"
                    scores = None
                else:
                    gen_arr = np.asarray(
                        Image.open(png).convert("L"), dtype=np.float32
                    ) / 255.0
                    scores = compare(gen_arr, gt_arr)
                    if scores is None:
                        decision = "blank" if not ink_mask(gen_arr[:, :256]).any() else "review"
                    else:
                        decision = decide(
                            scores,
                            iou_threshold=args.iou_threshold,
                            hausdorff_threshold=args.hausdorff_threshold,
                            aspect_threshold=args.aspect_threshold,
                        )

                counts["total"] += 1
                counts[decision] = counts.get(decision, 0) + 1

                rel = png.relative_to(args.input_dir)
                record = {
                    "family": key,
                    "codepoint": codepoint,
                    "codepoint_hex": f"U+{codepoint:04X}",
                    "file": str(rel),
                    "font": inst["path"],
                    "decision": decision,
                }
                if scores is not None:
                    record.update(scores)
                manifest_f.write(json.dumps(record) + "\n")

                if decision == "acceptable":
                    acceptable_f.write(str(rel) + "\n")
                    if args.move:
                        dest = args.accepted_dir / rel
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        png.rename(dest)
    finally:
        manifest_f.close()
        acceptable_f.close()

    print(
        "Triage complete:\n"
        f"  total           {counts['total']}\n"
        f"  acceptable      {counts['acceptable']} "
        f"({100 * counts['acceptable'] / max(1, counts['total']):.1f}%)\n"
        f"  review          {counts['review']}\n"
        f"  no-gt           {counts['no-gt']}\n"
        f"  blank           {counts['blank']}\n"
        f"  unknown-family  {counts['unknown-family']}"
    )
    print(f"Manifest: {args.manifest}")
    print(f"Acceptable list: {args.acceptable_list}")


if __name__ == "__main__":
    main()
