#!/usr/bin/env python
"""Re-frame generated glyphs into the tight critic frame and emit a labelled
dataset manifest.

The critics (``content_detector``, ``set_membership``) are trained on *rendered*
glyphs but evaluated on *generated* ones.  To close that gap we feed them the
generated glyphs directly, in the same tight frame the content detector renders
GT into.  A generated glyph is stored as a ``place_glyph`` canvas (em space at
128 ppm), so re-framing is a pure crop + resize via
:func:`hrothgar.render.crop_placed_to_frame` — no re-render.

For every generated PNG under ``--input-dir`` this script:

* re-frames it into ``--frame`` (default the content detector's tight 1.40em
  window) at ``--size``, writing the result to a parallel tree under
  ``--output-dir``;
* looks up its hand label from ``--state`` (``{family_key/U+XXXX: label}``, the
  ``review_state.json`` the review rig writes) and tags it ``positive``
  (``acceptable``) or ``negative`` (the five defect labels);
* assigns a family-level ``train``/``eval`` split (a family's glyphs are never
  split across the two), so the held-out set is a clean evaluation benchmark.

GT positives are *not* produced here — the critics render those themselves from
the checkout.  This script's output is the generated side: re-framed images plus
a manifest classifying each as positive/negative and train/eval.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image

from hrothgar.render import Frame, crop_placed_to_frame

# The content detector's tight frame (ContentDetectorConfig.frame_left/right/
# bottom/top), a 1.40em square measured by scripts/analyze_glyph_extents.py.
TIGHT_FRAME = "-0.20,1.20,-0.36,1.04"

BAD_LABELS = {
    "ambiguous",
    "bad-execution",
    "bad-skeleton",
    "wrong-content",
    "wrong-style",
}


def parse_frame(spec: str) -> Frame:
    """Parse a ``'left,right,bottom,top'`` em spec into a :class:`Frame`."""
    parts = [float(x) for x in spec.split(",")]
    if len(parts) != 4:
        raise ValueError(f"frame must be 'left,right,bottom,top', got {spec!r}")
    return Frame(*parts)


def role_of(label: str | None) -> str | None:
    if label == "acceptable":
        return "positive"
    if label in BAD_LABELS:
        return "negative"
    return None


def _save(arr: np.ndarray, path: Path) -> None:
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L")
    im.save(path)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input-dir", type=Path, default=Path("outputs/generated"))
    p.add_argument(
        "--state",
        type=Path,
        default=Path("outputs/generated/review_state.json"),
        help="Label map {family_key/U+XXXX: label}",
    )
    p.add_argument("--output-dir", type=Path, default=Path("outputs/generated_tight"))
    p.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Manifest JSONL path (default: <output-dir>/manifest.jsonl)",
    )
    p.add_argument("--size", type=int, default=256)
    p.add_argument(
        "--frame",
        type=str,
        default=TIGHT_FRAME,
        help="'left,right,bottom,top' em frame (default: content detector's tight frame)",
    )
    p.add_argument("--heldout-fraction", type=float, default=0.25)
    p.add_argument("--split-seed", type=int, default=1234)
    p.add_argument(
        "--labelled-only",
        action="store_true",
        help="Only re-frame glyphs that have a label",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--families", type=int, default=None)
    args = p.parse_args()

    state: dict[str, str] = {}
    if args.state.exists():
        state = json.loads(args.state.read_text())
    frame = parse_frame(args.frame)

    # First pass: collect (rel_path, family, cp, label) for every PNG, without
    # loading images.  Family-level split needs the labelled families up front.
    entries: list[tuple[Path, str, int, str | None]] = []
    labelled_families: set[str] = set()
    family_dirs = sorted(d for d in args.input_dir.iterdir() if d.is_dir())
    if args.families is not None:
        family_dirs = family_dirs[: args.families]

    for family_dir in family_dirs:
        family = family_dir.name
        for png in sorted(family_dir.glob("*.png")):
            m = re.match(r".*-U\+([0-9A-Fa-f]{4,6})\.png$", png.name)
            if m is None:
                continue
            cp = int(m.group(1), 16)
            key = f"{family}/U+{cp:04X}"
            label = state.get(key)
            if label is not None:
                labelled_families.add(family)
            entries.append((png, family, cp, label))

    # Deterministic family-level held-out split.
    rng = random.Random(args.split_seed)
    sorted_labelled = sorted(labelled_families)
    n_holdout = int(args.heldout_fraction * len(sorted_labelled))
    heldout = set(rng.sample(sorted_labelled, n_holdout)) if n_holdout else set()

    manifest_path = args.manifest_out or (args.output_dir / "manifest.jsonl")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {"total": 0, "positive": 0, "negative": 0, "unlabelled": 0}
    with manifest_path.open("w", encoding="utf-8") as mf:
        for png, family, cp, label in tqdm.tqdm(entries, total=len(entries)):
            if args.limit is not None and counts["total"] >= args.limit:
                break
            if args.labelled_only and label is None:
                continue

            role = role_of(label)
            if role is None:
                split = None
                counts["unlabelled"] += 1
            else:
                split = "eval" if family in heldout else "train"
                counts[role] += 1
            counts["total"] += 1

            rel = png.relative_to(args.input_dir)
            img = np.asarray(Image.open(png).convert("L"), dtype=np.float32) / 255.0
            tight = crop_placed_to_frame(img, frame, size=args.size)
            out_path = args.output_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            _save(tight, out_path)

            mf.write(
                json.dumps(
                    {
                        "file": str(rel),
                        "family": family,
                        "codepoint": cp,
                        "codepoint_hex": f"U+{cp:04X}",
                        "label": label,
                        "role": role,
                        "split": split,
                    }
                )
                + "\n"
            )

    print(
        "Re-framing complete:\n"
        f"  total      {counts['total']}\n"
        f"  positive   {counts['positive']}\n"
        f"  negative   {counts['negative']}\n"
        f"  unlabelled {counts['unlabelled']}"
    )
    print(f"images -> {args.output_dir}")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
