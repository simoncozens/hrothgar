#!/usr/bin/env python
"""Build the glyph review rig.

Writes, into the directory that will be served by ``python3 -m http.server``:

* ``review_manifest.json`` -- the static structure: every family (in
  alphabetical order) with its Google-Fonts name/style/weight and the list of
  generated glyphs (codepoint, image path, whether the font has a ground-truth
  glyph, and the triage score when available).
* ``review_state.json`` -- the mutable state the rig loads at start and
  downloads on button push: ``{"<family>/U+XXXX": "<category>"}``.  This is
  pre-filled with ``acceptable`` for every glyph the triage script accepted.
  An existing file is **merged into**, not overwritten, so labels you have
  already downloaded and put back are preserved (use ``--force`` to discard
  them).
* ``index.html`` / ``app.js`` / ``style.css`` -- copied from
  ``scripts/labeling/review_rig/``.

The manifest is derived from ``outputs/generated/<family>/*.png`` +
``.instances.json``; the triage pre-fill comes from the triage script's
``manifest.jsonl`` (pass ``--triage-manifest`` if it lives elsewhere).  A
missing or partial triage manifest is fine -- those glyphs simply start
untagged.

Ground-truth presence (``has_gt``) is read from each font's cmap, so it does
not depend on the triage manifest being complete.  The variable-font axis
position does not matter here: the cmap is shared across axes.

Usage::

    python3 scripts/labeling/build_review_rig.py
    python3 scripts/labeling/build_review_rig.py \\
        --triage-manifest outputs/labels/manifest.jsonl
    python3 scripts/labeling/build_review_rig.py --force   # reset labels

Then::

    cd outputs/generated && python3 -m http.server 8000
    # open http://localhost:8000/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hrothgar.googlefonts import StandaloneFont  # noqa: E402
from triage_identical import load_instance_map  # noqa: E402

CATEGORIES = [
    "acceptable",
    "ambiguous",
    "bad-skeleton",
    "wrong-style",
    "bad-execution",
    "wrong-content",
]

_CP_RE = re.compile(r".*-U\+([0-9A-Fa-f]{4,6})\.png$")


def load_triage(path: Path | None) -> dict[str, dict]:
    """``file -> triage record`` from the triage script's JSONL manifest."""
    records: dict[str, dict] = {}
    if path is None or not path.exists():
        print(f"  (no triage manifest at {path}; nothing pre-filled)")
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


def load_seed_labels(path: Path, family_key: str) -> dict[str, str]:
    """``<family>/U+XXXX -> category`` from a ``mine.jsonl``-style label file.

    The file is a stream of JSON objects (``codepoint_hex`` + ``final``), as
    produced while calibrating the triage metric.
    """
    labels: dict[str, str] = {}
    text = path.read_text()
    decoder = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        obj, i = decoder.raw_decode(text, i)
        hexcp = obj.get("codepoint_hex")
        final = obj.get("final")
        if hexcp and final in CATEGORIES:
            labels[f"{family_key}/{hexcp}"] = final
    return labels


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--instances",
        type=Path,
        default=Path("models/fontid_diffusion.pth.instances.json"),
        help=".instances.json sidecar (family/style/weight/path per instance)",
    )
    p.add_argument(
        "--repo",
        type=Path,
        default=Path(
            os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts")
        ),
        help="Google Fonts checkout root, for has_gt checks",
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/generated"),
        help="Generated-glyph tree; the rig is written here too",
    )
    p.add_argument(
        "--triage-manifest",
        type=Path,
        default=Path("outputs/labels/manifest.jsonl"),
        help="Triage script JSONL manifest used to pre-fill acceptable glyphs",
    )
    p.add_argument(
        "--seed-file",
        type=Path,
        default=None,
        help="Optional mine.jsonl-style hand labels to seed the state with",
    )
    p.add_argument(
        "--seed-family",
        type=str,
        default=None,
        help="Family key the --seed-file labels belong to",
    )
    p.add_argument(
        "--rig-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "review_rig",
        help="Directory containing index.html / app.js / style.css",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild review_state.json from scratch, discarding downloaded labels",
    )
    p.add_argument(
        "--limit-families",
        type=int,
        default=None,
        help="Only process this many families (for a smoke test)",
    )
    args = p.parse_args()

    instances = load_instance_map(args.instances)
    triage = load_triage(args.triage_manifest)

    families: list[dict] = []
    prefill: dict[str, str] = {}
    n_glyphs = 0

    family_dirs = sorted(d for d in args.input_dir.iterdir() if d.is_dir())
    for fi, family_dir in enumerate(family_dirs):
        if args.limit_families is not None and fi >= args.limit_families:
            break

        key = family_dir.name
        inst = instances.get(key)
        if inst is None:
            print(f"  skip {key}: not in instances.json")
            continue

        try:
            font = StandaloneFont(args.repo / inst["path"])
        except Exception as exc:  # missing file / unreadable face
            print(f"  warn {key}: cannot open font ({exc})")
            font = None

        pngs: list[tuple[int, Path]] = []
        for png in family_dir.glob("*.png"):
            match = _CP_RE.match(png.name)
            if match is not None:
                pngs.append((int(match.group(1), 16), png))
        pngs.sort()
        if not pngs:
            continue

        glyphs: list[dict] = []
        for cp, png in pngs:
            rel = png.relative_to(args.input_dir).as_posix()
            has_gt: bool | None
            if font is None:
                has_gt = None
            else:
                try:
                    has_gt = bool(font.has_codepoint(cp))
                except Exception:
                    has_gt = None

            entry: dict = {"cp": cp, "file": rel, "has_gt": has_gt}
            record = triage.get(rel)
            if record is not None:
                entry["triage"] = record.get("decision")
                for field in ("iou", "hausdorff"):
                    if record.get(field) is not None:
                        entry[field] = record[field]
                if record.get("decision") == "acceptable":
                    prefill[f"{key}/U+{cp:04X}"] = "acceptable"
            glyphs.append(entry)
            n_glyphs += 1

        families.append(
            {
                "key": key,
                "family": inst["family"],
                "style": inst.get("style", "normal"),
                "weight": int(inst.get("weight", 400)),
                "glyphs": glyphs,
            }
        )

        if (fi + 1) % 100 == 0:
            print(f"  {fi + 1} families / {n_glyphs} glyphs…")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "categories": CATEGORIES,
        "families": families,
    }
    manifest_path = args.input_dir / "review_manifest.json"
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))
    size_mb = manifest_path.stat().st_size / 1e6
    print(
        f"wrote {manifest_path} "
        f"({len(families)} families, {n_glyphs} glyphs, {size_mb:.1f} MB)"
    )

    state_path = args.input_dir / "review_state.json"
    existing: dict[str, str] = {}
    if state_path.exists() and not args.force:
        try:
            existing = json.loads(state_path.read_text())
        except json.JSONDecodeError:
            print(f"  warn: {state_path} is not valid JSON; starting fresh")
    merged = dict(existing)

    seed_labels: dict[str, str] = {}
    if args.seed_file is not None:
        if args.seed_family is None:
            raise SystemExit("--seed-file requires --seed-family")
        seed_labels = load_seed_labels(args.seed_file, args.seed_family)
        print(f"seed labels: {len(seed_labels)} from {args.seed_file}")

    # Existing downloaded labels win; then hand-seeded labels; then triage.
    for key, value in seed_labels.items():
        merged.setdefault(key, value)
    for key, value in prefill.items():
        merged.setdefault(key, value)

    state_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {state_path} "
        f"({len(merged)} labels, {len(prefill)} triage pre-fills, "
        f"{len(seed_labels)} seeded)"
    )

    for name in ("index.html", "app.js", "style.css"):
        shutil.copyfile(args.rig_dir / name, args.input_dir / name)
    print(f"copied rig assets into {args.input_dir}")


if __name__ == "__main__":
    main()
