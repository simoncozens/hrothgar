#!/usr/bin/env python3
"""Cache frozen ``FontStyleEmbedder`` vectors for each strong row's Latin companion.

Reads ``non-latins.resolved.csv`` and, for every row whose ``Latin origin`` is
``Custom`` or ``Designed to match Latin``, renders the companion **Latin** font
and stores its frozen ``FontStyleEmbedder`` summary + projection vectors.

- ``Custom``: the Latin lives in the same font file as the non-Latin glyphs.
- ``Designed to match Latin``: the Latin lives in the referenced family
  (Regular weight, falling back to the family's only/lightest instance).

Outputs (keyed by the non-Latin font filename from the CSV's ``Font`` column):

- ``<out>``            ``dict[anchor] = {"embedding": (256,), "projection": (128,)}``
- ``<out>.json``       audit manifest: ``dict[anchor] = {companion, origin, family}``

Both are idempotent and safe to regenerate; re-running overwrites them.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from hrothgar.googlefonts import StandaloneFont
from hrothgar.style_embedding.config import FontStyleEmbedderConfig
from hrothgar.style_embedding.model import FontStyleEmbedder

STRONG_ORIGINS = {"Custom", "Designed to match Latin"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _key(s: str) -> str:
    return _norm(s).lower()


def build_indexes(repo: Path) -> tuple[dict, dict, dict]:
    """Index the ``ofl`` tree.

    Returns ``(filename_to_path, family_to_paths, fullname_to_path)`` where
    ``family_to_paths`` maps a normalized family name to a list of
    ``(weight, full_name, path)`` tuples.
    """
    from gftools.util.google_fonts import Metadata

    base = repo / "ofl" if (repo / "ofl").is_dir() else repo

    filename_to_path: dict[str, str] = {}
    family_to_paths: dict[str, list[tuple[int, str, str]]] = {}
    fullname_to_path: dict[str, str] = {}

    for pb in sorted(base.glob("*/METADATA.pb")):
        try:
            m = Metadata(str(pb))
        except Exception:
            continue
        family = m.name
        entries: list[tuple[int, str, str]] = []
        for font in m.fonts:
            path = str(pb.parent / font.filename)
            filename_to_path[font.filename] = path
            fullname_to_path[_key(font.full_name)] = path
            entries.append((int(font.weight), font.full_name, path))
        family_to_paths[_key(family)] = entries

    # Also index the actual .ttf files on disk, so a stale/renamed METADATA.pb
    # can't hide a Custom font file.
    for ttf in sorted(base.glob("*/*.ttf")):
        filename_to_path.setdefault(ttf.name, str(ttf))

    return filename_to_path, family_to_paths, fullname_to_path


def pick_regular(entries: list[tuple[int, str, str]]) -> str | None:
    """Pick the Regular (weight 400) instance, else the closest, else None."""
    if not entries:
        return None
    return min(entries, key=lambda e: (abs(e[0] - 400), e[0]))[2]


def resolve_companion(
    row: dict,
    filename_to_path: dict,
    family_to_paths: dict,
    fullname_to_path: dict,
) -> tuple[str | None, str]:
    """Return ``(companion_path, error)`` for one strong row."""
    origin = _norm(row.get("Latin origin", ""))
    font = _norm(row.get("Font", ""))
    family = _norm(row.get("Latin family", ""))
    full_name = _norm(row.get("Latin full name", ""))

    if origin == "Custom":
        path = filename_to_path.get(font)
        if path is None:
            return None, f"Custom font file not found in repo: {font!r}"
        return path, ""

    if origin == "Designed to match Latin":
        if full_name and _key(full_name) in fullname_to_path:
            return fullname_to_path[_key(full_name)], ""
        path = pick_regular(family_to_paths.get(_key(family), []))
        if path is None:
            return None, f"Latin family not found in repo: {family!r}"
        return path, ""

    return None, f"unexpected origin {origin!r}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts"),
        help="Google Fonts repo root, or the ofl/ directory directly",
    )
    parser.add_argument("--csv", default="non-latins.resolved.csv")
    parser.add_argument("--model", default="models/style_embedding_finetune.pth")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="latin_companion_embeddings.pt")
    parser.add_argument(
        "--limit", type=int, default=None, help="process only the first N strong rows"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    repo = Path(args.repo)

    print("Indexing Google Fonts metadata …")
    filename_to_path, family_to_paths, fullname_to_path = build_indexes(repo)
    print(
        f"  {len(filename_to_path)} files, {len(family_to_paths)} families, "
        f"{len(fullname_to_path)} full names"
    )

    cfg = FontStyleEmbedderConfig.from_sidecar(args.model)
    model = FontStyleEmbedder(cfg)
    model.load(args.model, device)
    model.to(device)
    model.eval()
    print(
        f"Loaded {args.model} (glyph_size={cfg.glyph_size}, "
        f"{len(cfg.input_codepoints)} codepoints)"
    )

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, skipinitialspace=True))

    strong = [r for r in rows if _norm(r.get("Latin origin", "")) in STRONG_ORIGINS]
    if args.limit is not None:
        strong = strong[: args.limit]

    cache: dict[str, dict[str, torch.Tensor]] = {}
    manifest: dict[str, dict[str, str]] = {}
    failures: list[tuple[str, str]] = []

    for i, row in enumerate(strong, 1):
        anchor = _norm(row.get("Font", ""))
        origin = _norm(row.get("Latin origin", ""))
        family = _norm(row.get("Family", ""))

        companion, err = resolve_companion(
            row, filename_to_path, family_to_paths, fullname_to_path
        )
        if companion is None:
            failures.append((anchor, err))
            continue

        try:
            font = StandaloneFont(companion)
            with torch.no_grad():
                emb = model.compute_embedding(font, device)
                proj = (
                    F.normalize(
                        model.projection(emb.unsqueeze(0).to(device)), p=2, dim=-1
                    )
                    .squeeze(0)
                    .cpu()
                )
            cache[anchor] = {"embedding": emb.cpu(), "projection": proj}
            manifest[anchor] = {
                "companion": companion,
                "origin": origin,
                "family": family,
            }
        except Exception as exc:  # blank glyph, missing file, render failure …
            failures.append((anchor, f"{type(exc).__name__}: {exc}"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out)
    manifest_path = out.with_suffix(".json")
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"\nCached {len(cache)}/{len(strong)} embeddings -> {out}")
    print(f"Manifest -> {manifest_path}")
    if failures:
        print(f"\n{len(failures)} failures:")
        for anchor, err in failures:
            print(f"  {anchor}: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
