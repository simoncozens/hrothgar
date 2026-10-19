#!/usr/bin/env python
"""Scan the Google Fonts catalogue for out-of-style glyphs.

For every font file, render each visible glyph (uniform scale-and-pad, matching
the model's training framing), embed it with the instance-membership model, and
score it leave-one-out against the rest of the font's glyphs.  A low score
means the glyph does not look like it belongs to that font instance's style —
a QA signal.

The "score" is the calibrated in-style probability from
:class:`~hrothgar.set_membership.calibration.ConfidenceScorer` (0 = certainly
out-of-style, 1 = certainly in-style).  Rows are written only for glyphs whose
score is below ``--threshold``.

Output is a CSV with columns ``instance``, ``character``, ``score``.  The scan
uses the same ``should_skip`` policy as training (no Noto, small-caps/guide,
symbol/barcode/redaction, or CJK fonts).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import unicodedata
from pathlib import Path

import torch

from hrothgar.googlefonts import GoogleFont, GoogleFonts
from hrothgar.render import render_glyph_uniform
from hrothgar.set_membership.calibration import (
    ConfidenceScorer,
    PlattCalibrator,
    collect_scores,
)
from hrothgar.set_membership.config import SetMembershipConfig
from hrothgar.set_membership.dataset import SetMembershipDatasetMaker
from hrothgar.set_membership.evaluation import build_membership_panel
from hrothgar.set_membership.model import SetMembershipModel
from hrothgar.utils import pick_device

# Unicode categories that render as blank (separators, control, format, ...).
_INVISIBLE_CATEGORIES = frozenset({"Zs", "Zl", "Zp", "Cc", "Cf", "Cs", "Co", "Cn"})


def visible_codepoints(font: GoogleFont) -> list[int]:
    """Visible (non-blank) codepoints in ``font``, sorted."""
    out = []
    for cp in font.codepoints:
        try:
            category = unicodedata.category(chr(cp))
        except ValueError:
            continue
        if category in _INVISIBLE_CATEGORIES:
            continue
        out.append(cp)
    return sorted(out)


def instance_name(font: GoogleFont) -> str:
    """Canonical instance name (full name from METADATA, else the file stem)."""
    for entry in font.metadata.fonts:
        if entry.filename == font.path.name:
            return entry.full_name or entry.post_script_name or font.path.stem
    return font.path.stem


def render_visible_glyphs(
    font: GoogleFont, codepoints: list[int], size: int
) -> tuple[list[torch.Tensor], list[int], int]:
    """Render visible glyphs, skipping blank (missing) glyphs.

    Returns ``(images, kept_codepoints, n_blank)``.
    """
    images: list[torch.Tensor] = []
    kept: list[int] = []
    n_blank = 0
    for cp in codepoints:
        try:
            image, _geom = render_glyph_uniform(font, cp, size)
        except Exception:
            continue
        if bool((image >= 1.0).all()):
            n_blank += 1
            continue
        images.append(image.unsqueeze(0))
        kept.append(cp)
    return images, kept, n_blank


def embed(model, images: list[torch.Tensor], device):
    """Embed a list of glyphs in chunks, returning ``(N, D)``."""
    chunks = []
    for start in range(0, len(images), 64):
        imgs = torch.stack(images[start : start + 64]).to(device)
        with torch.no_grad():
            chunks.append(model(imgs).cpu())
    return torch.cat(chunks)


def load_scorer(
    model: SetMembershipModel,
    repo: str,
    config: SetMembershipConfig,
    device,
    calibration_path: str,
    calibration_instances: int,
) -> ConfidenceScorer:
    """Load a ``ConfidenceScorer``, fitting + caching calibration if missing."""
    path = Path(calibration_path)
    if path.exists():
        print(f"loading calibration from {path}")
        return ConfidenceScorer.from_calibration_file(str(path))

    print("no calibration sidecar found; fitting from a held-out panel...")
    maker = SetMembershipDatasetMaker(repo, batch_size=64, config=config)
    panel = build_membership_panel(
        maker.test_instances, config, max_instances=calibration_instances
    )
    scores = collect_scores(model, panel, device)
    calibrator = PlattCalibrator.fit(scores.in_score, scores.out_score)
    scorer = ConfidenceScorer(calibrator)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "scale": calibrator.scale,
                "bias": calibrator.bias,
                "std_floor": scorer.std_floor,
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")
    print(
        f"fitted scale={calibrator.scale:.3f} bias={calibrator.bias:.3f}; "
        f"saved {path}"
    )
    return scorer


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repo",
        default=os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts"),
    )
    p.add_argument("--model-path", default="models/set_membership.pth")
    p.add_argument(
        "--calibration",
        default=None,
        help="calibration JSON (default: <model>.calibration.json)",
    )
    p.add_argument(
        "--calibration-instances",
        type=int,
        default=400,
        help="panel size if the calibration sidecar has to be fitted",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="flag glyphs whose in-style score is below this",
    )
    p.add_argument("--min-glyphs", type=int, default=3)
    p.add_argument("--limit", type=int, default=None, help="cap fonts scanned (canary)")
    p.add_argument("--max-glyphs-per-font", type=int, default=None)
    p.add_argument(
        "--latin-only",
        action="store_true",
        help="restrict to the model's training codepoints",
    )
    p.add_argument("--out", default="set_membership_qa.csv")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    device = torch.device(args.device) if args.device else pick_device()

    config = SetMembershipConfig.from_sidecar(args.model_path)
    model = SetMembershipModel(config)
    model.load(args.model_path, device)
    model.to(device)
    model.eval()
    size = config.glyph_size

    calibration_path = args.calibration or args.model_path.replace(
        ".pth", ".calibration.json"
    )
    scorer = load_scorer(
        model, args.repo, config, device, calibration_path, args.calibration_instances
    )

    fonts = GoogleFonts(args.repo, max_fonts=args.limit).fonts
    train_cps = set(config.input_codepoints)
    print(f"scanning {len(fonts)} fonts (threshold={args.threshold})")

    n_fonts = n_glyphs = n_flagged = n_blank = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["instance", "character", "score"])
        for font in fonts:
            cps = visible_codepoints(font)
            if args.latin_only:
                cps = [cp for cp in cps if cp in train_cps]
            if args.max_glyphs_per_font:
                cps = cps[: args.max_glyphs_per_font]
            if len(cps) < args.min_glyphs:
                continue

            images, kept, blanks = render_visible_glyphs(font, cps, size)
            if len(kept) < args.min_glyphs:
                continue

            embeddings = embed(model, images, device)
            scores = scorer.score_leave_one_out(embeddings)
            name = instance_name(font)
            for cp, score in zip(kept, scores.tolist()):
                n_glyphs += 1
                if score < args.threshold:
                    writer.writerow([name, chr(cp), f"{score:.6f}"])
                    n_flagged += 1
            n_fonts += 1
            n_blank += blanks
            if n_fonts % 50 == 0:
                print(
                    f"  {n_fonts} fonts, {n_glyphs} glyphs, {n_flagged} flagged",
                    flush=True,
                )

    print(
        f"\ndone: {n_fonts} fonts, {n_glyphs} glyphs, "
        f"{n_blank} blank skipped, {n_flagged} flagged"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
