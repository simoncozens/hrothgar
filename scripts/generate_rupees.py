#!/usr/bin/env python
"""Generate missing rupee glyphs (U+20B9) for the font instances a checkpoint
was trained on.

Given a trained factorized (codepoint, font-instance) diffusion checkpoint, this
walks every training instance it recorded, skips instances whose font already
contains the rupee, and for the rest:

  * samples a rupee glyph and predicts its geometry (the five em-unit labels), and
  * writes the glyph as a denormalized (true-aspect-ratio) PNG, plus a single
    ``geometry.json`` mapping each output to its geometry labels.

The checkpoint needs the ``.codepoints.json`` and ``.instances.json`` sidecars
written by the training loop, plus ``--repo`` to resolve the repo-relative font
paths (defaults to ``$GOOGLE_FONTS_REPO``).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from hrothgar.diffusion.config import FontIdDiffusionConfig
from hrothgar.diffusion.fontid import build_fontid_model
from hrothgar.glyph_rendering import GEOMETRY_NAMES
from hrothgar.googlefonts import StandaloneFont
from hrothgar.utils import pick_device

RUPEE = ord("\u20b9")  # U+20B9


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_grayscale(image: np.ndarray, path: Path) -> None:
    """Save a [0, 1] ink=0 / white=1 array as a grayscale PNG."""
    from PIL import Image

    Image.fromarray(
        (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L"
    ).save(path)


def _denormalize(image: np.ndarray, scale_x: float, scale_y: float, ppm: int) -> np.ndarray:
    """Resize a crop-to-ink square back to its true ink aspect ratio."""
    from PIL import Image

    h_px = max(1, int(round(scale_y * ppm)))
    w_px = max(1, int(round(scale_x * ppm)))
    pil = Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    return np.asarray(pil.resize((w_px, h_px), Image.BILINEAR), dtype=np.float32) / 255.0


def _resolve(repo: Path | None, path: str) -> Path:
    p = Path(path)
    if p.is_absolute() or repo is None:
        return p
    return repo / p


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=str, required=True,
                   help="Path to the trained checkpoint (sidecars are derived from it)")
    p.add_argument("--repo", type=str, default=os.environ.get("GOOGLE_FONTS_REPO"),
                   help="Google Fonts repo root, to resolve repo-relative instance paths")
    p.add_argument("--output-dir", type=str, default="outputs/rupees")
    p.add_argument("--ppm", type=int, default=128,
                   help="PNG resolution in pixels per em")
    p.add_argument("--limit", type=int, default=None,
                   help="Only generate this many missing glyphs (for a quick smoke test)")
    p.add_argument("--seed", type=int, default=0,
                   help="Sampling seed (DDIM is deterministic, but initial noise is seeded)")
    p.add_argument("--variants", type=int, default=1,
                   help="Independent samples per instance (each uses a distinct seed)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()

    model_path = Path(args.model_path)
    config = FontIdDiffusionConfig.from_sidecar(model_path)
    codepoints = _load_json(Path(str(model_path) + ".codepoints.json"))
    instances = _load_json(Path(str(model_path) + ".instances.json"))
    repo = Path(args.repo) if args.repo else None

    if RUPEE not in codepoints:
        raise SystemExit(
            f"Rupee U+{RUPEE:04X} is not in this checkpoint's vocabulary; "
            "retrain with --extra-codepoints '\u20b9'."
        )
    rupee_idx = codepoints.index(RUPEE)

    model = build_fontid_model(config).to(device)
    model.load(str(model_path), device=device)
    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, float]] = {}
    generated = 0
    skipped = 0

    for iid, inst in enumerate(instances):
        font = StandaloneFont(_resolve(repo, inst["path"]))
        if RUPEE in font.codepoints:
            skipped += 1
            continue

        meta = torch.tensor(
            [[inst["family_id"], inst["weight_norm"], inst["style_bucket"]]],
            device=device, dtype=torch.float32,
        )
        for variant in range(args.variants):
            torch.manual_seed(args.seed + 1000 * variant + iid)
            cp = torch.tensor([rupee_idx], device=device, dtype=torch.long)
            with torch.no_grad():
                glyph = model.sample(cp, meta)  # (1, 1, H, W) in [0, 1]
                geometry = model.predict_geometry(cp, meta, glyph)[0].cpu().tolist()
                image = glyph[0, 0].cpu().numpy()  # (H, W) in [0, 1]

            geom = dict(zip(GEOMETRY_NAMES, geometry))
            # Recover the absolute baseline position for any downstream consumer
            # that still expects ``baseline_offset``.
            geom["baseline_offset"] = geom["scale_y"] - geom["descender_depth"]
            suffix = "" if variant == 0 else f"_v{variant}"
            style = "" if inst["style"] == "normal" else "_italic"
            stem = (
                f"{iid:04d}_{Path(inst['path']).stem}"
                f"{style}_w{inst['weight']}{suffix}"
            )
            _save_grayscale(
                _denormalize(image, geom["scale_x"], geom["scale_y"], args.ppm),
                out_dir / f"{stem}.png",
            )
            results[stem] = geom

            generated += 1
            if args.limit is not None and generated >= args.limit:
                break
        if args.limit is not None and generated >= args.limit:
            break

    json_path = out_dir / "geometry.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    print(
        f"Generated {generated} rupee glyphs ({skipped} instances already have the "
        f"rupee) -> {out_dir}\nGeometry written to {json_path}"
    )


if __name__ == "__main__":
    main()
