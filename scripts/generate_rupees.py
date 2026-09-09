#!/usr/bin/env python
"""Generate missing rupee glyphs (U+20B9) for the fonts a checkpoint was trained on.

Given a trained factorized (codepoint, font-ID) diffusion checkpoint, this walks
every font it was trained against, skips fonts that already contain the rupee,
and for the rest:

  * samples a rupee glyph and predicts its geometry (the five em-unit labels), and
  * writes the glyph as a denormalized (true-aspect-ratio) PNG, plus a single
    ``geometry.json`` mapping each font path to ``scale_x``, ``scale_y``,
    ``left_sidebearing``, ``baseline_offset``, and ``advance``.

The checkpoint must have been trained with the rupee in its vocabulary (e.g.
``--extra-codepoints '₹'``); it also needs the ``.fonts.json``,
``.codepoints.json`` and ``.font_meta.json`` sidecars written by the training
loop.
"""

from __future__ import annotations

import argparse
import json
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=str, required=True,
                   help="Path to the trained checkpoint (sidecars are derived from it)")
    p.add_argument("--output-dir", type=str, default="outputs/rupees")
    p.add_argument("--ppm", type=int, default=128,
                   help="PNG resolution in pixels per em")
    p.add_argument("--limit", type=int, default=None,
                   help="Only generate this many missing glyphs (for a quick smoke test)")
    p.add_argument("--seed", type=int, default=0,
                   help="Sampling seed (DDIM is deterministic, but initial noise is seeded)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()

    model_path = Path(args.model_path)
    config = FontIdDiffusionConfig.from_sidecar(model_path)
    font_paths = _load_json(Path(str(model_path) + ".fonts.json"))
    codepoints = _load_json(Path(str(model_path) + ".codepoints.json"))
    font_meta = _load_json(Path(str(model_path) + ".font_meta.json"))  # per-font [family, weight, style]

    if len(font_meta) != len(font_paths):
        raise SystemExit(
            f"font_meta ({len(font_meta)}) and fonts ({len(font_paths)}) sidecars disagree; "
            "checkpoint may predate the family/weight/style factorization."
        )

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

    for font_id, path_str in enumerate(font_paths):
        font = StandaloneFont(path_str)
        if RUPEE in font.codepoints:
            skipped += 1
            continue

        cp = torch.tensor([rupee_idx], device=device, dtype=torch.long)
        meta = torch.tensor([font_meta[font_id]], device=device, dtype=torch.float32)  # (1, 3)
        with torch.no_grad():
            image = model.sample(cp, meta)[0, 0].cpu().numpy()  # (H, W) in [0, 1]
            geometry = model.predict_geometry(cp, meta)[0].cpu().tolist()  # 5 em units

        geom = dict(zip(GEOMETRY_NAMES, geometry))
        png_path = out_dir / f"{font_id:04d}_{Path(path_str).stem}.png"
        _save_grayscale(
            _denormalize(image, geom["scale_x"], geom["scale_y"], args.ppm), png_path
        )
        results[path_str] = geom

        generated += 1
        if args.limit is not None and generated >= args.limit:
            break

    json_path = out_dir / "geometry.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    print(
        f"Generated {generated} rupee glyphs ({skipped} already present) -> {out_dir}\n"
        f"Geometry written to {json_path}"
    )


if __name__ == "__main__":
    main()
