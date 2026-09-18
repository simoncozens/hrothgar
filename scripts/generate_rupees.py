#!/usr/bin/env python
"""Generate missing rupee glyphs (U+20B9) for the font instances a checkpoint
was trained on.

Given a trained factorized (codepoint, font-instance) diffusion checkpoint, this
walks every training instance it recorded, skips instances whose font already
contains the rupee, and for the rest:

  * samples a rupee glyph and predicts its geometry (the five em-unit labels), then
  * writes **two** images per rupee:
      - a 512x512 crop-to-ink raster (upscaled from the diffusion output via the
        super-resolution model) with a one-pixel white border, for vectorization;
      - an evaluation image rendering the string ``ABC5$₹`` on a single canvas,
        baseline-aligned and spaced by each glyph's advance width (the ₹ is the
        generated glyph, the rest are rendered from the font).
  * and writes a single ``geometry.json`` mapping each output to its geometry.

The checkpoint needs the ``.codepoints.json`` and ``.instances.json`` sidecars
written by the training loop, plus ``--repo`` to resolve the repo-relative font
paths (defaults to ``$GOOGLE_FONTS_REPO``).  ``--upscaler-path`` is the
super-resolution checkpoint (with its ``.conf.json`` sidecar).
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
from hrothgar.render_utils import render_glyph_with_geometry
from hrothgar.upscaler.model import UpscalerConfig, UpscalerModel
from hrothgar.utils import pick_device

RUPEE = ord("\u20b9")  # U+20B9
EVAL_STRING = "ABC5$\u20b9"  # "ABC5$₹"


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_grayscale(image: np.ndarray, path: Path) -> None:
    """Save a [0, 1] ink=0 / white=1 array as a grayscale PNG."""
    from PIL import Image

    Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").save(
        path
    )


def _resize(image: np.ndarray, w_px: int, h_px: int) -> np.ndarray:
    """Resize a [0, 1] (H, W) ink array to (h_px, w_px) with bilinear."""
    from PIL import Image

    pil = Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    return (
        np.asarray(pil.resize((w_px, h_px), Image.BILINEAR), dtype=np.float32) / 255.0
    )


def _add_white_border(image: np.ndarray, border: int = 1) -> np.ndarray:
    """Shrink ``image`` to leave a ``border``-px white ring, keeping its size.

    The crop-to-ink glyph fills the square (ink touches the edges); this insets
    it so the outermost pixels are white, which vectorizers prefer.
    """
    h, w = image.shape
    inner_h = max(1, h - 2 * border)
    inner_w = max(1, w - 2 * border)
    inner = _resize(image, inner_w, inner_h)
    canvas = np.ones((h, w), dtype=np.float32)
    canvas[border : h - border, border : w - border] = inner
    return canvas


def _resolve(repo: Path | None, path: str) -> Path:
    p = Path(path)
    if p.is_absolute() or repo is None:
        return p
    return repo / p


def _compose_string(
    glyphs: list[tuple[np.ndarray, dict[str, float]]],
    ppm: int,
    ascender_em: float = 1.5,
    descender_em: float = 0.5,
    origin_x_em: float = 0.5,
    right_margin_em: float = 0.5,
) -> np.ndarray:
    """Compose a list of ``(crop_to_ink_image, geometry)`` onto one canvas.

    Each glyph is baseline-aligned (``baseline_offset = scale_y - descender_depth``)
    and the pen advances by the glyph's advance width.  Returns a ``(H, W)``
    grayscale array in ``[0, 1]`` (0 = ink, 1 = white).
    """
    total_advance = sum(float(g["advance"]) for _, g in glyphs)
    canvas_h = max(1, round((ascender_em + descender_em) * ppm))
    canvas_w = max(1, round((origin_x_em + total_advance + right_margin_em) * ppm))
    canvas = np.ones((canvas_h, canvas_w), dtype=np.float32)

    baseline_y = round(ascender_em * ppm)
    origin_x = round(origin_x_em * ppm)
    pen = origin_x
    for image, g in glyphs:
        scale_x = float(g["scale_x"])
        scale_y = float(g["scale_y"])
        lsb = float(g["left_sidebearing"])
        descender_depth = float(g["descender_depth"])
        advance = float(g["advance"])

        baseline_offset = scale_y - descender_depth
        w_px = max(1, round(scale_x * ppm))
        h_px = max(1, round(scale_y * ppm))
        placed = _resize(image, w_px, h_px)  # (h_px, w_px)

        x0 = pen + round(lsb * ppm)
        y0 = baseline_y - round(baseline_offset * ppm)

        gy, gx = placed.shape
        cy0, cy1 = max(0, y0), min(canvas_h, y0 + gy)
        cx0, cx1 = max(0, x0), min(canvas_w, x0 + gx)
        if cy1 > cy0 and cx1 > cx0:
            canvas[cy0:cy1, cx0:cx1] = placed[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]

        pen += round(advance * ppm)

    return canvas


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained diffusion checkpoint (sidecars derived)",
    )
    p.add_argument(
        "--upscaler-path",
        type=str,
        required=True,
        help="Path to the trained super-resolution checkpoint",
    )
    p.add_argument(
        "--repo",
        type=str,
        default=os.environ.get("GOOGLE_FONTS_REPO"),
        help="Google Fonts repo root, to resolve repo-relative instance paths",
    )
    p.add_argument("--output-dir", type=str, default="outputs/rupees")
    p.add_argument(
        "--ppm",
        type=int,
        default=128,
        help="Pixels-per-em resolution for the evaluation image canvas",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only generate this many missing glyphs (for a quick smoke test)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Sampling seed (DDIM is deterministic, but initial noise is seeded)",
    )
    p.add_argument(
        "--variants",
        type=int,
        default=1,
        help="Independent samples per instance (each uses a distinct seed)",
    )
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
    glyph_size = config.image_size

    # Diffusion model (sampling + geometry).
    model = build_fontid_model(config).to(device)
    model.load(str(model_path), device=device)
    model.eval()

    # Super-resolution model.
    upscaler_cfg = UpscalerConfig.from_sidecar(args.upscaler_path)
    if upscaler_cfg.low_res_size != glyph_size:
        raise SystemExit(
            f"Upscaler low_res_size ({upscaler_cfg.low_res_size}) does not match "
            f"the diffusion image_size ({glyph_size})."
        )
    upscaler = UpscalerModel(upscaler_cfg).to(device)
    upscaler.load(str(args.upscaler_path), device=device)
    upscaler.eval()

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
            device=device,
            dtype=torch.float32,
        )
        for variant in range(args.variants):
            torch.manual_seed(args.seed + 1000 * variant + iid)
            cp = torch.tensor([rupee_idx], device=device, dtype=torch.long)
            with torch.no_grad():
                glyph = model.sample(cp, meta)  # (1, 1, H, W) in [0, 1]
                geometry = model.predict_geometry(cp, meta, glyph)[0].cpu().tolist()
                upscaled = upscaler(glyph)  # (1, 1, 512, 512) in [0, 1]

            geom = dict(zip(GEOMETRY_NAMES, geometry))
            # Recover the absolute baseline position for downstream consumers.
            geom["baseline_offset"] = geom["scale_y"] - geom["descender_depth"]

            image = glyph[0, 0].cpu().numpy()  # (H, W) crop-to-ink, [0, 1]
            up512 = upscaled[0, 0].cpu().numpy()  # (512, 512) crop-to-ink, [0, 1]

            suffix = "" if variant == 0 else f"_v{variant}"
            style = "" if inst["style"] == "normal" else "_italic"
            stem = (
                f"{iid:04d}_{Path(inst['path']).stem}"
                f"{style}_w{inst['weight']}{suffix}"
            )

            # Vectorization image: 512x512 with a 1px white border.
            _save_grayscale(
                _add_white_border(up512, border=1), out_dir / f"{stem}_512.png"
            )

            # Evaluation image: "ABC5$₹" baseline-aligned, advance-spaced.
            glyphs: list[tuple[np.ndarray, dict[str, float]]] = []
            for ch in EVAL_STRING:
                cp = ord(ch)
                if cp == RUPEE:
                    glyphs.append((image, geom))
                elif cp in font.codepoints:
                    try:
                        ref_img, ref_geom = render_glyph_with_geometry(
                            font, cp, glyph_size
                        )
                        glyphs.append((ref_img, ref_geom))
                    except Exception:
                        continue
            eval_canvas = _compose_string(glyphs, args.ppm)
            _save_grayscale(eval_canvas, out_dir / f"{stem}_eval.png")

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
        f"Generated {generated} rupee glyph pairs ({skipped} instances already "
        f"have the rupee) -> {out_dir}\nGeometry written to {json_path}"
    )


if __name__ == "__main__":
    main()
