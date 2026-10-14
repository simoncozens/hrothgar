#!/usr/bin/env python
"""Generate glyph rasters, geometry, and an evaluation preview for the masters
of a single font source, ready for ``hrothgar`` (the combined generate +
vectorize entry point) to trace and insert.

This is the per-source counterpart of ``generate_rupees.py``: instead of
enumerating every instance in a Google Fonts checkout, it generates exactly the
instances a font source's masters correspond to. The font source (any format
Babelfont reads: UFO, designspace, Glyphs) supplies the family name, style, and
each master's user-space weight. The model's conditioning is computed directly
from those values — the family id from the checkpoint's ``.instances.json``
vocabulary, and the style as bucket conditioning (weight is applied post-hoc in
vector space, not by the generator) — so styles the model was *not* trained on
are valid inputs. The checkout
(``--repo``) is only used to render the evaluation preview's reference glyphs;
when no exact instance exists the nearest one (or none) is substituted and only
the preview context degrades, never generation.

For each master it writes:

  * ``{stem}_512.png``          — super-resolved crop-to-ink raster (one-pixel
                                  white border), the image ``hrothgar`` traces;
  * ``{stem}_{size}.png``       — the raw diffusion output at native resolution,
                                  same border, so the super-resolution model's
                                  contribution can be compared directly;
  * ``{stem}_eval.png``         — a preview of ``<eval-string><generated>``,
                                  reference glyphs rendered from the checkout
                                  font, baseline-aligned and advance-spaced;
  * ``geometry.json``           — ``stem ->`` the five em-unit geometry labels
                                  (plus ``baseline_offset``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from hrothgar.diffusion.config import FontIdDiffusionConfig
from hrothgar.diffusion.dataset_fontid import load_or_build_units
from hrothgar.diffusion.fontid import build_fontid_model
from hrothgar.googlefonts import StandaloneFont
from hrothgar.render import GEOMETRY_NAMES, render_glyph_with_geometry
from hrothgar.upscaler.model import UpscalerConfig, UpscalerModel
from hrothgar.utils import pick_device


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_grayscale(image: np.ndarray, path: Path) -> None:
    """Save a ``[0, 1]`` ink=0 / white=1 array as a grayscale PNG."""
    from PIL import Image

    Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").save(
        path
    )


def _resize(image: np.ndarray, w_px: int, h_px: int) -> np.ndarray:
    """Resize a ``[0, 1]`` ``(H, W)`` ink array to ``(h_px, w_px)`` with bilinear."""
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


def _parse_codepoint(text: str) -> int:
    """Parse ``U+20B9``, ``20B9``, or a literal character into a codepoint."""
    t = text.strip()
    low = t.lower()
    if low.startswith("u+"):
        return int(t[2:], 16)
    if len(t) == 1 and not ("0" <= t <= "9" or "a" <= t <= "f"):
        return ord(t)
    return int(t, 16)


def _resolve(repo: Path, path: str) -> Path:
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
    and the pen advances by the glyph's advance width. Returns a ``(H, W)``
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


def _family_id(family_to_id: dict[str, int], family: str) -> int | None:
    """Look up a family's vocabulary id, exact then space-insensitive."""
    if family in family_to_id:
        return family_to_id[family]
    for fam, fid in family_to_id.items():
        if fam.replace(" ", "") == family.replace(" ", ""):
            return fid
    return None


def _find_unit(units, family: str, style: str):
    """Find a checkout sampling unit to render reference glyphs from.

    Exact ``(family, style)`` match first, then a space-insensitive family
    match, then any style for the family. The returned unit only supplies the
    reference font for the evaluation preview; the model's conditioning is
    computed directly from the requested family/style/weight, so an
    approximate (or missing) unit is not an error.
    """
    for u in units:
        if u.family == family and u.style == style:
            return u
    for u in units:
        if u.family.replace(" ", "") == family.replace(" ", "") and u.style == style:
            return u
    for u in units:
        if u.family == family or u.family.replace(" ", "") == family.replace(" ", ""):
            return u
    return None


def _axis_position(axes: list[list], weight: int) -> list[float]:
    """Full design-coordinate list in fvar order, defaulting non-wght axes."""
    return [
        float(min(max(weight, lo), hi)) if tag == "wght" else float(default)
        for tag, lo, default, hi in axes
    ]


def _reference_for(unit, weight: int):
    """Best-effort ``(path, axis_position)`` to render preview reference glyphs.

    Prefers a static instance at exactly ``weight``; otherwise a variable
    instance's location synthesized for ``weight`` (clamped to the axis range);
    otherwise the nearest static weight. Returns ``None`` if the unit has no
    usable instance. This only affects the preview — the model is always
    conditioned on the requested weight/style directly.
    """
    statics = [i for i in unit.instances if not i.variable]
    variables = [i for i in unit.instances if i.variable]
    static_by_weight = {i.weight: i for i in statics}

    inst = static_by_weight.get(weight)
    if inst is not None:
        return inst.path, None

    if variables:
        var = variables[0]
        return var.path, _axis_position(var.axes or [], weight)

    if statics:
        nearest = min(statics, key=lambda i: abs(i.weight - weight))
        return nearest.path, None

    return None


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model-path", type=str, required=True, help="Diffusion checkpoint")
    p.add_argument("--upscaler-path", type=str, required=True, help="SR checkpoint")
    p.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Google Fonts checkout root (resolve repo-relative font paths)",
    )
    p.add_argument("--codepoint", type=str, default="U+20B9")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--family", type=str, required=True)
    p.add_argument("--style", type=str, default="normal")
    p.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Comma-separated user-space weights, one per source master",
    )
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument(
        "--eval-string",
        type=str,
        default="ABC123",
        help="Reference characters to render before the generated glyph in the preview",
    )
    p.add_argument("--ppm", type=int, default=128, help="Preview canvas resolution")
    p.add_argument(
        "--cache",
        type=Path,
        default=Path(os.environ.get("FONT_DB_CACHE", "/tmp/hrothgar_units.json")),
        help="Sampling-units JSON cache for checkout enumeration",
    )
    p.add_argument("--rebuild-cache", action="store_true")
    args = p.parse_args()

    codepoint = _parse_codepoint(args.codepoint)
    torch.manual_seed(args.seed)
    device = pick_device()

    model_path = Path(args.model_path)
    config = FontIdDiffusionConfig.from_sidecar(model_path)
    codepoints = _load_json(Path(str(model_path) + ".codepoints.json"))
    trained = _load_json(Path(str(model_path) + ".instances.json"))
    family_to_id = {inst["family"]: inst["family_id"] for inst in trained}

    if codepoint not in codepoints:
        raise SystemExit(
            f"U+{codepoint:04X} is not in this checkpoint's vocabulary; "
            "retrain with the glyph added to the codepoint set."
        )
    cp_idx = codepoints.index(codepoint)

    family_id = _family_id(family_to_id, args.family)
    if family_id is None:
        raise SystemExit(
            f"Family {args.family!r} is not in this checkpoint's vocabulary "
            "(.instances.json); the model only generates in-corpus families."
        )

    if args.repo is None:
        raise SystemExit(
            "--repo (or $GOOGLE_FONTS_REPO) is required to resolve the checkout "
            "font used for the preview's reference glyphs."
        )
    repo = Path(args.repo)
    weights = [int(w.strip()) for w in args.weights.split(",") if w.strip()]
    if not weights:
        raise SystemExit("--weights produced no masters")

    # The checkout unit only supplies the binary font the preview's reference
    # glyphs are rendered from. The model's conditioning (family id, style
    # bucket) is computed directly from the requested values; weight is not a
    # conditioning input, so the requested weights only affect the preview and
    # output stems — only the preview context degrades when no exact instance
    # exists.
    units = load_or_build_units(repo, set(codepoints), args.cache, args.rebuild_cache)
    unit = _find_unit(units, args.family, args.style)
    if unit is None:
        print(
            f"warning: no checkout font for {args.family!r}/{args.style!r}; "
            "preview will show only the generated glyph",
            file=sys.stderr,
        )
    style_bucket = 0 if args.style == "normal" else 1

    # Diffusion model (sampling + geometry).
    model = build_fontid_model(config).to(device)
    model.load(str(model_path), device=device)
    model.eval()

    # Super-resolution model.
    upscaler_cfg = UpscalerConfig.from_sidecar(args.upscaler_path)
    if upscaler_cfg.low_res_size != config.image_size:
        raise SystemExit(
            f"Upscaler low_res_size ({upscaler_cfg.low_res_size}) does not match "
            f"the diffusion image_size ({config.image_size})."
        )
    upscaler = UpscalerModel(upscaler_cfg).to(device)
    upscaler.load(str(args.upscaler_path), device=device)
    upscaler.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    family_key = args.family.replace(" ", "")
    style_suffix = "" if args.style == "normal" else "_italic"
    eval_prefix = args.eval_string
    glyph_size = config.image_size

    results: dict[str, dict[str, float]] = {}
    for i, weight in enumerate(weights):
        stem = f"{family_key}{style_suffix}_w{weight}"

        # Reference font + axis position for the preview only. ``None`` when
        # the checkout has nothing for this family/style.
        ref = _reference_for(unit, weight) if unit is not None else None
        font = StandaloneFont(_resolve(repo, ref[0])) if ref is not None else None
        axis_position = ref[1] if ref is not None else None

        torch.manual_seed(args.seed + i)
        meta = torch.tensor(
            [[family_id, style_bucket]],
            device=device,
            dtype=torch.float32,
        )
        cp = torch.tensor([cp_idx], device=device, dtype=torch.long)
        with torch.no_grad():
            glyph = model.sample(cp, meta)  # (1, 1, H, W) in [0, 1]
            geometry = model.predict_geometry(cp, meta, glyph)[0].cpu().tolist()
            upscaled = upscaler(glyph)  # (1, 1, 512, 512) in [0, 1]

        geom = dict(zip(GEOMETRY_NAMES, geometry))
        geom["baseline_offset"] = geom["scale_y"] - geom["descender_depth"]

        image = glyph[0, 0].cpu().numpy()  # (H, W) crop-to-ink, [0, 1]
        up512 = upscaled[0, 0].cpu().numpy()  # (512, 512) crop-to-ink, [0, 1]

        _save_grayscale(
            _add_white_border(image, border=1),
            out_dir / f"{stem}_{image.shape[0]}.png",
        )
        _save_grayscale(_add_white_border(up512, border=1), out_dir / f"{stem}_512.png")

        # Evaluation preview: <eval-string> rendered from the checkout font,
        # then the generated glyph, baseline-aligned and advance-spaced.
        glyphs: list[tuple[np.ndarray, dict[str, float]]] = []
        for ch in eval_prefix + chr(codepoint):
            cp_ref = ord(ch)
            if cp_ref == codepoint:
                glyphs.append((image, geom))
            elif font is not None and cp_ref in font.codepoints:
                try:
                    ref_img, ref_geom = render_glyph_with_geometry(
                        font, cp_ref, glyph_size, axis_position=axis_position
                    )
                    glyphs.append((ref_img.detach().cpu().numpy(), ref_geom))
                except Exception:
                    continue
        _save_grayscale(_compose_string(glyphs, args.ppm), out_dir / f"{stem}_eval.png")

        results[stem] = geom

    json_path = out_dir / "geometry.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    print(f"Generated {len(results)} master glyph(s) -> {out_dir}")


if __name__ == "__main__":
    main()
