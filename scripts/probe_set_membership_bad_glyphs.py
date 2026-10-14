#!/usr/bin/env python
"""Score harvested degenerate glyphs with the instance-membership model.

The question this answers: does ``set_membership`` *already* separate the
hand-picked "bad" generated glyphs from the "good" ones, or do the bad ones slip
through?

For each family that appears in ``bad-glyphs/``, the glyphs in
``outputs/generated/`` that are *not* in ``bad-glyphs/`` are treated as the
family's in-style evidence ("good").  An instance prototype is the mean of the
good glyphs' embeddings; every glyph (good and bad) is then scored against it.

Good glyphs are scored leave-one-out (each good glyph against the mean of the
*other* good glyphs), so the good baseline is not inflated by self-similarity.
Bad glyphs are scored against the full good prototype.  The per-glyph z-score is
``(cosine - mean_good) / std_good``, so a large negative z-score means the bad
glyph sits well below the good distribution — i.e. set_membership catches it.

Both good and bad glyphs share the *font's* geometry for their codepoint, so the
image is the only thing that can separate them.  Generated PNGs are still
crop-to-ink, so each is denormalized into the uniform scale-and-pad framing
(via ``hrothgar.render.denormalize_uniform``) before it is scored.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from hrothgar.googlefonts import GoogleFont
from hrothgar.render import (
    denormalize_uniform,
    geometry_tensor,
    render_glyph_with_geometry,
)
from hrothgar.set_membership.config import SetMembershipConfig
from hrothgar.set_membership.model import SetMembershipModel
from hrothgar.utils import pick_device

STD_FLOOR = 1e-3


def _sanitize(name: str) -> str:
    """Match ``harvest_degenerate_glyphs._sanitize`` (spaces -> underscores)."""
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_") or "family"


def parse_bad_filename(path: Path) -> tuple[str, str, int] | None:
    """Return ``(family_key, style, codepoint)`` from a bad-glyph filename."""
    name = path.name
    if name.startswith("._") or name == ".DS_Store":
        return None
    stem = path.stem
    if "-U+" not in stem:
        return None
    base, cp_hex = stem.rsplit("-U+", 1)
    try:
        cp = int(cp_hex, 16)
    except ValueError:
        return None
    style = "italic" if base.endswith("-italic") else "normal"
    family_key = base[: -len("-italic")] if style == "italic" else base
    return family_key, style, cp


def load_glyph(path: Path, size: int, device) -> torch.Tensor:
    """Load a generated PNG (128x128, ink=0/white=1) as (1, size, size) float."""
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).to(device).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[
        0
    ]  # (1, size, size)


def loo_scores(embeddings: torch.Tensor) -> torch.Tensor:
    """Leave-one-out cosine of each row against the others' mean."""
    e = F.normalize(embeddings, dim=-1)
    n = e.shape[0]
    if n < 2:
        return torch.zeros(n, device=e.device)
    loo = F.normalize((e.sum(0) - e) / (n - 1), dim=-1)
    return (e * loo).sum(-1)


def pairwise_auc(bad: torch.Tensor, good: torch.Tensor) -> float:
    """Fraction of (bad, good) pairs where bad scores strictly below good."""
    if bad.numel() == 0 or good.numel() == 0:
        return float("nan")
    return float((bad[:, None] < good[None, :]).float().mean())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repo",
        default="/home/simon/others-repos/fonts",
        help="Google Fonts repo root",
    )
    p.add_argument(
        "--model-path",
        default="models/set_membership.pth",
        help="Path to the trained set_membership checkpoint",
    )
    p.add_argument("--bad-dir", default="bad-glyphs")
    p.add_argument("--generated-dir", default="outputs/generated")
    p.add_argument(
        "--instances",
        default="models/fontid_diffusion.pth.instances.json",
        help="Diffusion instances sidecar mapping family+style -> font path",
    )
    p.add_argument("--out", default=None, help="write full report as JSON")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    repo = Path(args.repo)
    bad_dir = Path(args.bad_dir)
    gen_dir = Path(args.generated_dir)
    device = torch.device(args.device) if args.device else pick_device()

    # --- Model -----------------------------------------------------------
    config = SetMembershipConfig.from_sidecar(args.model_path)
    model = SetMembershipModel(config)
    model.load(args.model_path, device)
    model.to(device).eval()
    size = config.glyph_size

    # --- Instance resolution --------------------------------------------
    instances = json.load(open(args.instances, encoding="utf-8"))
    key_to_inst = {
        (_sanitize(i["family"]), i["style"]): i for i in instances
    }

    # --- Index generated + bad glyphs -----------------------------------
    bad_files = [p for p in bad_dir.iterdir() if p.suffix == ".png"]
    parsed = [parse_bad_filename(p) for p in bad_files]
    parsed = [x for x in parsed if x is not None]

    # Group bad glyphs by (family_key, style).
    bad_by_key: dict[tuple[str, str], list[tuple[int, Path]]] = {}
    for family_key, style, cp in parsed:
        bad_by_key.setdefault((family_key, style), []).append((cp, next(
            p for p in bad_files if p.stem.startswith(
                f"{family_key}{'-italic' if style == 'italic' else ''}-U+{cp:04X}"
            )
        )))

    # Index generated glyphs by (family_key, style) -> {codepoint: path}.
    gen_by_key: dict[tuple[str, str], dict[int, Path]] = {}
    for p in gen_dir.iterdir():
        if p.suffix != ".png":
            continue
        r = parse_bad_filename(p)
        if r is None:
            continue
        family_key, style, cp = r
        gen_by_key.setdefault((family_key, style), {})[cp] = p

    families = sorted(bad_by_key)

    # --- Score -----------------------------------------------------------
    all_bad_z: list[float] = []
    all_bad_cos: list[float] = []
    all_good_cos: list[float] = []
    per_family: list[dict] = []
    per_glyph: list[dict] = []

    for key in families:
        family_key, style = key
        inst = key_to_inst.get(key)
        if inst is None:
            print(f"!! no instance for {family_key} / {style}; skipping")
            continue
        style_suffix = "-italic" if style == "italic" else ""
        prefix = f"{family_key}{style_suffix}-U+"

        gen_map = gen_by_key.get(key, {})
        bad_cps = {cp for cp, _ in bad_by_key[key]}
        good_cps = [cp for cp in gen_map if cp not in bad_cps]

        if not good_cps:
            print(f"!! no good glyphs for {family_key} / {style}; skipping")
            continue

        try:
            font = GoogleFont(repo / inst["path"])
        except Exception as exc:
            print(f"!! could not open font for {family_key}: {exc}; skipping")
            continue
        axis = inst.get("axis_position")

        # GT geometry per codepoint (shared by good and bad glyphs) so the
        # denormalized image — not the geometry — is the only discriminator.
        geom_cache: dict[int, torch.Tensor] = {}
        for cp in list(good_cps) + list(bad_cps):
            try:
                _, g = render_glyph_with_geometry(font, cp, size, axis_position=axis)
                geom_cache[cp] = geometry_tensor(g).to(device)
            except Exception:
                geom_cache[cp] = torch.zeros(5, device=device)

        def embed(cp: int, path: Path) -> torch.Tensor:
            img = load_glyph(path, size, device).cpu().numpy()[0]  # (size, size)
            framed = denormalize_uniform(img, geom_cache[cp].cpu().tolist(), size)
            framed = framed.to(device).unsqueeze(0).unsqueeze(0)  # (1,1,size,size)
            with torch.no_grad():
                return model(framed)[0]  # (D,)

        good_emb = torch.stack([embed(cp, gen_map[cp]) for cp in good_cps])
        bad_emb = torch.stack([embed(cp, path) for cp, path in bad_by_key[key]])

        good_proto = F.normalize(good_emb.mean(0), dim=-1)
        good_cos = loo_scores(good_emb)
        bad_cos = F.normalize(bad_emb, dim=-1) @ good_proto

        mu, sigma = float(good_cos.mean()), float(good_cos.std())
        bad_z = (bad_cos - mu) / (sigma + STD_FLOOR)

        fam_auc = pairwise_auc(bad_cos, good_cos)
        per_family.append({
            "family": inst["family"],
            "style": style,
            "n_good": int(len(good_cps)),
            "n_bad": int(len(bad_by_key[key])),
            "mean_good_cos": float(good_cos.mean()),
            "std_good_cos": sigma,
            "mean_bad_cos": float(bad_cos.mean()),
            "mean_bad_z": float(bad_z.mean()),
            "auc": fam_auc,
        })

        for i, (cp, path) in enumerate(bad_by_key[key]):
            rec = {
                "family": inst["family"],
                "style": style,
                "codepoint": cp,
                "char": chr(cp) if 32 <= cp < 0x110000 else "?",
                "filename": path.name,
                "cosine": float(bad_cos[i]),
                "z_score": float(bad_z[i]),
            }
            per_glyph.append(rec)
            all_bad_z.append(float(bad_z[i]))
            all_bad_cos.append(float(bad_cos[i]))
        all_good_cos.extend(good_cos.tolist())

    # --- Report ----------------------------------------------------------
    bad_z_t = torch.tensor(all_bad_z)
    caught = int((bad_z_t < -1.0).sum())  # more than 1 sigma below good mean
    print(f"\n{len(per_family)} families, {len(all_bad_z)} bad glyphs, "
          f"{len(all_good_cos)} good glyphs")
    print(f"mean bad z-score: {bad_z_t.mean():.3f}  "
          f"(< -1 caught): {caught}/{len(all_bad_z)}")
    overall_auc = pairwise_auc(torch.tensor(all_bad_cos), torch.tensor(all_good_cos))
    print(f"overall AUC (P(bad < good)): {overall_auc:.3f}\n")

    print(f"{'family':28} {'st':7} {'good':>4} {'bad':>4} {'g_mean':>7} "
          f"{'b_mean':>7} {'b_z':>7} {'auc':>6}")
    for f in per_family:
        print(f"{f['family']:28} {f['style']:7} {f['n_good']:>4} {f['n_bad']:>4} "
              f"{f['mean_good_cos']:>7.3f} {f['mean_bad_cos']:>7.3f} "
              f"{f['mean_bad_z']:>7.2f} {f['auc']:>6.3f}")

    print("\nBad glyphs, worst (most caught) first:")
    print(f"{'z_score':>8} {'cos':>6} {'char':>5} {'family':28} filename")
    for r in sorted(per_glyph, key=lambda r: r["z_score"]):
        print(f"{r['z_score']:>8.2f} {r['cosine']:>6.3f} {r['char']:>5} "
              f"{r['family']:28} {r['filename']}")

    if args.out:
        report = {
            "mean_bad_z": float(bad_z_t.mean()),
            "caught_lt_neg1": caught,
            "n_bad": len(all_bad_z),
            "overall_auc": overall_auc,
            "per_family": per_family,
            "per_glyph": per_glyph,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
