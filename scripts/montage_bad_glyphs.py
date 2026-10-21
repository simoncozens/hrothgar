#!/usr/bin/env python
"""Render per-family montages of every generated glyph, labeled by the user's
good/bad call and the set_membership model's score.

Each family gets one PNG.  Glyphs are sorted by the model's z-score (most
suspicious first), so:

* user-labelled **bad** glyphs have a red border, **good** a green border;
* the caption shows the codepoint char and the z-score, and the z-score is
  printed red when the model flags it (z < -1) and grey otherwise.

If the model is catching degeneracy, the red-bordered (bad) cells should cluster
at the top-left; green-bordered (good) cells sitting at the top are potential
howlers the user hasn't re-audited.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from hrothgar.googlefonts import GoogleFont
from hrothgar.render import (
    denormalize_uniform,
    geometry_tensor,
)
from hrothgar.set_membership.config import SetMembershipConfig
from hrothgar.set_membership.model import SetMembershipModel
from hrothgar.utils import pick_device

STD_FLOOR = 1e-3
BAD_COLOR = "#d62728"
GOOD_COLOR = "#2ca02c"
FLAG_COLOR = "#d62728"
MUTE_COLOR = "#666666"


def _sanitize(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_") or "family"


def parse_filename(path: Path) -> tuple[str, str, int] | None:
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
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).to(device).unsqueeze(0).unsqueeze(0)
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0]


def loo_scores(embeddings: torch.Tensor) -> torch.Tensor:
    e = F.normalize(embeddings, dim=-1)
    n = e.shape[0]
    if n < 2:
        return torch.zeros(n, device=e.device)
    loo = F.normalize((e.sum(0) - e) / (n - 1), dim=-1)
    return (e * loo).sum(-1)


def pairwise_auc(bad: torch.Tensor, good: torch.Tensor) -> float:
    if bad.numel() == 0 or good.numel() == 0:
        return float("nan")
    return float((bad[:, None] < good[None, :]).float().mean())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="/home/simon/others-repos/fonts")
    p.add_argument("--model-path", default="models/set_membership.pth")
    p.add_argument("--bad-dir", default="bad-glyphs")
    p.add_argument("--generated-dir", default="outputs/generated")
    p.add_argument("--instances", default="models/fontid_diffusion.pth.instances.json")
    p.add_argument("--out-dir", default="outputs/bad_glyph_montages")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    repo = Path(args.repo)
    bad_dir = Path(args.bad_dir)
    gen_dir = Path(args.generated_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device) if args.device else pick_device()

    config = SetMembershipConfig.from_sidecar(args.model_path)
    model = SetMembershipModel(config)
    model.load(args.model_path, device)
    model.to(device).eval()
    size = config.glyph_size

    instances = json.load(open(args.instances, encoding="utf-8"))
    key_to_inst = {(_sanitize(i["family"]), i["style"]): i for i in instances}

    bad_files = [p for p in bad_dir.iterdir() if p.suffix == ".png"]
    bad_set = {p.name for p in bad_files}

    # group generated glyphs by (family_key, style) -> {codepoint: path}
    gen_by_key: dict[tuple[str, str], dict[int, Path]] = {}
    for p in gen_dir.iterdir():
        if p.suffix != ".png":
            continue
        r = parse_filename(p)
        if r is None:
            continue
        family_key, style, cp = r
        gen_by_key.setdefault((family_key, style), {})[cp] = p

    # families that have at least one bad glyph
    families = sorted({parse_filename(p)[:2] for p in bad_files if parse_filename(p)})

    for key in families:
        family_key, style = key
        inst = key_to_inst.get(key)
        if inst is None:
            continue
        style_suffix = "-italic" if style == "italic" else ""
        prefix = f"{family_key}{style_suffix}-U+"

        gen_map = gen_by_key.get(key, {})
        bad_cps = {parse_filename(p)[2] for p in bad_files if parse_filename(p) and parse_filename(p)[:2] == key}
        if not bad_cps:
            continue
        good_cps = [cp for cp in gen_map if cp not in bad_cps]
        if not good_cps:
            continue

        try:
            font = GoogleFont(repo / inst["path"])
        except Exception as exc:
            print(f"!! {inst['family']}: {exc}")
            continue
        # axis = inst.get("axis_position")

        # Generated images are now framed correctly

        # geom_cache: dict[int, torch.Tensor] = {}
        # for cp in list(good_cps) + list(bad_cps):
        #     try:
        #         _, g = render_glyph_with_geometry(font, cp, size, axis_position=axis)
        #         geom_cache[cp] = geometry_tensor(g).to(device)
        #     except Exception:
        #         geom_cache[cp] = torch.zeros(5, device=device)

        def embed(cp: int, path: Path) -> torch.Tensor:
            img = load_glyph(path, size, device).cpu().numpy()[0]  # (size, size)
            framed = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,size,size)
            # framed = denormalize_uniform(img, geom_cache[cp].cpu().tolist(), size)
            # framed = framed.to(device).unsqueeze(0).unsqueeze(0)  # (1,1,size,size)
            with torch.no_grad():
                return model(framed)[0]

        good_emb = torch.stack([embed(cp, gen_map[cp]) for cp in good_cps])
        bad_emb = torch.stack([embed(cp, gen_map[cp]) for cp in bad_cps])
        good_proto = F.normalize(good_emb.mean(0), dim=-1)
        good_cos = loo_scores(good_emb)
        bad_cos = F.normalize(bad_emb, dim=-1) @ good_proto
        mu, sigma = float(good_cos.mean()), float(good_cos.std())

        rows: list[dict] = []
        for cp, path in gen_map.items():
            is_bad = cp in bad_cps
            if is_bad:
                cos = float(bad_cos[list(bad_cps).index(cp)])
            else:
                cos = float(good_cos[good_cps.index(cp)])
            z = (cos - mu) / (sigma + STD_FLOOR)
            rows.append({
                "cp": cp,
                "path": path,
                "is_bad": is_bad,
                "z": z,
                "cos": cos,
            })

        rows.sort(key=lambda r: r["z"])  # most suspicious first

        auc = pairwise_auc(bad_cos, good_cos)
        n = len(rows)
        cols = 8
        nrows = (n + cols - 1) // cols
        fig, axes = plt.subplots(
            nrows, cols, figsize=(cols * 1.15, nrows * 1.15)
        )
        if nrows == 1:
            axes = axes[None, :]
        axes_flat = axes.reshape(-1)

        for ax in axes_flat:
            ax.set_xticks([])
            ax.set_yticks([])

        for i, r in enumerate(rows):
            ax = axes_flat[i]
            arr = np.asarray(Image.open(r["path"]).convert("L"), dtype=np.float32) / 255.0
            # framed = denormalize_uniform(arr, geom_cache[r["cp"]].cpu().tolist(), size)
            ax.imshow(
                arr, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest"
            )
            border = BAD_COLOR if r["is_bad"] else GOOD_COLOR
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(border)
                spine.set_linewidth(4.0)
            ch = chr(r["cp"]) if 32 <= r["cp"] < 0x110000 else f"U+{r['cp']:04X}"
            ztxt = f"{r['z']:+.2f}"
            zcolor = FLAG_COLOR if r["z"] < -1.0 else MUTE_COLOR
            ax.set_title(f"{ch}  {ztxt}", fontsize=7, color=zcolor, pad=2)

        for i in range(n, len(axes_flat)):
            axes_flat[i].axis("off")

        title = (
            f"{inst['family']} ({style})   auc={auc:.2f}   "
            f"bad={len(bad_cps)} good={len(good_cps)}   "
            f"red border=user-bad, green=user-good; caption z-score red when model flags"
        )
        fig.suptitle(title, fontsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        out = out_dir / f"{family_key}{style_suffix}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"wrote {out}  (auc={auc:.2f}, bad={len(bad_cps)}, good={len(good_cps)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
