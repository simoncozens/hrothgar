#!/usr/bin/env python
"""Font-ID holdout canary — does the factorized model *compose*, not just recall?

The recall canary (``diffusion_fontid_canary.py``) trains every ``(font,
codepoint)`` pair, so it only proves the font embedding can carry ROND detail.
This canary holds out a target codepoint across a subset of fonts: the model
sees ``('r', ROND 25/50/75)`` but **never** ``('r', ROND 0)`` or ``('r', ROND
100)``.  Those fonts are still trained on every *other* codepoint, so their
style embedding is learned — only the composition ``r-ness ⊗ font-style`` for
the held-out pairs is unseen.

At eval we ask: given ``(codepoint='r', font=0)`` and ``(codepoint='r',
font=100)``, does the model produce flat terminals vs rounded terminals?  A
``d/o < 1`` on the held-out fonts means it composed style and content; ``d/o =
1.0`` means it fell back to an average ``r`` (no composition).
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from hrothgar.diffusion.config import FontIdDiffusionConfig
from hrothgar.diffusion.dataset import build_fontid_rond_data
from hrothgar.diffusion.fontid import build_fontid_model
from hrothgar.diffusion.losses import diag_off, save_montage
from hrothgar.glyphloss_curvature import CurvatureWeightedGlyphLoss
from hrothgar.googlefonts import StandaloneFont
from hrothgar.gtok.llamagen_lpips import LPIPS
from hrothgar.style_extraction.render_utils import render_glyph


def parse_rond(name: str) -> int:
    m = re.search(r"ROND(\d+)", name)
    if m is None:
        raise ValueError(f"cannot parse ROND value from {name!r}")
    return int(m.group(1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fonts", default="YTM-ROND*.ttf")
    p.add_argument(
        "--glyphs", default="anKMro", help="Codepoints available (as a string)."
    )
    p.add_argument(
        "--holdout-glyph", default="r", help="Codepoint to hold out (default 'r')."
    )
    p.add_argument(
        "--holdout-ronds",
        default="0,100",
        help="Comma-separated ROND values to hold out (>= 2).",
    )
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--timesteps", type=int, default=250)
    p.add_argument("--sampling-timesteps", type=int, default=50)
    p.add_argument("--report-every", type=int, default=200)
    p.add_argument("--montage-dir", default="outputs/canary_fontid_holdout")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    glyphs = sorted({ord(ch) for ch in args.glyphs})
    holdout_cp = ord(args.holdout_glyph)
    if holdout_cp not in glyphs:
        raise SystemExit(
            f"--holdout-glyph {args.holdout_glyph!r} not in --glyphs {args.glyphs!r}"
        )

    pairs = sorted((parse_rond(Path(x).stem), x) for x in glob.glob(args.fonts))
    if not pairs:
        raise SystemExit(f"no fonts matched {args.fonts!r}")
    rond_values = [r for r, _ in pairs]
    fonts = [StandaloneFont(path) for _, path in pairs]
    render_fns = [(lambda cp, size, f=f: render_glyph(f, cp, size)) for f in fonts]

    holdout_ronds = [int(x) for x in args.holdout_ronds.split(",") if x.strip()]
    holdout_font_ids = [rond_values.index(r) for r in holdout_ronds]
    if len(holdout_font_ids) < 2:
        raise SystemExit("--holdout-ronds must name at least 2 fonts (d/o needs >= 2)")
    print(f"Instanced mode: {len(pairs)} fonts; ROND values {rond_values}")
    print(
        f"Holdout: glyph '{chr(holdout_cp)}' held out for ROND {holdout_ronds} "
        f"(font ids {holdout_font_ids})"
    )

    # ---- Full data, then mask out the held-out (font, codepoint) pairs. ----
    images, codepoints, font_ids, cp_to_idx = build_fontid_rond_data(
        render_fns, glyphs, args.image_size
    )
    holdout_cp_idx = cp_to_idx[holdout_cp]
    holdout_set = set(holdout_font_ids)
    is_holdout = (codepoints == holdout_cp_idx) & torch.isin(
        font_ids, torch.tensor(sorted(holdout_set))
    )
    train_mask = ~is_holdout

    images = images[train_mask].to(device)
    codepoints = codepoints[train_mask].to(device)
    font_ids = font_ids[train_mask].to(device)
    # Map each ROND instance to its own family slot (weight/style fixed), so the
    # new factorized style embedding still distinguishes the instances.
    font_meta = torch.stack(
        [
            font_ids.float(),
            torch.zeros_like(font_ids, dtype=torch.float32),
            torch.zeros_like(font_ids, dtype=torch.float32),
        ],
        dim=1,
    )  # (n, 3) = [family_id, weight, style]
    n = len(images)
    print(
        f"Training samples: {n} (after masking {int(is_holdout.sum())} held-out pairs)"
    )

    config = FontIdDiffusionConfig(
        image_size=args.image_size,
        num_codepoints=len(glyphs),
        num_families=len(pairs),
        dim=args.dim,
        timesteps=args.timesteps,
        sampling_timesteps=args.sampling_timesteps,
        learning_rate=args.lr,
    )
    model = build_fontid_model(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    lpips = LPIPS().to(device)
    glyphloss_fn = CurvatureWeightedGlyphLoss(
        k=20.0, lambda_pixel=0.0, lambda_spectral=2.5
    ).to(device)

    gen = torch.Generator(device=device).manual_seed(args.seed)
    eval_seed = args.seed + 9999

    def evaluate() -> tuple[float, torch.Tensor, torch.Tensor]:
        gts, recs = [], []
        for fid in holdout_font_ids:
            gt = render_fns[fid](holdout_cp, args.image_size)
            cp_t = torch.tensor([holdout_cp_idx], device=device)
            meta_t = torch.tensor([[float(fid), 0.0, 0.0]], device=device)
            torch.manual_seed(eval_seed)
            rec = model.sample(cp_t, meta_t)[0, 0].cpu()
            gts.append(gt)
            recs.append(rec)
        gts_t = torch.stack(gts)
        recs_t = torch.stack(recs)
        return diag_off(gts_t, recs_t), gts_t, recs_t

    print(f"{'step':>6} {'loss':>8} {'L1':>8} {'LPIPS':>8} {'glyph':>8} {'d/o':>8}")
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n, (args.batch_size,), generator=gen, device=device)
        opt.zero_grad(set_to_none=True)
        loss = model(images[idx], codepoints[idx], font_meta[idx])
        loss.backward()
        opt.step()

        if step % args.report_every == 0 or step == 1:
            d_o, gts, recs = evaluate()
            l1 = F.l1_loss(recs, gts).item()
            recs_dev = recs.unsqueeze(1).to(device).clamp(0, 1)
            gts_dev = gts.unsqueeze(1).to(device).clamp(0, 1)
            lpips_val = lpips(recs_dev, gts_dev).mean().item()
            glyph = glyphloss_fn(recs_dev, gts_dev).item()
            save_montage(
                gts,
                recs,
                [f"ROND {r} (held-out)" for r in holdout_ronds],
                Path(args.montage_dir) / f"step_{step:06d}.png",
                f"held-out '{chr(holdout_cp)}' @ step {step}",
            )
            print(
                f"{step:>6} {loss.item():>8.4f} {l1:>8.4f} {lpips_val:>8.4f} "
                f"{glyph:>8.4f} {d_o:>8.3f}"
            )


if __name__ == "__main__":
    main()
