#!/usr/bin/env python
"""Diffusion canary: can a class-conditional diffusion model track fine axis detail?

This mirrors ``scripts/axis_canary.py`` but swaps the autoencoder for a
class-conditional diffusion model (``Lib/hrothgar/diffusion``, Phase 1).  The
question is the same one the v1–v5 autoencoders failed: given the same target
glyph rendered at several ROND values, does the sampled output visibly move with
the axis (``d/o < 1``)?

Unlike the autoencoder canary there is **no evidence glyph** — the model is
*told* the style as a class id (codepoint x ROND).  This is the cleanest possible
test of whether diffusion can reproduce fine terminal-roundness detail at all.
Phase 2 will make the style conditional on evidence glyphs instead.

The fixed-batch bug from the autoencoder canary is avoided by construction: every
``(glyph, ROND)`` pair is rendered into the training set, so the fixed target
glyph is always trained.
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from hrothgar.diffusion.config import DiffusionConfig
from hrothgar.diffusion.dataset import build_rond_dataset, materialize
from hrothgar.diffusion.losses import diag_off, save_montage
from hrothgar.diffusion.model import build_diffusion_model
from hrothgar.diffusion.train import DiffusionTrainer
from hrothgar.glyphloss_curvature import CurvatureWeightedGlyphLoss
from hrothgar.googlefonts import StandaloneFont
from hrothgar.gtok.llamagen_lpips import LPIPS
from hrothgar.style_extraction.render_utils import render_glyph


def parse_rond(name: str) -> int:
    """Extract the ROND value from a font/family name like ``YTM-ROND25``."""
    m = re.search(r"ROND(\d+)", name)
    if m is None:
        raise ValueError(f"cannot parse ROND value from {name!r}")
    return int(m.group(1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fonts", default="YTM-ROND*.ttf", help="Glob for instanced ROND fonts."
    )
    p.add_argument(
        "--glyphs", default="anKMro", help="Codepoints to train on (as a string)."
    )
    p.add_argument(
        "--target-glyph",
        default="r",
        help="Glyph decoded for the tracking report (default 'r').",
    )
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--timesteps", type=int, default=250)
    p.add_argument("--sampling-timesteps", type=int, default=50)
    p.add_argument("--cond-drop-prob", type=float, default=0.1)
    p.add_argument("--cond-scale", type=float, default=3.0)
    p.add_argument(
        "--glyphloss-weight",
        type=float,
        default=0.0,
        help="Weight of the glyph reconstruction loss on a "
        "differentiably-sampled glyph (0 = pure diffusion).",
    )
    p.add_argument("--report-every", type=int, default=200)
    p.add_argument("--montage-dir", default="outputs/canary_diffusion")
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
    target_cp = ord(args.target_glyph)
    if target_cp not in glyphs:
        raise SystemExit(
            f"--target-glyph {args.target_glyph!r} not in --glyphs {args.glyphs!r}"
        )

    # ---- Style sources: one renderer per ROND value (sorted, aligned). ----
    pairs = sorted((parse_rond(Path(x).stem), x) for x in glob.glob(args.fonts))
    if not pairs:
        raise SystemExit(f"no fonts matched {args.fonts!r}")
    rond_values = [r for r, _ in pairs]
    fonts = [StandaloneFont(path) for _, path in pairs]
    render_fns = [(lambda cp, size, f=f: render_glyph(f, cp, size)) for f in fonts]
    print(f"Instanced mode: {len(pairs)} fonts; ROND values {rond_values}")

    # ---- Dataset: every (glyph, ROND) pair -> (image, class id). ----
    dataset, vocab = build_rond_dataset(
        render_fns, rond_values, glyphs, args.image_size
    )
    images, class_ids = materialize(dataset)
    images = images.to(device)
    class_ids = class_ids.to(device)
    print(f"Training samples: {len(dataset)} (vocab {vocab.num_classes} classes)")

    # ---- Model. ----
    glyphloss_fn = CurvatureWeightedGlyphLoss(
        k=20.0, lambda_pixel=0.0, lambda_spectral=2.5
    ).to(device)
    config = DiffusionConfig(
        image_size=args.image_size,
        num_classes=vocab.num_classes,
        dim=args.dim,
        timesteps=args.timesteps,
        sampling_timesteps=args.sampling_timesteps,
        cond_drop_prob=args.cond_drop_prob,
        cond_scale=args.cond_scale,
        learning_rate=args.lr,
        glyphloss_weight=args.glyphloss_weight,
    )
    model = build_diffusion_model(config, glyphloss_fn=glyphloss_fn)
    trainer = DiffusionTrainer(model, config, device)

    # ---- Metrics. ----
    lpips = LPIPS().to(device)

    # Fixed RNG for batch sampling (deterministic across runs).
    gen = torch.Generator(device=device).manual_seed(args.seed)
    n_samples = len(dataset)
    eval_seed = args.seed + 9999

    def render_gt(r: int) -> torch.Tensor:
        """Ground-truth target glyph at a given ROND value."""
        return render_glyph(fonts[rond_values.index(r)], target_cp, args.image_size)

    def evaluate() -> tuple[float, torch.Tensor, torch.Tensor]:
        gts, recs = [], []
        for r in rond_values:
            gt = render_gt(r)
            class_id = torch.tensor([vocab.encode(target_cp, r)], device=device)
            torch.manual_seed(eval_seed)
            rec = trainer.sample(class_id)[0, 0].cpu()
            gts.append(gt)
            recs.append(rec)
        gts_t = torch.stack(gts)
        recs_t = torch.stack(recs)
        return diag_off(gts_t, recs_t), gts_t, recs_t

    print(
        f"{'step':>6} {'loss':>8} {'diff':>8} {'aux':>8} {'L1':>8} {'LPIPS':>8} {'glyph':>8} {'d/o':>8}"
    )
    for step in range(1, args.steps + 1):
        idx = torch.randint(
            0, n_samples, (args.batch_size,), generator=gen, device=device
        )
        total, diff, aux = trainer.train_step(images[idx], class_ids[idx])

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
                rond_values,
                Path(args.montage_dir) / f"step_{step:06d}.png",
                f"target '{chr(target_cp)}' @ step {step}",
            )
            print(
                f"{step:>6} {total.item():>8.4f} {diff.item():>8.4f} {aux.item():>8.4f} "
                f"{l1:>8.4f} {lpips_val:>8.4f} {glyph:>8.4f} {d_o:>8.3f}"
            )


if __name__ == "__main__":
    main()
