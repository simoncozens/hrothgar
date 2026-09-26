#!/usr/bin/env python
"""Phase 2 exemplar-conditional canary.

The same ROND tracking question as ``diffusion_axis_canary.py``, but now the
model is *not* told the style as a class label.  It is given a handful of
**evidence glyphs** (other codepoints rendered in the target style) plus the
**target codepoint**, and must infer the style from the evidence.  This is the
many-shot setup that Phase 1 could not express.

The metric is still ``d/o``: for a fixed target glyph swept across ROND values,
does the sampled output move with the axis?
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from hrothgar.diffusion.config import ExemplarDiffusionConfig
from hrothgar.diffusion.dataset import build_exemplar_rond_data
from hrothgar.diffusion.exemplar import build_exemplar_model
from hrothgar.diffusion.losses import attention_health, diag_off, save_montage
from hrothgar.glyphloss_curvature import CurvatureWeightedGlyphLoss
from hrothgar.googlefonts import StandaloneFont
from hrothgar.gtok.llamagen_lpips import LPIPS
from hrothgar.style_extraction.render_utils import render_glyph


def parse_rond(name: str) -> int:
    m = re.search(r"ROND(\d+)", name)
    if m is None:
        raise ValueError(f"cannot parse ROND value from {name!r}")
    return int(m.group(1))


def report_attention(model) -> None:
    """Print per-layer cross-attention health from the last forward/sample."""
    ws = model.attention_weights()
    if not ws:
        print("  (no cross-attention weights recorded)")
        return
    print(f"  cross-attn ({len(ws)} layers):")
    print(f"    {'layer':>5}  {'entropy':>7}  {'eff_tok':>7}  {'q_var':>8}")
    for i, w in enumerate(ws):
        q_var, ent, eff = attention_health(w)
        print(
            f"    {i:>5}  {ent.item():>7.3f}  {eff.item():>7.1f}  {q_var.item():>8.4f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fonts", default="YTM-ROND*.ttf")
    p.add_argument(
        "--glyphs", default="anKMro", help="Codepoints available (as a string)."
    )
    p.add_argument(
        "--target-glyph",
        default="r",
        help="Glyph decoded for the tracking report (default 'r').",
    )
    p.add_argument(
        "--num-evidence",
        type=int,
        default=3,
        help="Number of evidence glyphs per sample.",
    )
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--timesteps", type=int, default=250)
    p.add_argument("--sampling-timesteps", type=int, default=50)
    p.add_argument("--report-every", type=int, default=200)
    p.add_argument("--montage-dir", default="outputs/canary_exemplar")
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
    if args.num_evidence >= len(glyphs):
        raise SystemExit(
            f"--num-evidence {args.num_evidence} too large for {len(glyphs)} glyphs"
        )

    pairs = sorted((parse_rond(Path(x).stem), x) for x in glob.glob(args.fonts))
    if not pairs:
        raise SystemExit(f"no fonts matched {args.fonts!r}")
    rond_values = [r for r, _ in pairs]
    fonts = [StandaloneFont(path) for _, path in pairs]
    render_fns = [(lambda cp, size, f=f: render_glyph(f, cp, size)) for f in fonts]
    print(f"Instanced mode: {len(pairs)} fonts; ROND values {rond_values}")

    evidence, codepoints, targets, cp_to_idx = build_exemplar_rond_data(
        render_fns, rond_values, glyphs, args.num_evidence, args.image_size
    )
    evidence = evidence.to(device)
    codepoints = codepoints.to(device)
    targets = targets.to(device)
    n = len(evidence)
    print(
        f"Training samples: {n} (codepoints {len(glyphs)}, evidence {args.num_evidence})"
    )

    config = ExemplarDiffusionConfig(
        image_size=args.image_size,
        num_codepoints=len(glyphs),
        dim=args.dim,
        timesteps=args.timesteps,
        sampling_timesteps=args.sampling_timesteps,
        num_evidence_glyphs=args.num_evidence,
        learning_rate=args.lr,
    )
    model = build_exemplar_model(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    lpips = LPIPS().to(device)
    glyphloss_fn = CurvatureWeightedGlyphLoss(
        k=20.0, lambda_pixel=0.0, lambda_spectral=2.5
    ).to(device)

    gen = torch.Generator(device=device).manual_seed(args.seed)
    eval_seed = args.seed + 9999
    tgt_idx = cp_to_idx[target_cp]
    target_evidence_cps = [g for g in glyphs if g != target_cp][: args.num_evidence]

    def evaluate() -> tuple[float, torch.Tensor, torch.Tensor]:
        gts, recs = [], []
        for r in rond_values:
            fn = render_fns[rond_values.index(r)]
            ev = torch.stack([fn(cp, args.image_size) for cp in target_evidence_cps])
            ev = ev.unsqueeze(1).unsqueeze(0).to(device)  # (1, N, 1, H, W)
            gt = fn(target_cp, args.image_size)
            cp_t = torch.tensor([tgt_idx], device=device)
            torch.manual_seed(eval_seed)
            rec = model.sample(ev, cp_t)[0, 0].cpu()
            gts.append(gt)
            recs.append(rec)
        gts_t = torch.stack(gts)
        recs_t = torch.stack(recs)
        return diag_off(gts_t, recs_t), gts_t, recs_t

    print(f"{'step':>6} {'loss':>8} {'L1':>8} {'LPIPS':>8} {'glyph':>8} {'d/o':>8}")
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n, (args.batch_size,), generator=gen, device=device)
        opt.zero_grad(set_to_none=True)
        loss = model(targets[idx], evidence[idx], codepoints[idx])
        loss.backward()
        opt.step()

        if step % args.report_every == 0 or step == 1:
            d_o, gts, recs = evaluate()
            report_attention(model)
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
                f"{step:>6} {loss.item():>8.4f} {l1:>8.4f} {lpips_val:>8.4f} "
                f"{glyph:>8.4f} {d_o:>8.3f}"
            )


if __name__ == "__main__":
    main()
