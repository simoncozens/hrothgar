#!/usr/bin/env python
"""Canary: find where fine-axis tracking breaks as we add degrees of freedom.

Two modes, in increasing difficulty:

* **Instanced** (default) — five static copies of YouTubeMarquee at ROND ∈
  {0, 25, 50, 75, 100}.  Every font is one "style source"; ``--glyphs`` targets
  are all trained (cycled), and we report whether the fixed ``--target-glyph``
  tracks ROND.

* **Variable** (``--variable-font``) — one variable font rendered at ``N`` random
  points in a chosen design-space (``--axes``).  Start with just ``ROND``, then
  add ``wdth,wght,XOPQ,YOPQ``.  This keeps the *same glyph skeleton* (same font)
  while entangling ROND with other axes.

Both modes share a training loop (L1/LPIPS/glyphloss/axis regression) and a
diag/off (``d/o``) tracking report on the fixed target glyph.  ``d/o < 1`` means
the decoded output visibly moves with ROND.
"""

from __future__ import annotations

import argparse
import glob
import random
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from hrothgar.glyphloss_curvature import CurvatureWeightedGlyphLoss
from hrothgar.googlefonts import StandaloneFont
from hrothgar.gtok.llamagen_lpips import LPIPS
from hrothgar.style_extraction.config import (
    StyleExtractionV4Config,
    StyleExtractionV5Config,
)
from hrothgar.style_extraction.model_v4 import StyleExtractionModelV4
from hrothgar.style_extraction.model_v5 import StyleExtractionModelV5
from hrothgar.style_extraction.render_utils import render_glyph


def parse_rond(name: str) -> int:
    """Extract the ROND value from a font/family name like ``YTM-ROND25``."""
    m = re.search(r"ROND(\d+)", name)
    if m is None:
        raise ValueError(f"cannot parse ROND value from {name!r}")
    return int(m.group(1))


def read_fvar(path: str | Path):
    """Return ``(tags, defaults, mins, maxs)`` in fvar order."""
    from fontTools.ttLib import TTFont

    axes = TTFont(path)["fvar"].axes
    tags = [a.axisTag for a in axes]
    defaults = [float(a.defaultValue) for a in axes]
    mins = [float(a.minValue) for a in axes]
    maxs = [float(a.maxValue) for a in axes]
    return tags, defaults, mins, maxs


class AxisHead(nn.Module):
    """Regress the ROND value (0..1) from a decoded glyph image."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (B,)


def diag_off(gts: torch.Tensor, recs: torch.Tensor) -> float:
    """diag/off L1 tracking metric; ``< 1`` means reconstructions track the axis."""
    n = gts.shape[0]
    d = (gts[:, None] - recs[None, :]).abs().mean(dim=(-1, -2))  # (N, N)
    diag = d.diagonal().mean()
    off = (d.sum() - d.diagonal().sum()) / (n * (n - 1))
    return float(diag / (off + 1e-12))


def save_montage(gts, recs, rond_values, path: Path, title: str) -> None:
    """Save a GT / recon / |GT-recon| montage across the ROND sweep."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = gts.shape[0]
    fig, axes = plt.subplots(3, n, figsize=(n * 1.3, 3.9))
    if n == 1:
        axes = axes[:, None]

    errs = [(gts[j] - recs[j]).abs() for j in range(n)]
    max_err = max(e.max().item() for e in errs) or 1.0
    for j in range(n):
        axes[0, j].imshow(gts[j].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, j].imshow(recs[j].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        axes[2, j].imshow(errs[j].numpy(), cmap="hot", vmin=0.0, vmax=max_err)
        axes[0, j].set_title(f"ROND {rond_values[j]}", fontsize=7)

    for row in range(3):
        for j in range(n):
            axes[row, j].set_xticks([])
            axes[row, j].set_yticks([])
    axes[0, 0].set_ylabel("GT", fontsize=8)
    axes[1, 0].set_ylabel("recon", fontsize=8)
    axes[2, 0].set_ylabel("|GT-rec|", fontsize=8)

    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  saved {path}")


def decode_target(model, target_idx, style):
    """Decode a target glyph from style, handling v4 (two-stage) vs v5 (single)."""
    if hasattr(model, "decode_coarse"):
        coarse = model.decode_coarse(target_idx, style)
        return model.decode_fine(target_idx, coarse, style)
    return model.decode(target_idx, style)


def render_batch(
    render_fns,
    rond_labels,
    target_cp: int,
    evidence_cps: list[int],
    cp_to_idx: dict[int, int],
    size: int,
    device: torch.device,
) -> dict:
    """Render one batch from a list of ``(cp, size) -> glyph`` renderers."""
    ev_idx = torch.tensor([cp_to_idx[cp] for cp in evidence_cps], device=device)
    tgt_idx = torch.tensor([cp_to_idx[target_cp]], device=device)

    style_images, style_cps, targets = [], [], []
    for fn in render_fns:
        ev = torch.stack([fn(cp, size) for cp in evidence_cps])  # (G, H, W)
        style_images.append(ev.unsqueeze(1))  # (G, 1, H, W)
        style_cps.append(ev_idx)
        targets.append(fn(target_cp, size))  # (H, W)

    return {
        "style_images": torch.stack(style_images).to(device),  # (B, G, 1, H, W)
        "style_codepoint_idx": torch.stack(style_cps).to(device),  # (B, G)
        "target_images": torch.stack(targets).unsqueeze(1).to(device),  # (B, 1, H, W)
        "target_codepoint_idx": tgt_idx.expand(len(render_fns))
        .contiguous()
        .to(device),  # (B,)
        "rond": torch.tensor(rond_labels, device=device, dtype=torch.float32),  # (B,)
    }


def evaluate_tracking(
    model,
    render_fns,
    target_cp: int,
    evidence_cps: list[int],
    cp_to_idx: dict[int, int],
    size: int,
    device: torch.device,
):
    """Decode the fixed target for each style source; return ``(d/o, gts, recs)``."""
    ev_idx = torch.tensor([cp_to_idx[cp] for cp in evidence_cps], device=device)
    tgt_idx = torch.tensor([cp_to_idx[target_cp]], device=device)
    gts, recs = [], []
    with torch.no_grad():
        for fn in render_fns:
            ev = torch.stack([fn(cp, size) for cp in evidence_cps])
            ev = ev.unsqueeze(1).unsqueeze(0).to(device)  # (1, G, 1, H, W)
            gt = fn(target_cp, size)  # (H, W)
            region = model.encode_style(ev, style_codepoint_idx=ev_idx.unsqueeze(0))
            rec = decode_target(model, tgt_idx, region)[0, 0].cpu()
            gts.append(gt)
            recs.append(rec)
    gts = torch.stack(gts)
    recs = torch.stack(recs)
    return diag_off(gts, recs), gts, recs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-version",
        default="v4",
        choices=["v4", "v5"],
        help="Which model to train (v4 = region grid, v5 = slot attention).",
    )
    p.add_argument(
        "--fonts",
        default="YTM-ROND*.ttf",
        help="Glob for instanced ROND fonts (instanced mode).",
    )
    p.add_argument(
        "--variable-font",
        default=None,
        help="Path to a variable font (variable mode; overrides --fonts).",
    )
    p.add_argument(
        "--axes",
        default="ROND",
        help="Comma-separated axes to randomise in variable mode.",
    )
    p.add_argument(
        "--n-axis-samples",
        type=int,
        default=5,
        help="Number of random axis positions in variable mode.",
    )
    p.add_argument(
        "--glyphs",
        default="anKMro",
        help="Codepoints to train on (as a string); all are cycled as targets.",
    )
    p.add_argument(
        "--target-glyph",
        default="r",
        help="Glyph decoded for the tracking report (default 'r').",
    )
    p.add_argument("--evidence", type=int, default=3, help="Number of evidence glyphs.")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--l1-weight", type=float, default=1.0)
    p.add_argument("--lpips-weight", type=float, default=0.1)
    p.add_argument(
        "--glyphloss-weight",
        type=float,
        default=0.0,
        help="Weight of glyphloss in the objective (0 = report-only metric).",
    )
    p.add_argument(
        "--axis-weight",
        type=float,
        default=0.0,
        help="Weight of the ROND regression loss (0 = off).",
    )
    p.add_argument("--report-every", type=int, default=200)
    p.add_argument("--montage-dir", default="outputs/canary")
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
    if args.evidence + 1 > len(glyphs):
        raise SystemExit(
            f"--evidence {args.evidence} too large for {len(glyphs)} glyphs"
        )

    # ---- Build style sources (render functions + ROND labels) ----
    if args.variable_font:
        font = StandaloneFont(args.variable_font)
        tags, defaults, mins, maxs = read_fvar(args.variable_font)
        axes = [a.strip() for a in args.axes.split(",") if a.strip()]
        if "ROND" not in tags:
            raise SystemExit(f"'ROND' not in fvar axes {tags}")

        rng = random.Random(args.seed)
        render_fns, rond_labels = [], []
        for _ in range(args.n_axis_samples):
            coords = list(defaults)
            for tag in axes:
                if tag in tags:
                    idx = tags.index(tag)
                    coords[idx] = rng.uniform(mins[idx], maxs[idx])
            render_fns.append(
                lambda cp, size, c=coords: render_glyph(font, cp, size, axis_position=c)
            )
            rond_labels.append(coords[tags.index("ROND")] / 100.0)

        # Evaluation sweep: ROND 0..100, all other axes at default.
        eval_render_fns, rond_values = [], []
        for r in (0, 25, 50, 75, 100):
            coords = list(defaults)
            coords[tags.index("ROND")] = float(r)
            eval_render_fns.append(
                lambda cp, size, c=coords: render_glyph(font, cp, size, axis_position=c)
            )
            rond_values.append(r)
        print(
            f"Variable mode: {args.variable_font}; axes {axes}; "
            f"{args.n_axis_samples} random samples"
        )
    else:
        font_paths = sorted(Path(x) for x in glob.glob(args.fonts))
        if not font_paths:
            raise SystemExit(f"no fonts matched {args.fonts!r}")
        fonts = [StandaloneFont(p) for p in font_paths]
        rond_values = sorted(parse_rond(p.stem) for p in font_paths)
        render_fns = [lambda cp, size, f=f: render_glyph(f, cp, size) for f in fonts]
        rond_labels = [r / 100.0 for r in rond_values]
        eval_render_fns = render_fns
        print(f"Instanced mode: {len(fonts)} fonts; ROND values {rond_values}")

    if args.model_version == "v5":
        config = StyleExtractionV5Config(
            image_size=args.image_size,
            character_set=glyphs,
            num_evidence_glyphs=args.evidence,
        )
        model = StyleExtractionModelV5(config).to(device)
    else:
        config = StyleExtractionV4Config(
            image_size=args.image_size,
            character_set=glyphs,
            num_evidence_glyphs=args.evidence,
        )
        model = StyleExtractionModelV4(config).to(device)
    axis_head = AxisHead().to(device)
    lpips = LPIPS().to(device)
    glyphloss_fn = CurvatureWeightedGlyphLoss(
        k=20.0, lambda_pixel=0.0, lambda_spectral=2.5
    ).to(device)

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(axis_head.parameters()), lr=args.lr
    )

    cp_to_idx = {cp: i for i, cp in enumerate(glyphs)}

    # One batch per target glyph; cycle through them so every glyph is trained.
    batches = []
    for g in glyphs:
        evidence_cps = [cp for cp in glyphs if cp != g][: args.evidence]
        batches.append(
            render_batch(
                render_fns,
                rond_labels,
                g,
                evidence_cps,
                cp_to_idx,
                args.image_size,
                device,
            )
        )

    target_evidence = [cp for cp in glyphs if cp != target_cp][: args.evidence]

    print(
        f"Training glyphs: {''.join(chr(c) for c in glyphs)}; "
        f"report glyph '{chr(target_cp)}'; evidence: {''.join(chr(c) for c in target_evidence)}"
    )

    print(f"{'step':>6} {'L1':>8} {'LPIPS':>8} {'glyph':>8} {'axis':>8} {'d/o':>8}")
    for step in range(1, args.steps + 1):
        batch = batches[(step - 1) % len(batches)]
        style_images = batch["style_images"]
        style_cp = batch["style_codepoint_idx"]
        target_images = batch["target_images"]
        target_idx = batch["target_codepoint_idx"]
        rond = batch["rond"]

        opt.zero_grad(set_to_none=True)
        region = model.encode_style(style_images, style_codepoint_idx=style_cp)
        recon = decode_target(model, target_idx, region)

        l1 = F.l1_loss(recon, target_images)
        lpips_loss = lpips(recon.clamp(0, 1), target_images.clamp(0, 1)).mean()
        glyph = glyphloss_fn(recon, target_images)
        axis_loss = F.mse_loss(axis_head(recon), rond)
        total = (
            args.l1_weight * l1
            + args.lpips_weight * lpips_loss
            + args.glyphloss_weight * glyph
            + args.axis_weight * axis_loss
        )
        total.backward()
        opt.step()

        if step % args.report_every == 0 or step == 1:
            d_o, gts, recs = evaluate_tracking(
                model,
                eval_render_fns,
                target_cp,
                target_evidence,
                cp_to_idx,
                args.image_size,
                device,
            )
            save_montage(
                gts,
                recs,
                rond_values,
                Path(args.montage_dir) / f"step_{step:06d}.png",
                f"target '{chr(target_cp)}' @ step {step}",
            )
            print(
                f"{step:>6} {l1.item():>8.4f} {lpips_loss.item():>8.4f} "
                f"{glyph.item():>8.4f} {axis_loss.item():>8.4f} {d_o:>8.3f}"
            )


if __name__ == "__main__":
    main()
