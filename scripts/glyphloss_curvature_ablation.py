#!/usr/bin/env python
"""Curvature-weighted glyphloss ablation (offline, no training).

The verification script showed glyphloss *does* localise to terminals/corners,
but its absolute gradient there is ~10–16× smaller than for an ink-mass change,
because terminals are a tiny fraction of the image and every glyphloss term is a
mean. The proposed fix is a curvature/corner weighting mask that amplifies those
few pixels.

This script tests that offline: it computes a curvature mask from the *target*
(level-set curvature, i.e. |∇·(∇I/|∇I|)|), re-weights the existing glyphloss
terms by ``(1 + k·curvature)``, and sweeps ``k``. For each ``k`` it reports the
gradient norm for a curvature case (ROND/STLI) vs an ink control (YOPQ), plus the
directional derivative (must stay negative ⇒ gradient still points at the
target).

Acceptance: find the ``k`` where ``‖grad‖_curv`` becomes comparable to
``‖grad‖_ink`` (ratio → ~0.3–1.0) without the gradient flipping sign.

Example::

    PYTHONPATH=Lib python scripts/glyphloss_curvature_ablation.py \
        --curv-font "YouTubeMarquee[BASE,GRAD,ROND,SCAL,TANG,TRML,WDSP,XINK,XOFI,XOJN,XOLC,XOPQ,XORN,XOUC,XTFI,XTLC,XTLR,XTRA,XTSP,XTUC,YOFI,YOLC,YOPQ,YOUC,YTDE,YTLC,opsz,wdth,wght].ttf" \
        --curv-axis ROND --curv-a 0 --curv-b 100 \
        --ink-font RobotoDelta-Roman-VF.ttf --ink-axis YOPQ --ink-a 2 --ink-b 280
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from glyphloss import GlyphReconstructionLoss
from hrothgar.googlefonts import StandaloneFont
from hrothgar.style_embedding.config import DEFAULT_INPUT_CODEPOINTS


def read_fvar(path: str) -> tuple[list[str], list[float], list[float], list[float]]:
    from fontTools.ttLib import TTFont

    axes = TTFont(path)["fvar"].axes
    tags = [a.axisTag for a in axes]
    defaults = [float(a.defaultValue) for a in axes]
    mins = [float(a.minValue) for a in axes]
    maxs = [float(a.maxValue) for a in axes]
    return tags, defaults, mins, maxs


def render_instance(
    font: StandaloneFont,
    size: int,
    codepoints: list[int],
    tags: list[str],
    defaults: list[float],
    overrides: dict[str, float],
) -> torch.Tensor:
    coords = list(defaults)
    for tag, val in overrides.items():
        coords[tags.index(tag)] = float(val)
    glyphs = []
    for cp in codepoints:
        arr = font.render(cp, size=size, axis_position=coords)
        gray = arr[0].copy() if arr.ndim == 3 else np.asarray(arr, dtype=np.float32)
        glyphs.append(torch.from_numpy(gray))
    return torch.stack(glyphs).unsqueeze(1)  # (G,1,H,W)


def curvature_mask(target: torch.Tensor, mag_thresh: float = 0.05) -> torch.Tensor:
    """Return normalised level-set curvature κ ∈ [0,1] for the target.

    κ = |∇·(∇I/|∇I|)| — the divergence of the unit normal field — is the
    (signed) curvature of the image's iso-contours: near-zero on straight edges,
    large at corners/terminals. Gated to the contour and max-normalised.
    """
    gx = target[:, :, 2:, :] - target[:, :, :-2, :]  # (B,1,H-2,W)
    gy = target[:, :, :, 2:] - target[:, :, :, :-2]  # (B,1,H,W-2)
    gx = F.pad(gx, (0, 0, 1, 1))
    gy = F.pad(gy, (1, 1, 0, 0))
    mag = (gx**2 + gy**2).sqrt() + 1e-6
    nx, ny = gx / mag, gy / mag

    dnx = nx[:, :, 2:, :] - nx[:, :, :-2, :]
    dny = ny[:, :, :, 2:] - ny[:, :, :, :-2]
    dnx = F.pad(dnx, (0, 0, 1, 1))
    dny = F.pad(dny, (1, 1, 0, 0))

    kappa = (dnx + dny).abs()
    kappa = kappa * (mag > mag_thresh).float()
    mx = kappa.max()
    if mx > 1e-6:
        kappa = kappa / mx
    return kappa  # (B,1,H,W) in [0,1]


def weighted_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    k: float,
    loss_fn: GlyphReconstructionLoss,
) -> torch.Tensor:
    """glyphloss with the grey weights multiplied by ``(1 + k·curvature)``."""
    w = loss_fn._grey_weights(target)
    w2 = w * (1.0 + k * curvature_mask(target))
    return (
        loss_fn.lambda_pixel * loss_fn._pixel_loss(pred, target, w2)
        + loss_fn._gradient_loss(pred, target, w2)
        + loss_fn.lambda_spectral * loss_fn._spectral_loss(pred, target, w2)
    )


def grad_stats(
    pred: torch.Tensor, target: torch.Tensor, k: float, loss_fn
) -> tuple[float, float]:
    """Return (‖grad‖, directional derivative) for the curvature-weighted loss."""
    p = pred.clone().detach().requires_grad_(True)
    weighted_loss(p, target, k, loss_fn).backward()
    g = p.grad
    g_norm = float(g.norm().item())
    dd = float((g * (target - p)).sum().item())
    return g_norm, dd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--curv-font", required=True)
    p.add_argument("--curv-axis", required=True)
    p.add_argument("--curv-a", type=float, required=True)
    p.add_argument("--curv-b", type=float, required=True)
    p.add_argument("--ink-font", required=True)
    p.add_argument("--ink-axis", required=True)
    p.add_argument("--ink-a", type=float, required=True)
    p.add_argument("--ink-b", type=float, required=True)
    p.add_argument(
        "--glyph", default="a", help="Glyph for the gradient check (default 'a')."
    )
    p.add_argument("--size", type=int, default=64)
    p.add_argument(
        "--k-values", default="0,1,5,10,20,50,100", help="Comma-separated k sweep."
    )
    p.add_argument(
        "--plot-mask", action="store_true", help="Save a curvature-mask figure."
    )
    p.add_argument("--out-dir", default="outputs")
    args = p.parse_args()

    codepoints = list(DEFAULT_INPUT_CODEPOINTS)
    if ord(args.glyph) not in codepoints:
        raise SystemExit(f"--glyph {args.glyph!r} not in DEFAULT_INPUT_CODEPOINTS")
    gi = codepoints.index(ord(args.glyph))
    k_values = [float(x) for x in args.k_values.split(",") if x.strip()]

    def render_case(font_path, axis, va, vb):
        tags, defaults, _, _ = read_fvar(font_path)
        if axis not in tags:
            raise SystemExit(f"axis {axis!r} not in {Path(font_path).name}")
        font = StandaloneFont(font_path)
        A = render_instance(font, args.size, codepoints, tags, defaults, {axis: va})
        B = render_instance(font, args.size, codepoints, tags, defaults, {axis: vb})
        return A[gi : gi + 1], B[gi : gi + 1]  # single glyph

    curv_pred, curv_tgt = render_case(
        args.curv_font, args.curv_axis, args.curv_a, args.curv_b
    )
    ink_pred, ink_tgt = render_case(
        args.ink_font, args.ink_axis, args.ink_a, args.ink_b
    )

    loss_fn = GlyphReconstructionLoss()
    print(
        f"\ncurvature case: {Path(args.curv_font).name} {args.curv_axis} {args.curv_a}→{args.curv_b}"
    )
    print(
        f"ink control:    {Path(args.ink_font).name} {args.ink_axis} {args.ink_a}→{args.ink_b}"
    )
    print(
        f"\n{'k':>6} {'‖g‖_curv':>10} {'‖g‖_ink':>10} {'ratio':>8} {'dd_curv':>10} {'dd_ink':>10}"
    )
    for k in k_values:
        gc, ddc = grad_stats(curv_pred, curv_tgt, k, loss_fn)
        gi_n, ddi = grad_stats(ink_pred, ink_tgt, k, loss_fn)
        ratio = gc / (gi_n + 1e-12)
        print(
            f"{k:>6.1f} {gc:>10.4f} {gi_n:>10.4f} {ratio:>8.3f} {ddc:>10.4f} {ddi:>10.4f}"
        )

    if args.plot_mask:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm = curvature_mask(curv_tgt)
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
        axes[0].imshow(curv_tgt.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("target")
        axes[1].imshow(cm.squeeze().numpy(), cmap="magma", vmin=0, vmax=1)
        axes[1].set_title("curvature mask")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        out = (
            Path(args.out_dir)
            / f"curvature_mask_{Path(args.curv_font).stem}_{args.curv_axis}.png"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"\nsaved mask figure to {out}")


if __name__ == "__main__":
    main()
