#!/usr/bin/env python
"""Verify that ``glyphloss`` actually concentrates on the details we care about.

The parametric-font probe showed the shape head reproduces ink-mass changes
(YOPQ/XOPQ) but *not* curvature changes (STLI/STUO/ROND). The hypothesis is that
glyphloss *sees* curvature (its gradient-direction + spectral terms fire on
corners) but the *absolute* loss is negligible because terminals are a tiny
fraction of the image and every term is a mean.

This script checks that directly, offline (no training, no style model):

* **Loss magnitude + per-term breakdown** for a single axis sweep pair
  (e.g. ROND 0 vs 100), so we can see how much of the loss is pixel / gradient /
  spectral, and how the total scales with L1.
* **Loss-mass localisation** — a per-pixel map of where the grey-weighted pixel
  error and the gradient-direction error actually land, to confirm they
  concentrate on the terminal/corner (a tiny region) rather than spreading.
* **Gradient check** — backprop the loss through ``pred`` and report the gradient
  norm, the directional derivative toward the target, and the fraction of
  gradient mass that sits in the changed (terminal) region.

Run one pair at a time; use YOPQ (ink) as the positive control and ROND/STLI
(curvature) as the case of interest::

    PYTHONPATH=Lib python scripts/glyphloss_verify.py \
        --font "YouTubeMarquee[BASE,GRAD,ROND,SCAL,TANG,TRML,WDSP,XINK,XOFI,XOJN,XOLC,XOPQ,XORN,XOUC,XTFI,XTLC,XTLR,XTRA,XTSP,XTUC,YOFI,YOLC,YOPQ,YOUC,YTDE,YTLC,opsz,wdth,wght].ttf" \
        --axis ROND --value-a 0 --value-b 100 --glyph a

    PYTHONPATH=Lib python scripts/glyphloss_verify.py \
        --font RobotoDelta-Roman-VF.ttf --axis STLI --value-a 2 --value-b 412
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
    """Render ``codepoints`` at ``overrides`` → ``(G, 1, H, W)`` float [0,1]."""
    coords = list(defaults)
    for tag, val in overrides.items():
        coords[tags.index(tag)] = float(val)
    glyphs = []
    for cp in codepoints:
        arr = font.render(cp, size=size, axis_position=coords)
        gray = arr[0].copy() if arr.ndim == 3 else np.asarray(arr, dtype=np.float32)
        glyphs.append(torch.from_numpy(gray))
    return torch.stack(glyphs).unsqueeze(1)


def grey_weights(target: torch.Tensor, eps: float = 0.2) -> torch.Tensor:
    """Mirrors ``GlyphReconstructionLoss._grey_weights``."""
    return 2.0 * target * (1.0 - target) + eps


def gradient_direction_map(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 0.2
) -> torch.Tensor:
    """Per-pixel gradient-direction loss ``w * target_mag * (1 - cosθ)``.

    Mirrors ``GlyphReconstructionLoss._gradient_loss`` but returns the un-reduced
    ``(1, 1, H-1, W-1)`` map instead of the mean.
    """
    weights = grey_weights(target, eps)
    pd_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pd_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    td_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    td_y = target[:, :, 1:, :] - target[:, :, :-1, :]

    pd_x = pd_x[:, :, :-1, :]
    pd_y = pd_y[:, :, :, :-1]
    td_x = td_x[:, :, :-1, :]
    td_y = td_y[:, :, :, :-1]

    w_dx = (weights[:, :, :-1, :-1] + weights[:, :, :-1, 1:]) / 2.0
    w_dy = (weights[:, :, :-1, :-1] + weights[:, :, 1:, :-1]) / 2.0
    w = (w_dx + w_dy) / 2.0

    pred_mag = (pd_x**2 + pd_y**2 + 1e-8).sqrt()
    target_mag = (td_x**2 + td_y**2 + 1e-8).sqrt()
    dot = pd_x * td_x + pd_y * td_y
    cos = (dot / (pred_mag * target_mag + 1e-6)).clamp(-1.0, 1.0)
    return w * target_mag * (1.0 - cos)  # (1,1,H-1,W-1)


def _save_figure(
    target: torch.Tensor,
    pred: torch.Tensor,
    diff_mask: torch.Tensor,
    dir_map: torch.Tensor,
    out_path: str,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def img(t):
        return t.detach().squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    axes[0].imshow(img(target), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("target")
    axes[1].imshow(img(grey_weights(target)), cmap="magma")
    axes[1].set_title("grey weight 2p(1-p)+ε")
    axes[2].imshow(img(diff_mask.float()), cmap="Reds", vmin=0, vmax=1)
    axes[2].set_title("changed pixels")
    axes[3].imshow(
        img(F.interpolate(dir_map, size=target.shape[-2:], mode="bilinear")),
        cmap="viridis",
    )
    axes[3].set_title("grad-direction loss")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  saved localisation figure to {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--font", required=True, help="Path to a parametric font.")
    p.add_argument("--axis", required=True, help="Axis tag to vary.")
    p.add_argument("--value-a", type=float, required=True)
    p.add_argument("--value-b", type=float, required=True)
    p.add_argument(
        "--glyph", default="a", help="Glyph for localisation/gradient (default 'a')."
    )
    p.add_argument("--size", type=int, default=64, help="Render size.")
    p.add_argument("--out-dir", default="outputs", help="Directory for figures.")
    args = p.parse_args()

    codepoints = list(DEFAULT_INPUT_CODEPOINTS)
    if ord(args.glyph) not in codepoints:
        raise SystemExit(f"--glyph {args.glyph!r} not in DEFAULT_INPUT_CODEPOINTS")
    gi = codepoints.index(ord(args.glyph))

    tags, defaults, mins, maxs = read_fvar(args.font)
    if args.axis not in tags:
        raise SystemExit(f"axis {args.axis!r} not in {Path(args.font).name}")
    font = StandaloneFont(args.font)

    A = render_instance(
        font, args.size, codepoints, tags, defaults, {args.axis: args.value_a}
    )
    B = render_instance(
        font, args.size, codepoints, tags, defaults, {args.axis: args.value_b}
    )

    loss_fn = GlyphReconstructionLoss()
    w = loss_fn._grey_weights(B)
    pixel = loss_fn._pixel_loss(A, B, w).item()
    grad = loss_fn._gradient_loss(A, B, w).item()
    spec = loss_fn._spectral_loss(A, B, w).item()
    total = loss_fn(A, B).item()
    l1 = float((A - B).abs().mean().item())
    changed = float(((A - B).abs() > 0.01).float().mean().item())

    print(f"\n{Path(args.font).name}  {args.axis}: {args.value_a} → {args.value_b}")
    print(f"  L1(pred,target)          = {l1:.6f}")
    print(f"  changed-pixel fraction   = {changed:.4f}  ({changed*100:.2f}%)")
    print(f"  glyphloss terms (installed defaults):")
    print(f"    pixel    = {pixel:.6f}   (λ={loss_fn.lambda_pixel})")
    print(
        f"    gradient = {grad:.6f}   (mag λ={loss_fn.lambda_mag} + dir λ={loss_fn.lambda_dir})"
    )
    print(f"    spectral = {spec:.6f}   (λ={loss_fn.lambda_spectral})")
    print(f"    TOTAL    = {total:.6f}")
    print(f"  glyphloss / L1           = {total / (l1 + 1e-12):.3f}")

    # ── Gradient check on a single glyph ───────────────────────────────
    a = A[gi : gi + 1].clone().detach().requires_grad_(True)
    t = B[gi : gi + 1].clone().detach()
    loss_fn(a, t).backward()
    g = a.grad
    g_norm = float(g.norm().item())
    dd = float((g * (t - a)).sum().item())
    diff = (a.detach() - t).abs() > 0.01
    g_frac = float(
        (g.abs() * diff.float()).sum().item() / (g.abs().sum().item() + 1e-12)
    )

    print(f"\n  gradient check (glyph {args.glyph!r}, pred=A target=B):")
    print(f"    ||grad||                = {g_norm:.6f}")
    print(f"    ||grad|| / L1           = {g_norm / (l1 + 1e-12):.3f}")
    print(
        f"    directional deriv       = {dd:.6f}  (negative ⇒ gradient points at target)"
    )
    print(f"    grad mass in changed px = {g_frac:.3f}")

    _save_figure(
        B[gi : gi + 1],
        A[gi : gi + 1],
        (A[gi : gi + 1] - B[gi : gi + 1]).abs() > 0.01,
        gradient_direction_map(A[gi : gi + 1], B[gi : gi + 1]),
        str(Path(args.out_dir) / f"glyphloss_{Path(args.font).stem}_{args.axis}.png"),
        title=f"{Path(args.font).stem}  {args.axis} {args.value_a}→{args.value_b}  glyph {args.glyph!r}",
    )


if __name__ == "__main__":
    main()
