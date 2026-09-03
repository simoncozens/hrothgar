"""Evaluation helpers for the diffusion glyph generator.

The diffusion *training* loss is the noise-prediction MSE inside
``GaussianDiffusion`` (returned by ``DiffusionGlyphModel.forward``), so there is
nothing to implement here on the training side.  What we *do* need is a set of
**metrics** to answer the canary question — "does the sampled glyph track the
style axis?" — and to diagnose the failure modes we already know to look out for:

* ``diag_off`` (``d/o``) — the ratio of mean diagonal L1 to mean off-diagonal L1
  across an axis sweep.  ``< 1`` means the reconstruction moves with the axis;
  ``== 1.000`` means the reconstructions carry *no axis-aligned information* —
  either they are all identical, or they are noise uncorrelated with the sweep.
  This is the same metric the autoencoder canaries used, so results are directly
  comparable.
* ``AxisHead`` — a small CNN that regresses the axis value from a sampled glyph,
  giving a scalar "is the axis recoverable from the output" signal.
* ``mean_abs_diff`` — the mean ``|own - swapped|`` L1 used by the style-swap
  probe.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def diag_off(gts: torch.Tensor, recs: torch.Tensor) -> float:
    """diag/off L1 tracking metric; ``< 1`` means reconstructions track the axis.

    Args:
        gts: ``(N, H, W)`` ground-truth glyphs across an axis sweep.
        recs: ``(N, H, W)`` reconstructions for the same sweep.

    Returns:
        Mean diagonal L1 / mean off-diagonal L1.  ``~1`` means the
        reconstructions carry no axis-aligned information (all identical, or
        noise uncorrelated with the sweep).
    """
    n = gts.shape[0]
    d = (gts[:, None] - recs[None, :]).abs().mean(dim=(-1, -2))  # (N, N)
    diag = d.diagonal().mean()
    off = (d.sum() - d.diagonal().sum()) / (n * (n - 1))
    return float(diag / (off + 1e-12))


def mean_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean ``|a - b|`` L1 over all pixels (style-swap metric)."""
    return float((a - b).abs().mean().item())


def attention_health(attn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(q_var, entropy, effective_tokens)`` for cross-attention weights.

    ``attn`` is ``(B, heads, nq, K)`` post-softmax.  ``q_var`` is the variance
    of attention across query positions (~0 means every query attends the same
    way — the global-collapse failure mode); ``entropy`` is normalised to
    [0, 1] (1 = uniform, 0 = one-hot); ``effective_tokens`` is the
    participation ratio (K = uniform, 1 = one-hot).
    """
    import math

    k = attn.shape[-1]
    ent = -(attn * (attn + 1e-12).log()).sum(dim=-1) / math.log(k)
    eff = 1.0 / (attn ** 2).sum(dim=-1)
    q_var = attn.var(dim=2).mean()
    return q_var, ent.mean(), eff.mean()


class AxisHead(nn.Module):
    """Regress a style-axis value (0..1) from a grayscale glyph image."""

    def __init__(self, channels: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 16, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (B,)


def save_montage(
    gts: torch.Tensor,
    recs: torch.Tensor,
    labels,
    path: Path,
    title: str,
) -> None:
    """Save a GT / recon / |GT-recon| montage across an axis sweep."""
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
        axes[0, j].set_title(str(labels[j]), fontsize=7)

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
