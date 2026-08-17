"""Curvature-weighted glyph reconstruction loss.

``glyphloss`` concentrates on the anti-aliased edge (grey pixels) and penalises
edge *sharpness* (gradient magnitude) and *orientation* (gradient direction), but
it is a *mean* over the image, so the few terminal/corner pixels contribute
negligible absolute loss and the optimizer ignores them. This module re-weights
the existing glyphloss terms by a curvature mask so those few pixels carry
commensurate weight.

The mask is level-set curvature of the *target*::

    κ = |∇·(∇I / |∇I|)|

i.e. the magnitude of the divergence of the unit normal field, which is the
(signed) curvature of the image's iso-contours — near-zero on straight edges,
large at corners/terminals. It is a function of the target only, so it is a
constant per-sample weight (no gradient flows through it).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from glyphloss import GlyphReconstructionLoss


def curvature_mask(target: torch.Tensor, mag_thresh: float = 0.05) -> torch.Tensor:
    """Return normalised level-set curvature κ ∈ [0, 1] for the target.

    Args:
        target: ``(B, C, H, W)`` glyph image in [0, 1] (C = 1 for greyscale).
        mag_thresh: gradient-magnitude floor for "on the contour" gating.

    Returns:
        ``(B, C, H, W)`` curvature in [0, 1], max-normalised.
    """
    # Central differences for the gradient, padded back to full resolution.
    gx = target[:, :, 2:, :] - target[:, :, :-2, :]  # (B,C,H-2,W)
    gy = target[:, :, :, 2:] - target[:, :, :, :-2]  # (B,C,H,W-2)
    gx = F.pad(gx, (0, 0, 1, 1))
    gy = F.pad(gy, (1, 1, 0, 0))
    mag = (gx ** 2 + gy ** 2).sqrt() + 1e-6
    nx, ny = gx / mag, gy / mag

    # Divergence of the unit normal field = curvature of the level sets.
    dnx = nx[:, :, 2:, :] - nx[:, :, :-2, :]
    dny = ny[:, :, :, 2:] - ny[:, :, :, :-2]
    dnx = F.pad(dnx, (0, 0, 1, 1))
    dny = F.pad(dny, (1, 1, 0, 0))
    kappa = (dnx + dny).abs()

    # Only meaningful on the contour; zero it out in flat black/white regions.
    kappa = kappa * (mag > mag_thresh).float()
    mx = kappa.max()
    if mx > 1e-6:
        kappa = kappa / mx
    return kappa


class CurvatureWeightedGlyphLoss(nn.Module):
    """``GlyphReconstructionLoss`` with grey weights amplified by curvature.

    The grey weights become ``w * (1 + k·κ)``, so corners/terminals (high κ) get
    up to ``(1 + k)``× the base weight. All term internals are inherited from
    ``GlyphReconstructionLoss``.
    """

    def __init__(self, k: float = 20.0, **kwargs):
        super().__init__()
        self.k = float(k)
        self.base = GlyphReconstructionLoss(**kwargs)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = self.base._grey_weights(target)
        w2 = w * (1.0 + self.k * curvature_mask(target))
        return (
            self.base.lambda_pixel * self.base._pixel_loss(pred, target, w2)
            + self.base._gradient_loss(pred, target, w2)
            + self.base.lambda_spectral * self.base._spectral_loss(pred, target, w2)
        )
