"""Loss utilities for DiT-based glyph generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from hrothgar.ar.model import GlyphGenOutput


@dataclass(frozen=True)
class GlyphGenLossWeights:
    """Weights for the DiT training objectives."""

    noise_mse: float = 1.0
    pixel_l1: float = 1.0
    perceptual_lpips: float = 2.0


def compute_glyph_gen_loss(
    model_output: GlyphGenOutput,
    target_images: torch.Tensor,
    *,
    weights: GlyphGenLossWeights = GlyphGenLossWeights(),
    lpips_metric: Optional[object] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute DiT training loss and loggable terms.

    Primary objective: MSE between predicted and true noise.
    Auxiliary objectives: pixel L1 and LPIPS perceptual loss between
    the decoded x̂₀ prediction and the ground-truth image.

    Args:
        model_output: Output from ``GlyphGenerator.forward``.
        target_images: Ground-truth glyph images, ``(B, 3, H, W)``.
        weights: Loss weight configuration.
        lpips_metric: LPIPS module instance (required if
            ``perceptual_lpips > 0``).

    Returns:
        ``(total_loss, terms)`` where terms is suitable for TensorBoard
        logging.
    """
    device = target_images.device

    # Primary: noise prediction MSE.
    noise_mse = F.mse_loss(model_output.noise_pred, model_output.noise_target)

    # Auxiliary: pixel L1 between decoded x̂₀ and ground truth.
    pixel_l1 = F.l1_loss(model_output.reconstructed_images, target_images)

    # Auxiliary: LPIPS perceptual loss on Gumbel-softmax decoded image.
    perceptual_lpips = torch.tensor(0.0, device=device)
    if weights.perceptual_lpips > 0 and model_output.perceptual_recon is not None:
        if lpips_metric is None:
            raise ValueError(
                "lpips_metric is required when perceptual_lpips > 0 "
                "and perceptual_recon is provided"
            )
        perceptual_recon_clamped = torch.clamp(model_output.perceptual_recon, 0.0, 1.0)
        target_clamped = torch.clamp(target_images, 0.0, 1.0)
        perceptual_lpips = lpips_metric(perceptual_recon_clamped, target_clamped).mean()

    weighted_noise_mse = weights.noise_mse * noise_mse
    weighted_pixel_l1 = weights.pixel_l1 * pixel_l1
    weighted_perceptual_lpips = weights.perceptual_lpips * perceptual_lpips

    total_loss = weighted_noise_mse + weighted_pixel_l1 + weighted_perceptual_lpips

    terms: Dict[str, torch.Tensor] = {
        "total": total_loss.detach(),
        "noise_mse": noise_mse.detach(),
        "pixel_l1": pixel_l1.detach(),
        "perceptual_lpips": perceptual_lpips.detach(),
        "weighted_noise_mse": weighted_noise_mse.detach(),
        "weighted_pixel_l1": weighted_pixel_l1.detach(),
        "weighted_perceptual_lpips": weighted_perceptual_lpips.detach(),
    }

    return total_loss, terms
