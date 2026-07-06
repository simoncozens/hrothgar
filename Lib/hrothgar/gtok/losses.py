"""Loss utilities for G-Tok training.

This module combines reconstruction and quantization losses used to train the
G-Tok tokenizer:

1. L1 reconstruction loss between reconstructed and target glyph images
2. Optional perceptual loss (e.g. VGG feature-space MSE)
3. Glyphloss: grey-weighted gradient magnitude, direction, and spectral loss
   designed specifically for glyph contour fidelity — replaces the old
   Sobel-magnitude edge loss with decomposed direction+spectral terms that
   distinguish terminal shapes (sharp vs rounded, flared vs pointed).
4. Vector-quantizer losses returned by the model (VQ, commitment, entropy)

The public helper returns both the scalar total loss and an explicit dictionary
of individual terms for TensorBoard logging.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from glyphloss import GlyphReconstructionLoss
from hrothgar.gtok.config import GtokLossWeights


@dataclass
class GtokLossInfo:
    commit_loss: Optional[torch.Tensor]
    entropy_loss: Optional[torch.Tensor]
    codebook_usage: object
    perplexity: Optional[torch.Tensor] = None
    #aux_ar_loss: Optional[torch.Tensor] = None
    character_ce: Optional[torch.Tensor] = None
    font_ce: Optional[torch.Tensor] = None


def _as_scalar_tensor(value: object, *, device: torch.device) -> torch.Tensor:
    """Convert a Python scalar or tensor-like value to a scalar float tensor."""
    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(float(value), device=device)


def compute_gtok_loss(
    reconstructed_images: torch.Tensor,
    target_images: torch.Tensor,
    loss_info: GtokLossInfo,
    *,
    perceptual_loss_fn: Optional[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
    glyphloss_fn: Optional[GlyphReconstructionLoss] = None,
    weights: GtokLossWeights = GtokLossWeights(),
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the full G-Tok training loss and return logging terms.

    Args:
        reconstructed_images: Model reconstruction output,
            shape ``(B, C, H, W)``.
        target_images: Ground-truth target images, shape ``(B, C, H, W)``.
        loss_info: ``GtokLossInfo`` object containing VQ-related losses
            and metrics.
        perceptual_loss_fn: Optional callable (for example
            ``hrothgar.gtok.vgg_loss.VGG``) that takes
            ``(reconstructed_images, target_images)`` and returns a scalar
            tensor.
        glyphloss_fn: Optional ``GlyphReconstructionLoss`` module.  Should
            be instantiated with ``lambda_pixel=0.0`` since L1 handles pixel
            reconstruction separately.
        weights: Coefficients for each term in the final weighted sum.

    Returns:
        Tuple ``(total_loss, terms)`` where:
        - ``total_loss`` is the weighted scalar loss tensor for
          backpropagation.
        - ``terms`` contains scalar tensors for individual components and
          weighted values, suitable for TensorBoard logging.
    """
    l1_loss = F.l1_loss(reconstructed_images, target_images)

    if perceptual_loss_fn is None:
        perceptual_loss = torch.zeros((), device=reconstructed_images.device)
    else:
        perceptual_loss = perceptual_loss_fn(reconstructed_images, target_images)

    if glyphloss_fn is None:
        glyphloss_loss = torch.zeros((), device=reconstructed_images.device)
    else:
        glyphloss_loss = glyphloss_fn(reconstructed_images, target_images)

    def _or_zero(t: Optional[torch.Tensor]) -> torch.Tensor:
        return (
            t if t is not None else torch.zeros((), device=reconstructed_images.device)
        )

    commit_loss = _or_zero(loss_info.commit_loss)
    entropy_loss = _or_zero(loss_info.entropy_loss)
    #aux_ar_loss = _or_zero(loss_info.aux_ar_loss)
    character_ce = _or_zero(loss_info.character_ce)
    font_ce = _or_zero(loss_info.font_ce)

    codebook_usage = _as_scalar_tensor(
        loss_info.codebook_usage, device=reconstructed_images.device
    )
    perplexity = _or_zero(loss_info.perplexity)

    #weighted_aux_ar = weights.aux_ar * _or_zero(loss_info.aux_ar_loss)
    weighted_character_ce = weights.character_ce * _or_zero(loss_info.character_ce)
    weighted_font_ce = weights.font_ce * font_ce
    weighted_l1 = weights.l1 * l1_loss
    weighted_perceptual = weights.perceptual * perceptual_loss
    weighted_glyphloss = weights.glyphloss * glyphloss_loss
    weighted_commit = weights.commit * commit_loss
    weighted_entropy = weights.entropy * entropy_loss

    total_loss = (
        weighted_l1
        + weighted_perceptual
        + weighted_glyphloss
        + weighted_commit
        + weighted_entropy
        #+ weighted_aux_ar
        + weighted_character_ce
        + weighted_font_ce
    )

    terms: Dict[str, torch.Tensor] = {
        "total": total_loss,
        "l1": l1_loss,
        "perceptual": perceptual_loss,
        "glyphloss": glyphloss_loss,
        "commit": commit_loss,
        "entropy": entropy_loss,
        #"aux_ar": aux_ar_loss,
        "character_ce": character_ce,
        "codebook_usage": codebook_usage,
        "perplexity": perplexity,
        "weighted_l1": weighted_l1,
        "weighted_perceptual": weighted_perceptual,
        "weighted_glyphloss": weighted_glyphloss,
        "weighted_commit": weighted_commit,
        "weighted_entropy": weighted_entropy,
        #"weighted_aux_ar": weighted_aux_ar,
        "weighted_character_ce": weighted_character_ce,
        "font_ce": font_ce,
        "weighted_font_ce": weighted_font_ce,
    }

    return total_loss, terms
