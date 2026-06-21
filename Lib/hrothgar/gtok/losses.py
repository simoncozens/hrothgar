"""Loss utilities for G-Tok training.

This module combines reconstruction and quantization losses used to train the
G-Tok tokenizer:

1. Glyph reconstruction loss (via the external ``glyphloss`` package), which
   replaces the previous combination of L1 pixel loss, VGG perceptual loss, and
   Sobel edge loss with a single purpose-built glyph-image loss.
2. Vector-quantizer losses returned by the model (VQ, commitment, entropy)

The public helper returns both the scalar total loss and an explicit dictionary
of individual terms for TensorBoard logging.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch

from hrothgar.gtok.config import GtokLossWeights


@dataclass
class GtokLossInfo:
    vq_loss: Optional[torch.Tensor]
    commit_loss: Optional[torch.Tensor]
    entropy_loss: Optional[torch.Tensor]
    codebook_usage: object  # Can be a scalar or tensor-like value
    perplexity: Optional[torch.Tensor] = None
    aux_ar_loss: Optional[torch.Tensor] = None  # Set by model when aux_ar_head exists
    character_ce: Optional[torch.Tensor] = (
        None  # Set by model when char classifier exists
    )


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
    glyphloss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    weights: GtokLossWeights = GtokLossWeights(),
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the full G-Tok training loss and return logging terms.

    Args:
            reconstructed_images: Model reconstruction output, shape ``(B, C, H, W)``.
            target_images: Ground-truth target images, shape ``(B, C, H, W)``.
            loss_info: ``GtokLossInfo`` object containing VQ-related losses and metrics.
            glyphloss_fn: Callable that takes ``(reconstructed_images, target_images)``
                    and returns a scalar glyph reconstruction loss tensor.  Typically
                    ``GlyphReconstructionLoss`` from the ``glyphloss`` package or the
                    standalone ``glyph_reconstruction_loss`` function.
            weights: Coefficients for each term in the final weighted sum.

    Returns:
            Tuple ``(total_loss, terms)`` where:
            - ``total_loss`` is the weighted scalar loss tensor for backpropagation.
            - ``terms`` contains scalar tensors for individual components and weighted
              values, suitable for TensorBoard logging.
    """
    glyphloss = glyphloss_fn(reconstructed_images, target_images)

    (
        vq_loss_raw,
        commit_loss_raw,
        entropy_loss_raw,
        codebook_usage_raw,
        perplexity_raw,
    ) = (
        loss_info.vq_loss,
        loss_info.commit_loss,
        loss_info.entropy_loss,
        loss_info.codebook_usage,
        loss_info.perplexity,
    )

    vq_loss = (
        vq_loss_raw
        if vq_loss_raw is not None
        else torch.zeros((), device=reconstructed_images.device)
    )
    commit_loss = (
        commit_loss_raw
        if commit_loss_raw is not None
        else torch.zeros((), device=reconstructed_images.device)
    )
    entropy_loss = (
        entropy_loss_raw
        if entropy_loss_raw is not None
        else torch.zeros((), device=reconstructed_images.device)
    )
    codebook_usage = _as_scalar_tensor(
        codebook_usage_raw, device=reconstructed_images.device
    )
    perplexity = (
        perplexity_raw
        if perplexity_raw is not None
        else torch.zeros((), device=reconstructed_images.device)
    )

    # Auxiliary AR loss from the model's next-token prediction head
    aux_ar_loss_raw = loss_info.aux_ar_loss
    aux_ar_loss = (
        aux_ar_loss_raw
        if aux_ar_loss_raw is not None
        else torch.zeros((), device=reconstructed_images.device)
    )
    weighted_aux_ar = weights.aux_ar * aux_ar_loss

    # Character classification CE: encourages codebook to organise by character.
    character_ce_raw = loss_info.character_ce
    character_ce = (
        character_ce_raw
        if character_ce_raw is not None
        else torch.zeros((), device=reconstructed_images.device)
    )
    weighted_character_ce = weights.character_ce * character_ce

    weighted_glyphloss = weights.glyphloss * glyphloss
    weighted_vq = weights.vq * vq_loss
    weighted_commit = weights.commit * commit_loss
    weighted_entropy = weights.entropy * entropy_loss

    total_loss = (
        weighted_glyphloss
        + weighted_vq
        + weighted_commit
        + weighted_entropy
        + weighted_aux_ar
        + weighted_character_ce
    )

    terms: Dict[str, torch.Tensor] = {
        "total": total_loss,
        "glyphloss": glyphloss,
        "vq": vq_loss,
        "commit": commit_loss,
        "entropy": entropy_loss,
        "aux_ar": aux_ar_loss,
        "codebook_usage": codebook_usage,
        "perplexity": perplexity,
        "weighted_glyphloss": weighted_glyphloss,
        "weighted_vq": weighted_vq,
        "weighted_commit": weighted_commit,
        "weighted_entropy": weighted_entropy,
        "weighted_aux_ar": weighted_aux_ar,
        "character_ce": character_ce,
        "weighted_character_ce": weighted_character_ce,
    }

    return total_loss, terms
