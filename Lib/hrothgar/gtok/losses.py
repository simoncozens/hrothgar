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
    commit_loss: Optional[torch.Tensor]
    entropy_loss: Optional[torch.Tensor]
    codebook_usage: object
    perplexity: Optional[torch.Tensor] = None
    aux_ar_loss: Optional[torch.Tensor] = None
    character_ce: Optional[torch.Tensor] = None


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
    """Compute the full G-Tok training loss and return logging terms."""
    glyphloss = glyphloss_fn(reconstructed_images, target_images)

    def _or_zero(t: Optional[torch.Tensor]) -> torch.Tensor:
        return (
            t if t is not None else torch.zeros((), device=reconstructed_images.device)
        )

    commit_loss = _or_zero(loss_info.commit_loss)
    entropy_loss = _or_zero(loss_info.entropy_loss)
    aux_ar_loss = _or_zero(loss_info.aux_ar_loss)
    character_ce = _or_zero(loss_info.character_ce)

    codebook_usage = _as_scalar_tensor(
        loss_info.codebook_usage, device=reconstructed_images.device
    )
    perplexity = _or_zero(loss_info.perplexity)

    total_loss = (
        weights.glyphloss * glyphloss
        + commit_loss
        + entropy_loss
        + weights.aux_ar * aux_ar_loss
        + weights.character_ce * character_ce
    )

    terms: Dict[str, torch.Tensor] = {
        "total": total_loss,
        "glyphloss": glyphloss,
        "commit": commit_loss,
        "entropy": entropy_loss,
        "aux_ar": aux_ar_loss,
        "character_ce": character_ce,
        "codebook_usage": codebook_usage,
        "perplexity": perplexity,
    }

    return total_loss, terms
