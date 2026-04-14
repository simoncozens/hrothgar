"""Loss utilities for visual-pretraining of the AR generator."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from hrothgar.ar.model import ARModelOutput


@dataclass(frozen=True)
class ARLossWeights:
    """Weights for the AR visual-pretraining objectives."""

    token_cross_entropy: float = 1.0
    pixel_l1: float = 1.0


def compute_ar_loss(
    model_output: ARModelOutput,
    target_images: torch.Tensor,
    *,
    target_token_indices: Optional[torch.Tensor] = None,
    weights: ARLossWeights = ARLossWeights(),
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute AR visual-pretraining loss and loggable terms.

    The paper objective for this stage combines token-level cross-entropy and
    pixel-level L1 reconstruction.

    Args:
        model_output: Output from ``ARModel.forward``.
        target_images: Ground-truth target glyph images, shape ``(B, C, H, W)``.
        target_token_indices: Optional explicit token targets. If omitted, this
            function uses ``model_output.target_token_indices``.
        weights: Weights for token and pixel loss terms.

    Returns:
        ``(total_loss, terms)`` where terms is suitable for TensorBoard logging.
    """
    token_targets = target_token_indices
    if token_targets is None:
        token_targets = model_output.target_token_indices
    if token_targets is None:
        raise ValueError(
            "target_token_indices must be provided either explicitly or via ARModelOutput"
        )

    if model_output.logits.shape[:2] != token_targets.shape:
        raise ValueError(
            "Logit and target token shapes must match in batch and sequence dimensions "
            f"(got logits {tuple(model_output.logits.shape[:2])}, targets {tuple(token_targets.shape)})"
        )

    token_cross_entropy = F.cross_entropy(
        model_output.logits.reshape(-1, model_output.logits.shape[-1]),
        token_targets.reshape(-1),
    )
    pixel_l1 = F.l1_loss(model_output.reconstructed_images, target_images)

    weighted_token_cross_entropy = weights.token_cross_entropy * token_cross_entropy
    weighted_pixel_l1 = weights.pixel_l1 * pixel_l1
    total_loss = weighted_token_cross_entropy + weighted_pixel_l1

    token_predictions = torch.argmax(model_output.logits, dim=-1)
    token_accuracy = (token_predictions == token_targets).float().mean()

    terms: Dict[str, torch.Tensor] = {
        "total": total_loss,
        "token_cross_entropy": token_cross_entropy,
        "pixel_l1": pixel_l1,
        "token_accuracy": token_accuracy,
        "weighted_token_cross_entropy": weighted_token_cross_entropy,
        "weighted_pixel_l1": weighted_pixel_l1,
    }

    return total_loss, terms
