"""Loss utilities for G-Tok training.

This module combines reconstruction and quantization losses used to train the
G-Tok tokenizer:

1. L1 reconstruction loss between reconstructed and target glyph images
2. Optional perceptual loss (e.g. VGG feature-space MSE)
3. Edge/gradient loss: Sobel-filtered gradient magnitude difference, which
   penalises contour blurring more strongly than pixel-space L1 alone.
4. Vector-quantizer losses returned by the model (VQ, commitment, entropy)

The public helper returns both the scalar total loss and an explicit dictionary
of individual terms for TensorBoard logging.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from hrothgar.gtok.config import GtokLossWeights
import torch
import torch.nn.functional as F


def _sobel_gradient_magnitude(images: torch.Tensor) -> torch.Tensor:
    """Compute per-pixel Sobel gradient magnitude for a batch of images.

    Each spatial channel is filtered independently.  The result has the same
    shape as the input and represents the local edge strength at each pixel.

    Args:
        images: Float tensor of shape ``(B, C, H, W)``.

    Returns:
        Gradient magnitude tensor of shape ``(B, C, H, W)``.
    """
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=images.dtype,
        device=images.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=images.dtype,
        device=images.device,
    ).view(1, 1, 3, 3)

    B, C, H, W = images.shape
    flat = images.view(B * C, 1, H, W)
    grad_x = F.conv2d(flat, sobel_x, padding=1)
    grad_y = F.conv2d(flat, sobel_y, padding=1)
    # Add small epsilon to avoid zero-gradient sqrt instability.
    magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    return magnitude.view(B, C, H, W)


@dataclass
class GtokLossInfo:
    vq_loss: Optional[torch.Tensor]
    commit_loss: Optional[torch.Tensor]
    entropy_loss: Optional[torch.Tensor]
    codebook_usage: object  # Can be a scalar or tensor-like value
    perplexity: Optional[torch.Tensor] = None


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
    weights: GtokLossWeights = GtokLossWeights(),
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the full G-Tok training loss and return logging terms.

    Args:
            reconstructed_images: Model reconstruction output, shape ``(B, C, H, W)``.
            target_images: Ground-truth target images, shape ``(B, C, H, W)``.
            loss_info: ``GtokLossInfo`` object containing VQ-related losses and metrics.
            perceptual_loss_fn: Optional callable (for example ``hrothgar.gtok.vgg_loss.VGG``)
                    that takes ``(reconstructed_images, target_images)`` and returns a scalar tensor.
            weights: Coefficients for each term in the final weighted sum.

    Returns:
            Tuple ``(total_loss, terms)`` where:
            - ``total_loss`` is the weighted scalar loss tensor for backpropagation.
            - ``terms`` contains scalar tensors for individual components and weighted
              values, suitable for TensorBoard logging.
    """
    l1_loss = F.l1_loss(reconstructed_images, target_images)

    if perceptual_loss_fn is None:
        perceptual_loss = torch.zeros((), device=reconstructed_images.device)
    else:
        perceptual_loss = perceptual_loss_fn(reconstructed_images, target_images)

    recon_edges = _sobel_gradient_magnitude(reconstructed_images)
    target_edges = _sobel_gradient_magnitude(target_images)
    edge_loss = F.l1_loss(recon_edges, target_edges)

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

    weighted_l1 = weights.l1 * l1_loss
    weighted_perceptual = weights.perceptual * perceptual_loss
    weighted_edge = weights.edge * edge_loss
    weighted_vq = weights.vq * vq_loss
    weighted_commit = weights.commit * commit_loss
    weighted_entropy = weights.entropy * entropy_loss

    total_loss = (
        weighted_l1
        + weighted_perceptual
        + weighted_edge
        + weighted_vq
        + weighted_commit
        + weighted_entropy
    )

    terms: Dict[str, torch.Tensor] = {
        "total": total_loss,
        "l1": l1_loss,
        "perceptual": perceptual_loss,
        "edge": edge_loss,
        "vq": vq_loss,
        "commit": commit_loss,
        "entropy": entropy_loss,
        "codebook_usage": codebook_usage,
        "perplexity": perplexity,
        "weighted_l1": weighted_l1,
        "weighted_perceptual": weighted_perceptual,
        "weighted_edge": weighted_edge,
        "weighted_vq": weighted_vq,
        "weighted_commit": weighted_commit,
        "weighted_entropy": weighted_entropy,
    }

    return total_loss, terms
