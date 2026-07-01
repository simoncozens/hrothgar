"""Loss utilities for visual-pretraining of the AR generator."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from hrothgar.ar.model import ARAdaptationOutput, ARModelOutput


@dataclass(frozen=True)
class ARLossWeights:
    """Weights for the AR visual-pretraining objectives."""

    token_cross_entropy: float = 0.3
    pixel_l1: float = 1.0
    lookahead_cross_entropy: float = 0.1
    perceptual_lpips: float = 2.0


@dataclass(frozen=True)
class ARAdaptationLossWeights:
    """Weights for multimodal AR adaptation objectives."""

    alignment_l2: float = 1.0
    token_cross_entropy: float = 0.0
    pixel_l1: float = 0.0
    lookahead_cross_entropy: float = 0.0


def compute_ar_loss(
    model_output: ARModelOutput,
    target_images: torch.Tensor,
    *,
    target_token_indices: Optional[torch.Tensor] = None,
    weights: ARLossWeights = ARLossWeights(),
    lpips_metric: Optional[object] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute AR visual-pretraining loss and loggable terms.

    The paper objective for this stage combines token-level cross-entropy and
    pixel-level L1 reconstruction.

    When ``model_output.perceptual_recon`` is provided and
    ``weights.perceptual_lpips > 0``, an auxiliary LPIPS loss is computed
    between the Gumbel-softmax decoded image and the ground truth.  This
    directly addresses the many-to-one mapping problem by teaching the model
    that different code sequences are valid if they decode to the same image.

    Args:
        model_output: Output from ``ARModel.forward``.
        target_images: Ground-truth target glyph images, shape ``(B, C, H, W)``.
        target_token_indices: Optional explicit token targets. If omitted, this
            function uses ``model_output.target_token_indices``.
        weights: Weights for token and pixel loss terms.
        lpips_metric: Optional LPIPS module instance. Required when
            ``weights.perceptual_lpips > 0`` and
            ``model_output.perceptual_recon`` is not None.

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

    lookahead_cross_entropy = torch.tensor(0.0, device=token_targets.device)
    if model_output.lookahead_logits:
        for k, l_logits in enumerate(model_output.lookahead_logits, start=1):
            # l_logits[:, t] predicts token_targets[:, t+k]
            valid_len = token_targets.shape[1] - k
            if valid_len > 0:
                lookahead_cross_entropy = lookahead_cross_entropy + F.cross_entropy(
                    l_logits[:, :valid_len, :].reshape(-1, l_logits.shape[-1]),
                    token_targets[:, k:].reshape(-1),
                )
        lookahead_cross_entropy = lookahead_cross_entropy / len(
            model_output.lookahead_logits
        )

    pixel_l1 = F.l1_loss(model_output.reconstructed_images, target_images)

    # Perceptual LPIPS loss on Gumbel-softmax sampled reconstruction.
    perceptual_lpips = torch.tensor(0.0, device=token_targets.device)
    if weights.perceptual_lpips > 0 and model_output.perceptual_recon is not None:
        if lpips_metric is None:
            raise ValueError(
                "lpips_metric is required when perceptual_lpips > 0 "
                "and perceptual_recon is provided"
            )
        # Clamp to [0, 1] for LPIPS (decoder output can drift slightly).
        perceptual_recon_clamped = torch.clamp(model_output.perceptual_recon, 0.0, 1.0)
        target_clamped = torch.clamp(target_images, 0.0, 1.0)
        perceptual_lpips = lpips_metric(perceptual_recon_clamped, target_clamped).mean()

    weighted_token_cross_entropy = weights.token_cross_entropy * token_cross_entropy
    weighted_lookahead_cross_entropy = (
        weights.lookahead_cross_entropy * lookahead_cross_entropy
    )
    weighted_pixel_l1 = weights.pixel_l1 * pixel_l1
    weighted_perceptual_lpips = weights.perceptual_lpips * perceptual_lpips
    total_loss = (
        weighted_token_cross_entropy
        + weighted_lookahead_cross_entropy
        + weighted_pixel_l1
        + weighted_perceptual_lpips
    )

    token_predictions = torch.argmax(model_output.logits, dim=-1)
    token_accuracy = (token_predictions == token_targets).float().mean()

    terms: Dict[str, torch.Tensor] = {
        "total": total_loss,
        "token_cross_entropy": token_cross_entropy,
        "lookahead_cross_entropy": lookahead_cross_entropy,
        "pixel_l1": pixel_l1,
        "perceptual_lpips": perceptual_lpips,
        "token_accuracy": token_accuracy,
        "weighted_token_cross_entropy": weighted_token_cross_entropy,
        "weighted_lookahead_cross_entropy": weighted_lookahead_cross_entropy,
        "weighted_pixel_l1": weighted_pixel_l1,
        "weighted_perceptual_lpips": weighted_perceptual_lpips,
    }

    return total_loss, terms


def compute_ar_adaptation_loss(
    model_output: ARAdaptationOutput,
    *,
    target_images: Optional[torch.Tensor] = None,
    target_token_indices: Optional[torch.Tensor] = None,
    weights: ARAdaptationLossWeights = ARAdaptationLossWeights(),
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute multimodal adaptation loss and loggable terms.

    Core objective is L2 alignment between visual-only and multimodal
    aggregated style tokens. Optionally adds decoder supervision terms when
    decoder outputs are available.
    """
    alignment_l2 = F.mse_loss(
        model_output.multimodal_aggregated_style_tokens,
        model_output.visual_aggregated_style_tokens,
    )
    total_loss = weights.alignment_l2 * alignment_l2

    terms: Dict[str, torch.Tensor] = {
        "alignment_l2": alignment_l2,
        "weighted_alignment_l2": weights.alignment_l2 * alignment_l2,
    }

    has_decoder_outputs = (
        model_output.logits is not None
        and model_output.reconstructed_images is not None
    )
    decoder_requested = (
        weights.token_cross_entropy > 0.0
        or weights.pixel_l1 > 0.0
        or weights.lookahead_cross_entropy > 0.0
    )

    if decoder_requested and not has_decoder_outputs:
        raise ValueError(
            "Decoder-weighted adaptation loss requested, but model output does not contain decoder tensors. "
            "Run forward_adaptation(..., run_decoder=True)."
        )

    if has_decoder_outputs:
        token_targets = target_token_indices
        if token_targets is None:
            token_targets = model_output.target_token_indices

        if token_targets is not None:
            token_cross_entropy = F.cross_entropy(
                model_output.logits.reshape(-1, model_output.logits.shape[-1]),
                token_targets.reshape(-1),
            )
            weighted_token_cross_entropy = (
                weights.token_cross_entropy * token_cross_entropy
            )
            total_loss = total_loss + weighted_token_cross_entropy
            terms["token_cross_entropy"] = token_cross_entropy
            terms["weighted_token_cross_entropy"] = weighted_token_cross_entropy

            token_predictions = torch.argmax(model_output.logits, dim=-1)
            token_accuracy = (token_predictions == token_targets).float().mean()
            terms["token_accuracy"] = token_accuracy

            if model_output.lookahead_logits:
                lookahead_ce = torch.tensor(0.0, device=token_targets.device)
                for k, l_logits in enumerate(model_output.lookahead_logits, start=1):
                    valid_len = token_targets.shape[1] - k
                    if valid_len > 0:
                        lookahead_ce = lookahead_ce + F.cross_entropy(
                            l_logits[:, :valid_len, :].reshape(-1, l_logits.shape[-1]),
                            token_targets[:, k:].reshape(-1),
                        )
                lookahead_ce = lookahead_ce / len(model_output.lookahead_logits)
                weighted_lookahead_ce = weights.lookahead_cross_entropy * lookahead_ce

                total_loss = total_loss + weighted_lookahead_ce
                terms["lookahead_cross_entropy"] = lookahead_ce
                terms["weighted_lookahead_cross_entropy"] = weighted_lookahead_ce

        if target_images is not None:
            pixel_l1 = F.l1_loss(model_output.reconstructed_images, target_images)
            weighted_pixel_l1 = weights.pixel_l1 * pixel_l1
            total_loss = total_loss + weighted_pixel_l1
            terms["pixel_l1"] = pixel_l1
            terms["weighted_pixel_l1"] = weighted_pixel_l1

    terms["total"] = total_loss
    return total_loss, terms
