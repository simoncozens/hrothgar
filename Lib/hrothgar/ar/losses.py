"""Loss utilities for visual-pretraining of the AR generator."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from glyphloss import glyph_reconstruction_loss

from hrothgar.ar.model import ARModelOutput
from hrothgar.glyphloss_curvature import CurvatureWeightedGlyphLoss


# Curvature- and spectral-weighted glyphloss, matching GTok. The default
# glyphloss is a *mean* over the image, so the few terminal/corner pixels
# contribute negligible absolute loss and the optimizer ignores them. The
# curvature mask (k=20) amplifies those pixels, and lambda_spectral=2.5
# upweights high-frequency edge/terminal detail. lambda_pixel=0 because the
# raw-pixel term is handled separately by ``pixel_l1``.
_GLYPHLOSS_FN = CurvatureWeightedGlyphLoss(
    k=20.0,
    lambda_pixel=0.0,
    lambda_spectral=2.5,
)


@dataclass(frozen=True)
class ARLossWeights:
    """Weights for the AR visual-pretraining objectives."""

    token_cross_entropy: float = 2.0
    pixel_l1: float = 0.5
    glyphloss: float = 1.0
    perceptual_lpips: float = 1.0
    bbox_l1: float = 0.1


@dataclass(frozen=True)
class ARAdaptationLossWeights:
    """Weights for multimodal AR adaptation objectives."""

    alignment_l2: float = 1.0
    token_cross_entropy: float = 0.0
    pixel_l1: float = 0.0
    glyphloss: float = 1.0


def compute_ar_loss(
    model_output: ARModelOutput,
    target_images: torch.Tensor,
    *,
    target_token_indices: Optional[torch.Tensor] = None,
    target_bbox: Optional[torch.Tensor] = None,
    weights: ARLossWeights = ARLossWeights(),
    lpips_metric: Optional[object] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute AR visual-pretraining loss and loggable terms.

    The paper objective for this stage combines token-level cross-entropy and
    pixel-level L1 reconstruction.

    When ``weights.perceptual_lpips > 0``, an auxiliary LPIPS loss is computed
    between the soft-decoded reconstructed image and the ground truth.
    This provides perceptual-level supervision on top of the L1 pixel loss,
    helping the model produce sharper, more visually coherent glyphs.

    Args:
        model_output: Output from ``ARModel.forward``.
        target_images: Ground-truth target glyph images, shape ``(B, C, H, W)``.
        target_token_indices: Optional explicit token targets. If omitted, this
            function uses ``model_output.target_token_indices``.
        weights: Weights for token and pixel loss terms.
        lpips_metric: Optional LPIPS module instance. Required when
            ``weights.perceptual_lpips > 0``.

    Returns:
        ``(total_loss, terms)`` where terms is suitable for TensorBoard logging.
    """
    token_targets = target_token_indices
    if token_targets is None:
        token_targets = model_output.target_token_indices

    # Free-running steps have no token targets — skip token CE.
    has_token_targets = token_targets is not None

    # MaskGIT: token_mask indicates which positions were masked during
    # training (and should contribute to CE loss).
    maskgit_mask = model_output.token_mask

    if has_token_targets:
        if model_output.logits.shape[:2] != token_targets.shape:
            raise ValueError(
                "Logit and target token shapes must match in batch and sequence dimensions "
                f"(got logits {tuple(model_output.logits.shape[:2])}, targets {tuple(token_targets.shape)})"
            )

        if maskgit_mask is not None:
            # MaskGIT: CE computed only on masked positions.
            n_masked = maskgit_mask.sum()
            if n_masked > 0:
                masked_logits = model_output.logits[maskgit_mask]
                masked_targets = token_targets[maskgit_mask]
                token_cross_entropy = F.cross_entropy(masked_logits, masked_targets)
            else:
                token_cross_entropy = torch.tensor(0.0, device=token_targets.device)
        else:
            # AR: CE computed on all positions.
            token_cross_entropy = F.cross_entropy(
                model_output.logits.reshape(-1, model_output.logits.shape[-1]),
                token_targets.reshape(-1),
            )
    else:
        token_cross_entropy = torch.tensor(0.0, device=target_images.device)

    pixel_l1 = F.l1_loss(model_output.reconstructed_images, target_images)

    # Perceptual LPIPS loss on soft-decoded reconstruction.
    perceptual_lpips = torch.tensor(0.0, device=target_images.device)
    if weights.perceptual_lpips > 0:
        if lpips_metric is None:
            raise ValueError(
                "lpips_metric is required when perceptual_lpips > 0"
            )
        # Clamp to [0, 1] for LPIPS (decoder output can drift slightly).
        recon_clamped = torch.clamp(model_output.reconstructed_images, 0.0, 1.0)
        target_clamped = torch.clamp(target_images, 0.0, 1.0)
        perceptual_lpips = lpips_metric(recon_clamped, target_clamped).mean()
    # glyphloss loss
    glyphloss = torch.tensor(0.0, device=target_images.device)
    recon = getattr(model_output, "reconstructed_images", None)
    if recon is not None:
        glyphloss = _GLYPHLOSS_FN(recon, target_images)

    weighted_token_cross_entropy = weights.token_cross_entropy * token_cross_entropy
    weighted_pixel_l1 = weights.pixel_l1 * pixel_l1
    weighted_perceptual_lpips = weights.perceptual_lpips * perceptual_lpips
    weighted_glyphloss = weights.glyphloss * glyphloss

    # Auxiliary ink-bbox (width, height) prediction loss.
    bbox_l1 = torch.tensor(0.0, device=target_images.device)
    weighted_bbox_l1 = torch.tensor(0.0, device=target_images.device)
    if (
        weights.bbox_l1 > 0
        and target_bbox is not None
        and model_output.predicted_bbox is not None
    ):
        bbox_l1 = F.l1_loss(model_output.predicted_bbox, target_bbox)
        weighted_bbox_l1 = weights.bbox_l1 * bbox_l1

    total_loss = (
        weighted_token_cross_entropy
        + weighted_pixel_l1
        + weighted_perceptual_lpips
        + weighted_glyphloss
        + weighted_bbox_l1
    )

    token_accuracy = torch.tensor(0.0, device=target_images.device)
    if has_token_targets:
        token_predictions = torch.argmax(model_output.logits, dim=-1)
        if maskgit_mask is not None:
            # Accuracy on masked positions only.
            if maskgit_mask.sum() > 0:
                token_accuracy = (
                    (token_predictions[maskgit_mask] == token_targets[maskgit_mask])
                    .float()
                    .mean()
                )
        else:
            token_accuracy = (token_predictions == token_targets).float().mean()

    terms: Dict[str, torch.Tensor] = {
        "total": total_loss,
        "token_cross_entropy": token_cross_entropy,
        "pixel_l1": pixel_l1,
        "perceptual_lpips": perceptual_lpips,
        "glyphloss": glyphloss,
        "token_accuracy": token_accuracy,
        "weighted_token_cross_entropy": weighted_token_cross_entropy,
        "weighted_pixel_l1": weighted_pixel_l1,
        "weighted_perceptual_lpips": weighted_perceptual_lpips,
        "weighted_glyphloss": weighted_glyphloss,
        "bbox_l1": bbox_l1.detach(),
        "weighted_bbox_l1": weighted_bbox_l1.detach(),
    }

    return total_loss, terms


def compute_ar_adaptation_loss(
    model_output,  # ARAdaptationOutput (deprecated — kept for import compatibility)
    *,
    target_images=None,
    target_token_indices=None,
    weights=None,
):
    """Compute adaptation loss (deprecated — AR decoder removed).

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
        or weights.glyphloss > 0.0
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

        if target_images is not None:
            pixel_l1 = F.l1_loss(model_output.reconstructed_images, target_images)
            weighted_pixel_l1 = weights.pixel_l1 * pixel_l1
            total_loss = total_loss + weighted_pixel_l1
            terms["pixel_l1"] = pixel_l1
            terms["weighted_pixel_l1"] = weighted_pixel_l1
            glyphloss = glyph_reconstruction_loss(
                model_output.reconstructed_images, target_images
            )
            weighted_glyphloss = weights.glyphloss * glyphloss
            total_loss = total_loss + weighted_glyphloss
            terms["glyphloss"] = glyphloss
            terms["weighted_glyphloss"] = weighted_glyphloss

    terms["total"] = total_loss
    return total_loss, terms
