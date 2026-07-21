"""Loss utilities for visual-pretraining of the AR generator."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from hrothgar.ar.model import ARModelOutput
from glyphloss import glyph_reconstruction_loss


@dataclass(frozen=True)
class ARLossWeights:
    """Weights for the AR visual-pretraining objectives."""

    token_cross_entropy: float = 0.3
    glyphloss: float = 1.0
    lookahead_cross_entropy: float = 0.1
    width_l1: float = 0.1


@dataclass(frozen=True)
class ARAdaptationLossWeights:
    """Weights for multimodal AR adaptation objectives."""

    alignment_l2: float = 1.0
    token_cross_entropy: float = 0.0
    glyphloss: float = 1.0
    lookahead_cross_entropy: float = 0.0


def compute_ar_loss(
    model_output: ARModelOutput,
    target_images: torch.Tensor,
    *,
    target_token_indices: Optional[torch.Tensor] = None,
    target_widths: Optional[torch.Tensor] = None,
    weights: ARLossWeights = ARLossWeights(),
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

    # Free-running steps have no token targets — skip token CE and lookahead.
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

        lookahead_cross_entropy = torch.tensor(0.0, device=token_targets.device)
        lookahead_logits = getattr(model_output, "lookahead_logits", None)
        if lookahead_logits:
            for k, l_logits in enumerate(lookahead_logits, start=1):
                valid_len = token_targets.shape[1] - k
                if valid_len > 0:
                    lookahead_cross_entropy = lookahead_cross_entropy + F.cross_entropy(
                        l_logits[:, :valid_len, :].reshape(-1, l_logits.shape[-1]),
                        token_targets[:, k:].reshape(-1),
                    )
            lookahead_cross_entropy = lookahead_cross_entropy / len(lookahead_logits)
    else:
        token_cross_entropy = torch.tensor(0.0, device=target_images.device)
        lookahead_cross_entropy = torch.tensor(0.0, device=target_images.device)

    # glyphloss loss
    glyphloss = torch.tensor(0.0, device=target_images.device)
    recon = getattr(model_output, "reconstructed_images", None)
    if recon is not None:
        glyphloss = glyph_reconstruction_loss(recon, target_images)

    weighted_token_cross_entropy = weights.token_cross_entropy * token_cross_entropy
    weighted_lookahead_cross_entropy = (
        weights.lookahead_cross_entropy * lookahead_cross_entropy
    )
    weighted_glyphloss = weights.glyphloss * glyphloss

    # Auxiliary width prediction loss.
    width_l1 = torch.tensor(0.0, device=target_images.device)
    weighted_width_l1 = torch.tensor(0.0, device=target_images.device)
    if (
        weights.width_l1 > 0
        and target_widths is not None
        and model_output.predicted_width is not None
    ):
        width_l1 = F.l1_loss(model_output.predicted_width, target_widths)
        weighted_width_l1 = weights.width_l1 * width_l1

    total_loss = (
        weighted_token_cross_entropy
        + weighted_lookahead_cross_entropy
        + weighted_glyphloss
        + weighted_width_l1
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
        "lookahead_cross_entropy": lookahead_cross_entropy,
        "glyphloss": glyphloss,
        "token_accuracy": token_accuracy,
        "weighted_token_cross_entropy": weighted_token_cross_entropy,
        "weighted_lookahead_cross_entropy": weighted_lookahead_cross_entropy,
        "weighted_glyphloss": weighted_glyphloss,
        "width_l1": width_l1.detach(),
        "weighted_width_l1": weighted_width_l1.detach(),
    }
    if maskgit_mask is not None:
        terms["n_masked"] = maskgit_mask.sum().float()

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
        or weights.glyphloss > 0.0
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
            glyphloss = glyph_reconstruction_loss(
                model_output.reconstructed_images, target_images
            )
            weighted_glyphloss = weights.glyphloss * glyphloss
            total_loss = total_loss + weighted_glyphloss
            terms["glyphloss"] = glyphloss
            terms["weighted_glyphloss"] = weighted_glyphloss

    terms["total"] = total_loss
    return total_loss, terms
