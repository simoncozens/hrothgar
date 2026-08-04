"""Loss functions for font style embedding training.

All loss-computation functions return ``(total_loss, loss_info)`` where
``loss_info`` is a dict of per-term scalar tensors for TensorBoard logging.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from hrothgar.style_embedding.config import FontStyleEmbeddingLossWeights


def contrastive_loss(
    projections: torch.Tensor,
    temperature: float = 0.07,
    family_labels: Optional[list[str]] = None,
) -> torch.Tensor:
    """NT-Xent contrastive loss with same-family negative masking.

    Each font should be similar to itself across different renderings.
    Different weights/styles of the *same* font family should NOT be treated
    as negatives — they are masked out of the negative set.

    Args:
        projections: ``(2*B, projection_dim)`` — L2-normalized projections
            from two views of B fonts.  First B = view 1, second B = view 2.
        temperature: Softmax temperature (lower = sharper contrast).
        family_labels: Optional list of `B` family name strings.  When
            provided, any negative pair from the same family is masked out.

    Returns:
        Scalar loss.
    """
    B = projections.shape[0] // 2
    if B < 2:
        return torch.tensor(0.0, device=projections.device)

    # Similarity matrix: (2B, 2B).
    sim = torch.matmul(projections, projections.T) / temperature

    # Positive pairs: (i, i+B) and (i+B, i).
    labels = torch.arange(B, device=projections.device)
    labels = torch.cat([labels + B, labels], dim=0)  # (2B,)

    # Build mask: True = exclude from softmax.
    # 1. Self-similarity: mask out the diagonal.
    mask = torch.eye(2 * B, device=projections.device, dtype=torch.bool)

    # 2. Same-family negatives: mask out same-family pairs (excluding positives).
    if family_labels is not None:
        # Duplicate family labels for the two views.
        fams = family_labels + family_labels  # (2B,)
        for i in range(2 * B):
            for j in range(2 * B):
                if i == j:
                    continue
                # Don't mask the positive pairs.
                is_positive = (
                    (i < B and j == i + B)
                    or (i >= B and j == i - B)
                )
                if not is_positive and fams[i] == fams[j]:
                    mask[i, j] = True

    sim = sim.masked_fill(mask, float("-inf"))

    loss = F.cross_entropy(sim, labels)
    return loss


def tag_prediction_loss(
    predicted: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Masked tag prediction loss.

    When predicted values are ``(B,)`` scalars → MSE regression.
    When predicted values are ``(B, C)`` logits → cross-entropy classification.

    Only samples where ``masks[name]`` is True contribute to the loss.
    """
    total_loss = torch.tensor(0.0, device=next(iter(predicted.values())).device)
    total_weight = torch.tensor(0.0, device=total_loss.device)
    for name, pred in predicted.items():
        if name not in targets or name not in masks:
            continue
        m = masks[name]
        if not m.any():
            continue
        if pred.ndim == 2:
            # Classification: (B, C) logits → cross-entropy.
            target = targets[name][m].long()
            loss = F.cross_entropy(pred[m], target)
        else:
            # Regression: (B,) scalars → MSE.
            se = (pred[m] - targets[name][m]) ** 2
            loss = se.mean()
        total_loss = total_loss + loss
        total_weight = total_weight + 1.0
    if total_weight == 0:
        return total_loss
    return total_loss / total_weight


def category_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy loss and accuracy for broad category classification.

    Args:
        logits: ``(B, 6)`` category logits.
        targets: ``(B,)`` integer category indices (0–5).

    Returns:
        ``(loss, accuracy)`` where accuracy is the fraction of correct predictions.
    """
    loss = F.cross_entropy(logits, targets)
    accuracy = (logits.argmax(dim=-1) == targets).float().mean()
    return loss, accuracy


def compute_losses(
    projections: torch.Tensor,
    predicted_tags: Optional[dict[str, torch.Tensor]],
    target_tags: dict[str, torch.Tensor],
    *,
    weights: FontStyleEmbeddingLossWeights,
    temperature: float = 0.07,
    tag_masks: Optional[dict[str, torch.Tensor]] = None,
    family_labels: Optional[list[str]] = None,
    category_logits: Optional[torch.Tensor] = None,
    category_targets: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute combined loss and per-term breakdown.

    Returns:
        ``(total_loss, loss_info)`` for TensorBoard logging.
    """
    device = projections.device

    # Contrastive loss.
    contr = contrastive_loss(
        projections,
        temperature=temperature,
        family_labels=family_labels,
    )

    # Tag prediction loss.
    tag_loss = torch.tensor(0.0, device=device)
    if predicted_tags is not None and target_tags:
        tag_loss = tag_prediction_loss(
            predicted_tags, target_tags,
            masks=tag_masks or {},
        )

    # Category classification loss.
    cat_loss = torch.tensor(0.0, device=device)
    cat_accuracy = torch.tensor(0.0, device=device)
    if category_logits is not None and category_targets is not None:
        cat_loss, cat_accuracy = category_loss(category_logits, category_targets)

    total = (
        weights.contrastive * contr
        + weights.tag_prediction * tag_loss
        + cat_loss
    )

    loss_info: dict[str, torch.Tensor] = {
        "contrastive": contr.detach(),
        "contrastive_weighted": (weights.contrastive * contr).detach(),
        "loss": total.detach(),
    }
    if predicted_tags is not None and target_tags:
        loss_info["tag_prediction"] = tag_loss.detach()
        loss_info["tag_prediction_weighted"] = (weights.tag_prediction * tag_loss).detach()
        # Per-tag accuracy or MSE for diagnostics.
        for name, pred in predicted_tags.items():
            if name not in target_tags:
                continue
            m = (tag_masks or {}).get(name, torch.ones(pred.shape[0], dtype=torch.bool, device=device))
            if not m.any():
                continue
            safe_name = name.replace("/", "_").strip("_")
            if pred.ndim == 2:
                # Classification: per-tag accuracy.
                correct = (pred[m].argmax(dim=-1) == target_tags[name][m].long()).float().mean()
                loss_info[f"tag_acc_{safe_name}"] = correct.detach()
            else:
                # Regression: per-tag MSE.
                se = ((pred[m] - target_tags[name][m]) ** 2).mean()
                loss_info[f"tag_mse_{safe_name}"] = se.detach()
    if category_logits is not None and category_targets is not None:
        loss_info["category_loss"] = cat_loss.detach()
        loss_info["category_accuracy"] = cat_accuracy.detach()

    return total, loss_info
