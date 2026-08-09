"""Loss functions for font style embedding training.

All loss-computation functions return ``(total_loss, loss_info)`` where
``loss_info`` is a dict of per-term scalar tensors for TensorBoard logging.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from hrothgar.style_embedding.config import FontStyleEmbeddingLossWeights


def multipos_contrastive_loss(
    projections: torch.Tensor,
    text_embeddings: torch.Tensor,
    family_labels: list[str],
    temperature: float = 0.07,
    use_family_positives: bool = True,
    tag_vectors: Optional[torch.Tensor] = None,
    tag_threshold: float = 0.1,
) -> torch.Tensor:
    """Multi-positive contrastive loss with frozen text conditioning.

    Each image anchor (all 2B views) is contrasted against a joint candidate
    set of all 2B images plus B text embeddings.  Positives for anchor *i* are:

    * Its other view (same font, different phrase) — hard positive.
    * The text embedding for its font family — hard positive.
    * When ``tag_vectors`` is provided: other images weighted by tag cosine
      similarity (soft positive).  When absent and ``use_family_positives``
      is True: binary same-family positives.

    The loss is SupCon-style: the negative log-probability of each positive is
    averaged over the positive set (weighted when soft), then averaged over
    all anchors.

    Args:
        projections: ``(2B, D)`` L2-normalized — view 1 stacked on view 2.
        text_embeddings: ``(B, D)`` L2-normalized — one per font.
        family_labels: *2B* family name strings (duplicated for the two views).
        temperature: Softmax temperature (lower = sharper contrast).
        use_family_positives: Ignored when ``tag_vectors`` is provided.
            When True (default) and no tag vectors, same-family images in both
            views are additional hard positives.
        tag_vectors: Optional ``(B, num_tags)`` tag value vectors (centiles
            in [0, 1]).  Cosine similarity between tag vectors provides soft
            positive weights for cross-font image image pairs.
        tag_threshold: Minimum cosine similarity to count as a positive
            (default 0.1).  Below-threshold values are zeroed.

    Returns:
        Scalar loss.
    """
    B = projections.shape[0] // 2
    if B < 2:
        return torch.tensor(0.0, device=projections.device)

    # --- NaN guards -----------------------------------------------------
    def _check(t: torch.Tensor, label: str) -> None:
        if torch.isnan(t).any():
            nan_count = torch.isnan(t).sum().item()
            total = t.numel()
            fine = t[~t.isnan()]
            detail = ""
            if fine.numel() > 0:
                detail = (
                    f", min={fine.min().item():.4f}"
                    f", max={fine.max().item():.4f}"
                )
            raise RuntimeError(
                f"NaN in {label}: {nan_count}/{total} NaN values"
                f" (shape={tuple(t.shape)}{detail})"
            )

    _check(projections, "projections (input)")
    _check(text_embeddings, "text_embeddings (input)")
    if tag_vectors is not None:
        _check(tag_vectors, "tag_vectors (input)")

    N = 2 * B          # image anchors
    M = N + B          # total candidates (images + texts)
    device = projections.device

    # Image-image and image-text similarities.
    sim_img = torch.matmul(projections, projections.T) / temperature   # (2B, 2B)
    sim_txt = torch.matmul(projections, text_embeddings.T) / temperature  # (2B, B)
    _check(sim_img, "sim_img after matmul")
    _check(sim_txt, "sim_txt after matmul")
    sim = torch.cat([sim_img, sim_txt], dim=1)  # (2B, M)
    _check(sim, "sim after concat")

    # ---- Positive mask ---------------------------------------------------
    pos_mask = torch.zeros(N, M, device=device)

    # 1. Same-font, other view: shifted identity (hard positive).
    other = (torch.arange(N, device=device) + B) % N
    pos_mask[torch.arange(N), other] = 1.0

    # 2. Text embedding: column 2B + font index (hard positive).
    font_idx = torch.arange(N, device=device) % B
    pos_mask[torch.arange(N), 2 * B + font_idx] = 1.0

    # 3. Image-image soft positives: tag-weighted when available,
    #    binary same-family otherwise.
    if tag_vectors is not None:
        # Normalize tag vectors and compute cosine similarity.
        tag_norm = F.normalize(tag_vectors.float(), p=2, dim=-1)   # (B, T)
        tag_sim = torch.matmul(tag_norm, tag_norm.T)               # (B, B)
        # Threshold: zero out very weak similarities.
        tag_sim = torch.where(tag_sim > tag_threshold, tag_sim,
                              torch.zeros_like(tag_sim))
        # Duplicate to (2B, 2B) for the two views.
        top = torch.cat([tag_sim, tag_sim], dim=1)   # (B, 2B)
        bot = torch.cat([tag_sim, tag_sim], dim=1)   # (B, 2B)
        img_pos = torch.cat([top, bot], dim=0)        # (2B, 2B)
        # Exclude self and the other-view positive.
        img_pos.fill_diagonal_(0.0)
        for i in range(N):
            img_pos[i, other[i]] = 0.0
        pos_mask[:, :N] += img_pos
    elif use_family_positives:
        import numpy as np

        fams_arr = np.array(family_labels)  # (2B,)
        same_family = torch.tensor(
            fams_arr[:, None] == fams_arr[None, :],
            device=device,
            dtype=torch.float32,
        )  # (2B, 2B)
        same_family.fill_diagonal_(0.0)
        for i in range(N):
            same_family[i, other[i]] = 0.0
        pos_mask[:, :N] += same_family

    # ---- Mask self-similarity in sim ------------------------------------
    sim[torch.arange(N), torch.arange(N)] = float("-inf")
    _check(sim, "sim after masking self")

    # ---- SupCon-format loss: weighted mean over positives --------------
    log_prob = sim.log_softmax(dim=1)                         # (2B, M)
    _check(log_prob, "log_prob after log_softmax")
    # Zero out non-positive positions so 0 * -inf is safe, then
    # weighted-sum by pos_mask values and normalise by total weight.
    log_prob = log_prob.masked_fill(pos_mask == 0, 0.0)
    n_pos = pos_mask.sum(dim=1).clamp(min=1)                  # (2B,)
    loss_per_row = -(pos_mask * log_prob).sum(dim=1) / n_pos  # (2B,)
    return loss_per_row.mean()


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
    text_embeddings: Optional[torch.Tensor] = None,
    tag_vectors: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute combined loss and per-term breakdown.

    Returns:
        ``(total_loss, loss_info)`` for TensorBoard logging.
    """
    device = projections.device

    # Contrastive loss — use multi-positive text-conditioned variant
    # when text embeddings are available.
    if text_embeddings is not None and family_labels is not None:
        contr = multipos_contrastive_loss(
            projections,
            text_embeddings,
            family_labels,
            temperature=temperature,
            use_family_positives=weights.use_family_positives,
            tag_vectors=tag_vectors,
        )
        weight = weights.multipos_contrastive
    else:
        contr = contrastive_loss(
            projections,
            temperature=temperature,
            family_labels=family_labels,
        )
        weight = weights.contrastive

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
        weight * contr
        + weights.tag_prediction * tag_loss
        + cat_loss
    )

    loss_info: dict[str, torch.Tensor] = {
        "contrastive": contr.detach(),
        "contrastive_weighted": (weight * contr).detach(),
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
