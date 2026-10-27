"""Loss functions for font style embedding training.

All loss-computation functions return ``(total_loss, loss_info)`` where
``loss_info`` is a dict of per-term scalar tensors for TensorBoard logging.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hrothgar.similarity.config import FontStyleEmbeddingLossWeights


def _positive_mask(
    projections: torch.Tensor,
    family_labels: list[str],
    *,
    font_ids: list[str] | None,
    tag_vectors: torch.Tensor | None,
    text_embeddings: torch.Tensor | None,
    family_positive_weight: float,
    tag_weight: float,
    tag_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(sim, pos_mask)`` over the joint candidate set.

    ``sim`` is raw cosine similarity (no temperature) between the ``2B`` image
    anchors and the ``2B`` images (plus ``B`` text embeddings when provided).
    ``pos_mask`` is the corresponding weighted positive mask: same-font
    other-view (1.0), text positive (1.0), tag soft positives (``tag_weight`` ×
    tag cosine) and same-family different-font (``family_positive_weight``).
    The diagonal (self) is zeroed so it is neither positive nor negative.
    """
    N = projections.shape[0]
    B = N // 2
    device = projections.device

    sim_img = torch.matmul(projections, projections.T)  # (N, N)
    has_text = text_embeddings is not None
    if has_text:
        sim_txt = torch.matmul(projections, text_embeddings.T)  # (N, B)
        sim = torch.cat([sim_img, sim_txt], dim=1)  # (N, M)
    else:
        sim = sim_img
    M = sim.shape[1]

    pos_mask = torch.zeros(N, M, device=device)

    # 1. Same-font, other view (hard positive).
    other = (torch.arange(N, device=device) + B) % N
    pos_mask[torch.arange(N), other] = 1.0

    # 2. Text embedding (hard positive).
    if has_text:
        font_idx = torch.arange(N, device=device) % B
        pos_mask[torch.arange(N), 2 * B + font_idx] = 1.0

    # 3. Tag soft positives.
    if tag_vectors is not None:
        tag_norm = F.normalize(tag_vectors.float(), p=2, dim=-1)  # (B, T)
        tag_sim = torch.matmul(tag_norm, tag_norm.T)  # (B, B)
        tag_sim = torch.where(
            tag_sim > tag_threshold, tag_sim, torch.zeros_like(tag_sim)
        )
        top = torch.cat([tag_sim, tag_sim], dim=1)  # (B, 2B)
        bot = torch.cat([tag_sim, tag_sim], dim=1)  # (B, 2B)
        img_pos = torch.cat([top, bot], dim=0)  # (2B, 2B)
        img_pos.fill_diagonal_(0.0)
        for i in range(N):
            img_pos[i, other[i]] = 0.0
        pos_mask[:, :N] += tag_weight * img_pos

    # 4. Same-family, different-font (soft positive).
    if font_ids is not None and family_positive_weight > 0:
        import numpy as np

        fams_arr = np.array(family_labels)  # (2B,)
        fids_arr = np.array(font_ids)  # (2B,)
        same_family = torch.tensor(
            fams_arr[:, None] == fams_arr[None, :],
            device=device,
            dtype=torch.float32,
        )  # (2B, 2B)
        diff_font = torch.tensor(
            fids_arr[:, None] != fids_arr[None, :],
            device=device,
            dtype=torch.float32,
        )  # (2B, 2B)
        pos_mask[:, :N] += family_positive_weight * (same_family * diff_font)

    # Self is neither a positive nor a negative.
    pos_mask[torch.arange(N), torch.arange(N)] = 0.0

    return sim, pos_mask


def multipos_contrastive_loss(
    projections: torch.Tensor,
    family_labels: list[str],
    *,
    font_ids: list[str] | None = None,
    text_embeddings: torch.Tensor | None = None,
    tag_vectors: torch.Tensor | None = None,
    temperature: float = 0.07,
    family_positive_weight: float = 0.3,
    tag_threshold: float = 0.1,
    tag_weight: float = 1.0,
) -> torch.Tensor:
    """Multi-positive contrastive loss with font-/family-/text-/tag conditioning.

    Each image anchor (all 2B views) is contrasted against a joint candidate
    set of all 2B images plus (optionally) B text embeddings.  Positives:

    * Its other view (same font, different phrase) — hard positive (weight 1).
    * Same family, *different* font — soft positive (weight
      ``family_positive_weight``), so a font is closer to its family than to
      unrelated fonts, but not as close as it is to itself.
    * When ``text_embeddings`` is provided: the text embedding for its font
      family — hard positive.
    * When ``tag_vectors`` is provided: other images weighted by tag cosine
      similarity (soft positive).

    Args:
        projections: ``(2B, D)`` L2-normalized — view 1 stacked on view 2.
        family_labels: *2B* family name strings (duplicated for the two views).
        font_ids: Optional *2B* font identifiers (duplicated for the two views).
            Required for the same-family/different-font soft positive.
        text_embeddings: Optional ``(B, D)`` L2-normalized.
        tag_vectors: Optional ``(B, num_tags)`` tag centiles in [0, 1].
        temperature: Softmax temperature.
        family_positive_weight: Soft-positive weight for same-family,
            different-font pairs (0 disables it).
        tag_threshold: Minimum cosine similarity for tag-based positives.
        tag_weight: Weight of tag-based soft positives.

    Returns:
        Scalar loss.
    """
    B = projections.shape[0] // 2
    if B < 2:
        return torch.tensor(0.0, device=projections.device)

    # # --- NaN guards -----------------------------------------------------
    # def _check(t: torch.Tensor, label: str) -> None:
    #     if torch.isnan(t).any():
    #         nan_count = torch.isnan(t).sum().item()
    #         total = t.numel()
    #         fine = t[~t.isnan()]
    #         detail = ""
    #         if fine.numel() > 0:
    #             detail = (
    #                 f", min={fine.min().item():.4f}"
    #                 f", max={fine.max().item():.4f}"
    #             )
    #         raise RuntimeError(
    #             f"NaN in {label}: {nan_count}/{total} NaN values"
    #             f" (shape={tuple(t.shape)}{detail})"
    #         )

    # _check(projections, "projections (input)")
    # if text_embeddings is not None:
    #     _check(text_embeddings, "text_embeddings (input)")
    # if tag_vectors is not None:
    #     _check(tag_vectors, "tag_vectors (input)")

    sim, pos_mask = _positive_mask(
        projections,
        family_labels,
        font_ids=font_ids,
        tag_vectors=tag_vectors,
        text_embeddings=text_embeddings,
        family_positive_weight=family_positive_weight,
        tag_weight=tag_weight,
        tag_threshold=tag_threshold,
    )
    N = projections.shape[0]

    sim = sim / temperature
    # Exclude self from the softmax denominator.
    sim[torch.arange(N), torch.arange(N)] = float("-inf")

    # ---- SupCon-format loss: weighted mean over positives --------------
    log_prob = sim.log_softmax(dim=1)  # (2B, M)
    # _check(log_prob, "log_prob after log_softmax")
    # Zero out non-positive positions so 0 * -inf is safe, then
    # weighted-sum by pos_mask values and normalise by total weight.
    log_prob = log_prob.masked_fill(pos_mask == 0, 0.0)
    n_pos = pos_mask.sum(dim=1).clamp(min=1)  # (2B,)
    loss_per_row = -(pos_mask * log_prob).sum(dim=1) / n_pos  # (2B,)
    return loss_per_row.mean()


def circle_loss(
    projections: torch.Tensor,
    family_labels: list[str],
    *,
    font_ids: list[str] | None = None,
    text_embeddings: torch.Tensor | None = None,
    tag_vectors: torch.Tensor | None = None,
    family_positive_weight: float = 0.3,
    tag_threshold: float = 0.1,
    tag_weight: float = 1.0,
    margin: float = 0.25,
    gamma: float = 128.0,
) -> torch.Tensor:
    """Circle loss over the same positive set as ``multipos_contrastive_loss``.

    Unlike InfoNCE, the negative term is bounded by a hard margin: a negative is
    pushed only until its cosine falls ``margin`` below the positive decision
    boundary, then contributes no gradient.  This stops the late-training
    uniformity runaway (the loss falling purely by pushing every pair apart)
    that erodes fine-grained family/axis/attribution structure.  Positive pairs
    still pull toward cosine 1, so alignment is preserved.

    Args:
        margin: single margin ``m``; positive boundary ``1 - m``, negative
            boundary ``m`` (both in cosine space, [-1, 1]).
        gamma: scale factor (circle-loss analogue of the softmax temperature).
    """
    B = projections.shape[0] // 2
    if B < 2:
        return torch.tensor(0.0, device=projections.device)

    sim, pos_mask = _positive_mask(
        projections,
        family_labels,
        font_ids=font_ids,
        tag_vectors=tag_vectors,
        text_embeddings=text_embeddings,
        family_positive_weight=family_positive_weight,
        tag_weight=tag_weight,
        tag_threshold=tag_threshold,
    )
    N = sim.shape[0]

    positives = pos_mask > 0
    negatives = ~positives
    negatives[torch.arange(N), torch.arange(N)] = False

    delta_p = 1.0 - margin
    delta_n = margin

    # Self-paced weights (detached: they modulate, not differentiate).
    alpha_p = torch.clamp_min(1.0 + margin - sim, min=0.0).detach()  # (N, M)
    alpha_n = torch.clamp_min(sim + margin, min=0.0).detach()        # (N, M)

    # pos_mask doubles as the positive weight (hard positive = 1, soft < 1).
    logit_p = -gamma * pos_mask * alpha_p * (sim - delta_p)
    logit_n = gamma * alpha_n * (sim - delta_n)

    logit_p = logit_p.masked_fill(~positives, float("-inf"))
    logit_n = logit_n.masked_fill(~negatives, float("-inf"))

    lp = torch.logsumexp(logit_p, dim=1)  # (N,)
    ln = torch.logsumexp(logit_n, dim=1)  # (N,)
    return F.softplus(lp + ln).mean()


def contrastive_loss(
    projections: torch.Tensor,
    temperature: float = 0.07,
    family_labels: list[str] | None = None,
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
                is_positive = (i < B and j == i + B) or (i >= B and j == i - B)
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


def view_invariance_loss(embeddings: torch.Tensor) -> torch.Tensor:
    """Pull the two views of each font's *summary* embedding together.

    ``embeddings`` is ``(2B, D)`` with view 1 stacked on view 2.  The summary is
    the deliverable, so the content-invariance that the disjoint-glyph-subset
    views are meant to teach is enforced directly on it (rather than only through
    the contrastive objective, which can be satisfied by spreading negatives
    apart while the two views of the same font drift content-sensitive — visible
    as falling sampling stability even as the contrastive loss drops).

    Returns mean ``1 - cos(view1, view2)`` in ``[0, 2]``.
    """
    B = embeddings.shape[0] // 2
    if B < 1:
        return torch.tensor(0.0, device=embeddings.device)
    v1 = F.normalize(embeddings[:B].float(), p=2, dim=-1, eps=1e-8)
    v2 = F.normalize(embeddings[B:].float(), p=2, dim=-1, eps=1e-8)
    cos = (v1 * v2).sum(dim=-1)  # (B,)
    return (1.0 - cos).mean()


def _off_diag_cov(z: torch.Tensor) -> torch.Tensor:
    """Mean squared off-diagonal of the correlation matrix (anti-collapse).

    Rows are L2-normalized first so the term is scale-invariant: it reads the
    *directions* of the summary, not their arbitrary magnitude.
    """
    z = F.normalize(z, p=2, dim=-1, eps=1e-8)
    z = z - z.mean(dim=0, keepdim=True)
    n, d = z.shape
    cov = (z.T @ z) / max(n - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    return (off * off).sum() / d


def vicreg_loss(embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
    """Invariance + covariance between the two views of the *summary* embedding.

    ``embeddings`` is ``(2B, D)`` with view 1 in the first B rows and view 2 in
    the next B rows.

    * ``invariance`` — cosine distance between the two views (content-invariance).
    * ``covariance`` — off-diagonal decorrelation of the correlation matrix
      (discourages the summary from collapsing to a low-rank subspace).

    The variance term from the original VICReg is deliberately omitted: the
    circle loss already spreads instances, and a ``relu(1 - std)`` term on the
    raw summary just chases its arbitrary magnitude (which is normalized away
    downstream) and destabilizes training.
    """
    B = embeddings.shape[0] // 2
    if B < 2:
        return {
            "invariance": torch.tensor(0.0, device=embeddings.device),
            "covariance": torch.tensor(0.0, device=embeddings.device),
        }
    z1 = embeddings[:B].float()
    z2 = embeddings[B:].float()

    n1 = F.normalize(z1, p=2, dim=-1, eps=1e-8)
    n2 = F.normalize(z2, p=2, dim=-1, eps=1e-8)
    invariance = (1.0 - (n1 * n2).sum(dim=-1)).mean()

    covariance = _off_diag_cov(z1) + _off_diag_cov(z2)
    return {
        "invariance": invariance,
        "covariance": covariance,
    }


def content_averaged_shape(
    images: torch.Tensor,
    glyph_mask: torch.Tensor | None,
    grid: int,
) -> torch.Tensor:
    """Mean over *visible* glyphs of the raw image pooled to ``grid``×``grid``.

    This is the fixed reconstruction target: the content-averaged raw glyph
    shape, which carries the fine visual style (corner rounding, terminals)
    that the summary must retain.  It is derived from the raw images (not the
    encoder), so it cannot be distorted to make reconstruction trivial.
    """
    b, g, c, h, w = images.shape
    flat = images.flatten(0, 1)  # (B*G, C, H, W)
    shape = F.adaptive_avg_pool2d(flat, (grid, grid))  # (B*G, C, grid, grid)
    shape = shape.reshape(b, g, c, grid, grid)
    if glyph_mask is not None:
        mask = glyph_mask.to(shape.dtype).reshape(b, g, 1, 1, 1)
        return (shape * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return shape.mean(dim=1)


def reconstruction_loss(
    reconstruction: torch.Tensor, shape_target: torch.Tensor
) -> torch.Tensor:
    """Reconstruction of the raw content-averaged glyph shape.

    Forces the summary to retain the fine visual style it would otherwise
    discard in favour of coarse instance/category separation: if the summary can
    reproduce the raw glyph shape, two fonts with similar shape must land near
    each other.  The target is detached, and both sides are mean-centered before
    the cosine so the loss reads *shape* (ink distribution) rather than the
    dominant white-background DC component.

    Returns mean ``1 - cos(recon, shape)`` over the batch, in [0, 2].
    """
    recon = reconstruction.flatten(1)
    target = shape_target.detach().flatten(1)
    recon = recon - recon.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    recon = F.normalize(recon, p=2, dim=-1, eps=1e-8)
    target = F.normalize(target, p=2, dim=-1, eps=1e-8)
    cos = (recon * target).sum(dim=-1)  # (B,)
    return (1.0 - cos).mean()


def compute_losses(
    projections: torch.Tensor,
    predicted_tags: dict[str, torch.Tensor] | None,
    target_tags: dict[str, torch.Tensor],
    *,
    weights: FontStyleEmbeddingLossWeights,
    temperature: float = 0.07,
    contrastive_loss_type: str = "multipos",
    circle_margin: float = 0.25,
    circle_gamma: float = 128.0,
    tag_masks: dict[str, torch.Tensor] | None = None,
    family_labels: list[str] | None = None,
    font_ids: list[str] | None = None,
    category_logits: torch.Tensor | None = None,
    category_targets: torch.Tensor | None = None,
    text_embeddings: torch.Tensor | None = None,
    tag_vectors: torch.Tensor | None = None,
    embeddings: torch.Tensor | None = None,
    shape_target: torch.Tensor | None = None,
    reconstruction: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute combined loss and per-term breakdown.

    Returns:
        ``(total_loss, loss_info)`` for TensorBoard logging.
    """
    device = projections.device

    # Multi-positive contrastive loss.  Without text/tags this still uses the
    # same-font other-view hard positive plus (when font_ids are present) the
    # same-family soft positive.
    fam = family_labels if family_labels is not None else []
    if contrastive_loss_type == "circle":
        contr = circle_loss(
            projections,
            fam,
            font_ids=font_ids,
            text_embeddings=text_embeddings,
            tag_vectors=tag_vectors,
            family_positive_weight=weights.family_positive_weight,
            tag_weight=weights.tag_positive_weight,
            margin=circle_margin,
            gamma=circle_gamma,
        )
    else:
        contr = multipos_contrastive_loss(
            projections,
            fam,
            font_ids=font_ids,
            text_embeddings=text_embeddings,
            tag_vectors=tag_vectors,
            temperature=temperature,
            family_positive_weight=weights.family_positive_weight,
            tag_weight=weights.tag_positive_weight,
        )
    weight = weights.multipos_contrastive

    # Tag prediction loss.
    tag_loss = torch.tensor(0.0, device=device)
    if predicted_tags is not None and target_tags:
        tag_loss = tag_prediction_loss(
            predicted_tags,
            target_tags,
            masks=tag_masks or {},
        )

    # Category classification loss.
    cat_loss = torch.tensor(0.0, device=device)
    cat_accuracy = torch.tensor(0.0, device=device)
    if category_logits is not None and category_targets is not None:
        cat_loss, cat_accuracy = category_loss(category_logits, category_targets)

    # View invariance: pull the two views' summary embeddings together so the
    # content-invariance the disjoint-glyph views are meant to teach is enforced
    # on the deliverable vector, not just the contrastive projection.
    inv = torch.tensor(0.0, device=device)
    if embeddings is not None and weights.view_invariance > 0:
        inv = view_invariance_loss(embeddings)

    # VICReg (variance + invariance + covariance): prevents the summary from
    # collapsing to a low-rank subspace and enforces content-invariance between
    # the two views.  Subsumes the view-invariance term.
    vic = torch.tensor(0.0, device=device)
    vic_components: dict[str, torch.Tensor] = {}
    if embeddings is not None and weights.vicreg_weight > 0:
        vic_components = vicreg_loss(embeddings)
        vic = vic_components["invariance"] + vic_components["covariance"]

    # Visual reconstruction: force the summary to retain fine style.
    recon = torch.tensor(0.0, device=device)
    if (
        shape_target is not None
        and reconstruction is not None
        and weights.reconstruction_weight > 0
    ):
        recon = reconstruction_loss(reconstruction, shape_target)

    total = (
        weight * contr
        + weights.tag_prediction * tag_loss
        + cat_loss
        + weights.view_invariance * inv
        + weights.reconstruction_weight * recon
        + weights.vicreg_weight * vic
    )

    loss_info: dict[str, torch.Tensor] = {
        "contrastive": contr.detach(),
        "contrastive_weighted": (weight * contr).detach(),
        "loss": total.detach(),
    }
    if embeddings is not None and weights.view_invariance > 0:
        loss_info["view_invariance"] = inv.detach()
        loss_info["view_invariance_weighted"] = (
            weights.view_invariance * inv
        ).detach()
    if embeddings is not None and weights.vicreg_weight > 0:
        loss_info["vicreg"] = vic.detach()
        loss_info["vicreg_weighted"] = (weights.vicreg_weight * vic).detach()
        loss_info["vicreg_invariance"] = vic_components["invariance"].detach()
        loss_info["vicreg_covariance"] = vic_components["covariance"].detach()
    if (
        shape_target is not None
        and reconstruction is not None
        and weights.reconstruction_weight > 0
    ):
        loss_info["reconstruction"] = recon.detach()
        loss_info["reconstruction_weighted"] = (
            weights.reconstruction_weight * recon
        ).detach()
    if predicted_tags is not None and target_tags:
        loss_info["tag_prediction"] = tag_loss.detach()
        loss_info["tag_prediction_weighted"] = (
            weights.tag_prediction * tag_loss
        ).detach()
        # Per-tag accuracy or MSE for diagnostics.
        for name, pred in predicted_tags.items():
            if name not in target_tags:
                continue
            m = (tag_masks or {}).get(
                name, torch.ones(pred.shape[0], dtype=torch.bool, device=device)
            )
            if not m.any():
                continue
            safe_name = name.replace("/", "_").strip("_")
            if pred.ndim == 2:
                # Classification: per-tag accuracy.
                correct = (
                    (pred[m].argmax(dim=-1) == target_tags[name][m].long())
                    .float()
                    .mean()
                )
                loss_info[f"tag_acc_{safe_name}"] = correct.detach()
            else:
                # Regression: per-tag MSE.
                se = ((pred[m] - target_tags[name][m]) ** 2).mean()
                loss_info[f"tag_mse_{safe_name}"] = se.detach()
    if category_logits is not None and category_targets is not None:
        loss_info["category_loss"] = cat_loss.detach()
        loss_info["category_accuracy"] = cat_accuracy.detach()

    return total, loss_info
