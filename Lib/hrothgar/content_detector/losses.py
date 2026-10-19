"""Loss for the content detector.

The detector is one-vs-rest: each anchor predicts independently whether the
glyph is *that* codepoint.  Binary cross-entropy over every ``(glyph, anchor)``
pair is therefore the natural loss, and — crucially — it admits the "all
anchors low" answer that a softmax cannot express.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ovr_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | float | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary cross-entropy over the full ``(N, K)`` glyph × anchor grid.

    ``logits`` are ``(N, K)`` (one row per glyph, one column per anchor) and
    ``targets`` are the matching ``(N, K)`` multi-hot matrix: ``targets[n, k]``
    is 1 iff glyph ``n`` *is* anchor ``k``.  A glyph of a codepoint with no slot
    (a confusable, a diacritic variant, a glyph from another script) is
    negative for every anchor, so its whole row is zero.

    ``pos_weight`` rescales the positive term per anchor (``neg/pos`` to undo the
    training set's prevalence).  Left at 1.0 the loss teaches the *empirical*
    prior, which is what we want before calibration; set it when the class
    imbalance is extreme enough to stall learning.

    ``mask`` is an optional ``(N,)`` float (1 = keep) that excludes whole rows
    from the loss — used for "broken" glyphs, which must not contribute a binary
    target (they are handled by the quality margin instead).
    """
    loss = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    if mask is not None:
        w = mask.to(loss.dtype).unsqueeze(1)
        denom = w.sum()
        if float(denom) == 0.0:
            return logits.sum() * 0.0
        return (loss * w).sum() / denom
    return loss.mean()


def margin_hinges(
    cosine: torch.Tensor,
    targets: torch.Tensor,
    *,
    ranking_margin: float,
    positive_margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine-space margin hinges for the glyphs that have a claimed anchor.

    ``ranking`` requires a glyph's own anchor to beat every *other* anchor by
    ``ranking_margin``; ``positive`` requires it to clear ``positive_margin`` in
    absolute terms.  Both are computed in raw cosine, *before* the logit scale and
    per-anchor bias — that is the point.  The BCE term can be reduced by shifting a
    bias down, so it can leave the embedding space collapsed onto a shared "it's
    ink" component; these hinges cannot, so the only way to reduce them is to move
    the embedding or the anchor.  ``ranking`` is also what stops two similar anchors
    (``a`` and ``à``) from both firing on a good glyph.

    Rows with no claimed anchor (confusables, diacritic variants, off-sample) are
    left to the BCE term.
    """
    claimed = targets.sum(dim=1) > 0
    if not bool(claimed.any()):
        zero = cosine.sum() * 0.0
        return zero, zero

    own = targets.argmax(dim=1)
    own_cosine = cosine.gather(1, own.unsqueeze(1)).squeeze(1)

    other = cosine.clone()
    other[torch.arange(cosine.shape[0], device=cosine.device), own] = float("-inf")
    best_other = other.max(dim=1).values

    ranking = (ranking_margin - (own_cosine - best_other)).clamp(min=0.0)[claimed].mean()
    positive = (positive_margin - own_cosine).clamp(min=0.0)[claimed].mean()
    return ranking, positive


def quality_margin_hinge(
    cosine: torch.Tensor,
    targets: torch.Tensor,
    broken_mask: torch.Tensor,
    *,
    quality_margin: float,
) -> torch.Tensor:
    """Within-anchor quality margin: correct glyphs beat broken glyphs.

    ``targets`` is the ``(N, K)`` multi-hot grid, where both *correct* and
    *broken* glyphs claim their own anchor (so the ranking/positive hinges still
    apply to them); ``broken_mask`` is ``(N,)`` marking the broken (bad-skeleton)
    glyphs.  For every anchor that has both kinds, hinge on
    ``mean(correct cosine) - mean(broken cosine) >= quality_margin``.  This is
    what makes a "broken but recognisable" glyph score *between* a correct glyph
    and an unrelated codepoint, in raw cosine space where a bias cannot absorb
    it.
    """
    if quality_margin <= 0.0 or not bool(broken_mask.any()):
        return cosine.sum() * 0.0

    broken = broken_mask.to(targets.dtype).unsqueeze(1)  # (N, 1)
    correct = targets * (1.0 - broken)  # (N, K)
    broken_claimed = targets * broken  # (N, K)

    n_correct = correct.sum(0)
    n_broken = broken_claimed.sum(0)
    has_both = (n_correct > 0) & (n_broken > 0)
    if not bool(has_both.any()):
        return cosine.sum() * 0.0

    correct_mean = (cosine * correct).sum(0) / n_correct.clamp(min=1)
    broken_mean = (cosine * broken_claimed).sum(0) / n_broken.clamp(min=1)
    gap = correct_mean - broken_mean
    return (quality_margin - gap).clamp(min=0.0)[has_both].mean()


def height_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE between predicted and target relative heights, both ``(B,)`` in [0, 1].

    This is the auxiliary objective for the height head; it is folded into the
    training loss so the shared encoder must keep height in its features.
    ``mask`` (``(B,)``, 1 where the target is meaningful) excludes rows with no
    height to learn — images loaded off disk, and positives that rendered blank —
    whose zero target would otherwise pull the head toward zero.  An empty or
    fully-masked batch yields a zero that still carries the graph, so backward
    never sees ``None``.
    """
    if mask is not None:
        weights = mask.to(predicted.dtype)
        total = weights.sum()
        if float(total) == 0.0:
            return predicted.sum() * 0.0
        return (((predicted - target) ** 2) * weights).sum() / total
    if predicted.numel() == 0:
        return predicted.sum() * 0.0
    return F.mse_loss(predicted, target)
