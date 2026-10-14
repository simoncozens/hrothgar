"""Loss for the instance-membership model."""

from __future__ import annotations

import torch


def glyph_contrastive_loss(
    embeddings: torch.Tensor,
    instance_ids: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Multi-positive InfoNCE at glyph level.

    ``embeddings`` are L2-normalized ``(N, D)``; ``instance_ids`` are ``(N,)``.
    Positives are all glyphs sharing an instance (so different letters of the
    same instance attract, teaching content-invariance); negatives are glyphs
    of other instances in the batch.
    """
    n = embeddings.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=embeddings.device)

    sim = embeddings @ embeddings.T / temperature
    sim.fill_diagonal_(float("-inf"))

    same = (instance_ids[:, None] == instance_ids[None, :]) & ~torch.eye(
        n, dtype=torch.bool, device=embeddings.device
    )
    pos_count = same.sum(dim=1)
    rows = pos_count > 0
    if not rows.any():
        return torch.tensor(0.0, device=embeddings.device)

    log_denominator = torch.logsumexp(sim, dim=1)  # (N,)
    # `same` is False on the (now -inf) diagonal; use `where` so 0 never
    # multiplies -inf into NaN.
    positive_sim = torch.where(same, sim, torch.zeros_like(sim))
    mean_positive = positive_sim.sum(dim=1) / pos_count.clamp(min=1)
    return -(mean_positive[rows] - log_denominator[rows]).mean()
