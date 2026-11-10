"""Masked regression losses for tag prediction.

A tag is *unassessed* for a cell when it is absent from ``families.csv``, so the
loss is computed only over ``(cell, tag)`` pairs where ``mask`` is True.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_loss(
    pred: torch.Tensor,  # (B, T) predictions in [0, 1]
    target: torch.Tensor,  # (B, T) targets in [0, 1], NaN where unassessed
    mask: torch.Tensor,  # (B, T) bool
    kind: str = "huber",
) -> torch.Tensor:
    """Mean masked MSE or Huber loss over assessed ``(cell, tag)`` pairs."""
    if kind == "mse":
        loss = F.mse_loss(pred[mask], target[mask])
    elif kind == "huber":
        loss = F.smooth_l1_loss(pred[mask], target[mask])
    else:
        raise ValueError(f"unknown loss kind {kind!r}")
    return loss


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Map [0, 1] predictions back to 0-100 centiles."""
    return x * 100.0
