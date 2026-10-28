"""Permutation-invariant set encoder over contour tokens (Deep Sets / PointNet).

The input is a *set* of per-on-curve-point tokens (variable length), so the
encoder is permutation-invariant: a per-token MLP followed by global max + mean
pooling.  Tokens are padded to a common length and pooled through a mask, so a
whole batch of glyphs is processed in one pass.  This is the minimal thing that
can consume the tokens and is very hard to overfit at this scale; a Set
Transformer (self-attention over tokens) is the natural upgrade if the signal
turns out to need pairwise interactions.
"""

from __future__ import annotations

import torch
from torch import nn


class SetEncoder(nn.Module):
    def __init__(self, in_dim: int = 49, hidden: int = 128, num_classes: int = 4):
        super().__init__()
        self.token_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``tokens`` is ``(B, L, in_dim)``, ``mask`` is ``(B, L)`` bool (True = real).

        Returns logits ``(B, num_classes)``.
        """
        z = self.token_mlp(tokens)  # (B, L, H)
        z_max = z.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1).to(z.dtype)
        z_mean = (z * mask.unsqueeze(-1).to(z.dtype)).sum(dim=1) / counts
        pooled = torch.cat([z_max, z_mean], dim=-1)  # (B, 2H)
        return self.head(pooled)  # (B, C)
