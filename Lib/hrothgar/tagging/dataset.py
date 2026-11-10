"""Torch dataset over a precomputed :class:`FeatureTable`.

The train/validation split is by **family**, so two instances of the same
variable family (e.g. ``wght@400`` and ``wght@900``) never straddle the split —
their gap features are highly correlated and would leak the answer.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from hrothgar.tagging.features import FeatureTable


def family_train_val_split(
    table: FeatureTable, val_fraction: float, seed: int = 0
) -> tuple[list[int], list[int]]:
    """Return ``(train_idx, val_idx)`` with disjoint families."""
    rng = np.random.default_rng(seed)
    fam_to_idx: dict[str, list[int]] = {}
    for i, fam in enumerate(table.families):
        fam_to_idx.setdefault(fam, []).append(i)

    families = list(fam_to_idx)
    rng.shuffle(families)
    n_val = max(1, int(round(val_fraction * len(families))))
    val_families = set(families[:n_val])

    train_idx = [i for f in families if f not in val_families for i in fam_to_idx[f]]
    val_idx = [i for f in val_families for i in fam_to_idx[f]]
    return train_idx, val_idx


class TagDataset(Dataset):
    """Yields ``(features, target, mask)`` for the selected cells."""

    def __init__(self, table: FeatureTable, indices: list[int]):
        self.table = table
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        row = self.indices[i]
        feat = torch.from_numpy(self.table.features[row]).float()
        vals = self.table.values[row].astype(np.float32)
        mask = ~np.isnan(vals)
        target = np.where(mask, vals / 100.0, 0.0)  # normalise centiles to [0, 1]
        return feat, torch.from_numpy(target), torch.from_numpy(mask)
