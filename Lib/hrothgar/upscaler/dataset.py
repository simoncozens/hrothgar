"""Dataset maker for glyph super-resolution.

This prototype emits low-resolution and high-resolution raster pairs for the
Latin core gidacter set, using the shared train/test split logic from
``hrothgar.dataset.DatasetMaker``.
"""

from __future__ import annotations

from typing import Optional, Set

import torch
import torch.nn.functional as F

from hrothgar.dataset import AllGidsDataset, DatasetMaker


class UpscalerDatasetMaker(DatasetMaker):
    """Create (low_res, high_res) glyph pairs for super-resolution training."""

    def __init__(
        self,
        repo_url: str,
        batch_size: int,
        low_res_size: int = 128,
        high_res_size: int = 512,
        canary_size: Optional[int] = None,
        having: Optional[Set[int]] = None,
    ) -> None:
        if low_res_size <= 0 or high_res_size <= 0:
            raise ValueError(
                "low_res_size and high_res_size must be positive "
                f"(got {low_res_size} and {high_res_size})"
            )
        if high_res_size <= low_res_size:
            raise ValueError(
                "high_res_size must be greater than low_res_size "
                f"(got {high_res_size} <= {low_res_size})"
            )

        self.low_res_size = low_res_size
        self.high_res_size = high_res_size
        self.upscale_factor = high_res_size / low_res_size

        super().__init__(
            repo_url=repo_url,
            batch_size=batch_size,
            canary_size=canary_size,
            having=having,
            image_size=high_res_size,
        )

    def train_set(self):
        return AllGidsDataset(self.train_fonts)

    def test_set(self):
        return AllGidsDataset(self.test_fonts)

    def collate_fn(self, batch):
        gids = torch.tensor([item["gid"] for item in batch], dtype=torch.long)
        high_res = torch.stack(
            [
                torch.tensor(
                    item["font"].render_gid(item["gid"], size=self.high_res_size),
                    dtype=torch.float32,
                )
                for item in batch
            ]
        )

        # Downsample from the high-resolution raster so pairs stay perfectly aligned.
        low_res = F.interpolate(
            high_res,
            size=(self.low_res_size, self.low_res_size),
            mode="area",
        )

        return {
            "gid": gids,
            "low_res": low_res,
            "high_res": high_res,
        }


__all__ = ["UpscalerDatasetMaker"]
