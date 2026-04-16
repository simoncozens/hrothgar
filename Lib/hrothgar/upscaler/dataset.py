"""Dataset maker for glyph super-resolution.

This prototype emits low-resolution and high-resolution raster pairs for the
Latin core character set, using the shared train/test split logic from
``hrothgar.dataset.DatasetMaker``.
"""

from __future__ import annotations

from typing import Optional, Sequence, Set

import torch
import torch.nn.functional as F

from hrothgar.dataset import DatasetMaker, LATIN_CORE


class UpscalerDatasetMaker(DatasetMaker):
    """Create (low_res, high_res) glyph pairs for super-resolution training."""

    def __init__(
        self,
        repo_url: str,
        batch_size: int,
        low_res_size: int = 128,
        high_res_size: int = 512,
        target_codepoints: Optional[Sequence[int]] = None,
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

        codepoints = (
            set(target_codepoints) if target_codepoints is not None else set(LATIN_CORE)
        )

        super().__init__(
            repo_url=repo_url,
            batch_size=batch_size,
            target_codepoints=codepoints,
            canary_size=canary_size,
            having=having,
            image_size=high_res_size,
        )

    def collate_fn(self, batch):
        chars = torch.tensor([item["char"] for item in batch], dtype=torch.long)
        high_res = torch.stack(
            [
                torch.tensor(
                    item["font"].render(item["char"], size=self.high_res_size),
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

        descriptions = [item["font"].description_with_tags() for item in batch]
        font_names = [item["font"].family for item in batch]

        return {
            "char": chars,
            "low_res": low_res,
            "high_res": high_res,
            "description": descriptions,
            "font_family": font_names,
        }


__all__ = ["UpscalerDatasetMaker"]
