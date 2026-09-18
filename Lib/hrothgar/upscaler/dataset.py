"""Dataset maker for glyph super-resolution.

Emits ``(low_res, high_res)`` raster pairs for the Latin core character set,
both rendered **natively** with the same crop-to-ink normalize-to-square
convention the diffusion model uses.  Training on native 128/512 pairs matches
the upscaler's inference input (the diffusion model's 128px output, which is a
native 128 render) rather than an area-downsampled 512 — the two have different
anti-aliasing.
"""

from __future__ import annotations

import torch

from hrothgar.dataset import AllGidsDataset, DatasetMaker
from hrothgar.render_utils import render_gid_with_geometry


class UpscalerDatasetMaker(DatasetMaker):
    """Create (low_res, high_res) glyph pairs for super-resolution training."""

    @staticmethod
    def _render_gid_grayscale(font, gid: int, size: int) -> torch.Tensor:
        """Render a glyph by GID, cropped to ink and normalized to a square.

        Uses the same raw-bitmap + normalize-to-square pipeline as the factorized
        diffusion model (descenders and negative left sidebearings preserved), so
        the diffusion model's output can be fed directly to the upscaler.  Returns
        a ``(size, size)`` grayscale tensor in ``[0, 1]`` (0 = ink, 1 = white).
        """
        image, _ = render_gid_with_geometry(font, gid, size)
        return image  # (size, size)

    def __init__(
        self,
        repo_url: str,
        batch_size: int,
        low_res_size: int = 128,
        high_res_size: int = 512,
        canary_size: int | None = None,
        having: set[int] | None = None,
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

        super().__init__(
            repo_url=repo_url,
            batch_size=batch_size,
            canary_size=canary_size,
            having=having,
            image_size=high_res_size,
        )

    def train_set(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        return AllGidsDataset(self.train_fonts)

    def test_set(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        return AllGidsDataset(self.test_fonts)

    def collate_fn(self, batch):
        gids = torch.tensor([item["gid"] for item in batch], dtype=torch.long)
        low_res = torch.stack(
            [
                self._render_gid_grayscale(item["font"], item["gid"], self.low_res_size)
                for item in batch
            ]
        ).unsqueeze(1)  # (B, 1, low, low)
        high_res = torch.stack(
            [
                self._render_gid_grayscale(
                    item["font"], item["gid"], self.high_res_size
                )
                for item in batch
            ]
        ).unsqueeze(1)  # (B, 1, high, high)

        return {"gid": gids, "low_res": low_res, "high_res": high_res}
