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
        style_conformance_mode: bool = False,
        clean_font_only: bool = False,
        clean_font_display_score_threshold: float = 45.0,
        outline_noise_std: float = 0.08,
        outline_noise_edge_threshold: float = 0.12,
        low_res_noise_std: float = 0.01,
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
        if clean_font_display_score_threshold < 0.0:
            raise ValueError(
                "clean_font_display_score_threshold must be non-negative "
                f"(got {clean_font_display_score_threshold})"
            )
        if outline_noise_std < 0.0:
            raise ValueError(
                f"outline_noise_std must be non-negative (got {outline_noise_std})"
            )
        if not (0.0 <= outline_noise_edge_threshold <= 1.0):
            raise ValueError(
                "outline_noise_edge_threshold must be in [0, 1] "
                f"(got {outline_noise_edge_threshold})"
            )
        if low_res_noise_std < 0.0:
            raise ValueError(
                f"low_res_noise_std must be non-negative (got {low_res_noise_std})"
            )

        self.low_res_size = low_res_size
        self.high_res_size = high_res_size
        self.upscale_factor = high_res_size / low_res_size
        self.style_conformance_mode = style_conformance_mode
        self.clean_font_only = clean_font_only
        self.clean_font_display_score_threshold = clean_font_display_score_threshold
        self.outline_noise_std = outline_noise_std
        self.outline_noise_edge_threshold = outline_noise_edge_threshold
        self.low_res_noise_std = low_res_noise_std

        super().__init__(
            repo_url=repo_url,
            batch_size=batch_size,
            canary_size=canary_size,
            having=having,
            image_size=high_res_size,
        )

        if self.clean_font_only:
            self.train_fonts = [
                font for font in self.train_fonts if self._is_clean_font(font)
            ]
            self.test_fonts = [font for font in self.test_fonts if self._is_clean_font(font)]

            if not self.train_fonts:
                raise ValueError(
                    "No training fonts remain after clean-font filtering; "
                    "increase --clean-font-display-score-threshold or disable "
                    "--clean-font-only"
                )
            if not self.test_fonts:
                raise ValueError(
                    "No validation fonts remain after clean-font filtering; "
                    "increase --clean-font-display-score-threshold or disable "
                    "--clean-font-only"
                )

    def train_set(self):
        return AllGidsDataset(self.train_fonts)

    def test_set(self):
        return AllGidsDataset(self.test_fonts)

    def _is_clean_font(self, font: object) -> bool:
        display_score_fn = getattr(font, "display_score", None)
        if display_score_fn is None or not callable(display_score_fn):
            return True

        score = float(display_score_fn())
        return score <= self.clean_font_display_score_threshold

    def _edge_magnitude(self, images: torch.Tensor) -> torch.Tensor:
        grayscale = images.mean(dim=1, keepdim=True)
        gx = grayscale[:, :, :, 1:] - grayscale[:, :, :, :-1]
        gy = grayscale[:, :, 1:, :] - grayscale[:, :, :-1, :]
        gx = F.pad(gx, (0, 1, 0, 0), mode="replicate")
        gy = F.pad(gy, (0, 0, 0, 1), mode="replicate")
        magnitude = torch.sqrt(gx * gx + gy * gy + 1e-8)
        return magnitude

    def _corrupt_high_res_for_conformance(self, high_res: torch.Tensor) -> torch.Tensor:
        edge = self._edge_magnitude(high_res)
        edge_norm = edge / (edge.amax(dim=(2, 3), keepdim=True) + 1e-6)
        edge_mask = (edge_norm >= self.outline_noise_edge_threshold).to(high_res.dtype)

        # Mild local blur around edges to mimic patch-boundary wobble and stroke roughness.
        blurred = F.avg_pool2d(high_res, kernel_size=3, stride=1, padding=1)
        blur_mix = torch.rand(
            (high_res.shape[0], 1, 1, 1),
            dtype=high_res.dtype,
            device=high_res.device,
        )
        mixed = high_res * (1.0 - edge_mask * blur_mix) + blurred * (edge_mask * blur_mix)

        noise = torch.randn_like(high_res) * self.outline_noise_std
        corrupted = mixed + noise * edge_mask
        return corrupted.clamp(0.0, 1.0)

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

        low_res_source = high_res
        if self.style_conformance_mode:
            low_res_source = self._corrupt_high_res_for_conformance(high_res)

        # Downsample from the source raster so pairs stay aligned.
        low_res = F.interpolate(
            low_res_source,
            size=(self.low_res_size, self.low_res_size),
            mode="area",
        )

        if self.style_conformance_mode and self.low_res_noise_std > 0.0:
            low_res = (low_res + torch.randn_like(low_res) * self.low_res_noise_std).clamp(
                0.0, 1.0
            )

        return {
            "gid": gids,
            "low_res": low_res,
            "high_res": high_res,
        }


__all__ = ["UpscalerDatasetMaker"]
