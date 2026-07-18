"""Dataset maker for glyph super-resolution.

This prototype emits low-resolution and high-resolution raster pairs for the
Latin core character set, using the shared train/test split logic from
``hrothgar.dataset.DatasetMaker``.

Each batch also includes K style-reference glyphs per sample — other glyphs
from the same font rendered at high resolution — for style conditioning.
"""

from __future__ import annotations

from typing import Optional, Set

import numpy as np
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
        style_reference_count: int = 4,
        terminal_blur_sigma: float = 2.75,
        stem_blur_sigma: float = 0.75,
        blur_mix_min: float = 0.2,
        blur_mix_max: float = 0.85,
        blur_sigma_jitter: float = 0.3,
        mix_spatial_noise: float = 0.15,
        harris_window: int = 5,
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
        if not (0.0 <= low_res_noise_std <= 1.0):
            raise ValueError(
                "low_res_noise_std must be in [0, 1] when interpreted as "
                f"replacement probability (got {low_res_noise_std})"
            )
        if terminal_blur_sigma < 0.0:
            raise ValueError(
                f"terminal_blur_sigma must be non-negative (got {terminal_blur_sigma})"
            )
        if stem_blur_sigma < 0.0:
            raise ValueError(
                f"stem_blur_sigma must be non-negative (got {stem_blur_sigma})"
            )
        if not (0.0 <= blur_mix_min <= blur_mix_max <= 1.0):
            raise ValueError(
                "blur_mix_min and blur_mix_max must be in [0, 1] "
                f"with blur_mix_min <= blur_mix_max (got {blur_mix_min}, {blur_mix_max})"
            )
        if not (0.0 <= blur_sigma_jitter <= 1.0):
            raise ValueError(
                "blur_sigma_jitter must be in [0, 1] "
                f"(got {blur_sigma_jitter})"
            )
        if not (0.0 <= mix_spatial_noise <= 1.0):
            raise ValueError(
                "mix_spatial_noise must be in [0, 1] "
                f"(got {mix_spatial_noise})"
            )
        if harris_window < 3 or harris_window % 2 == 0:
            raise ValueError(
                f"harris_window must be an odd integer >= 3 (got {harris_window})"
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
        self.style_reference_count = style_reference_count
        self.terminal_blur_sigma = terminal_blur_sigma
        self.stem_blur_sigma = stem_blur_sigma
        self.blur_mix_min = blur_mix_min
        self.blur_mix_max = blur_mix_max
        self.blur_sigma_jitter = blur_sigma_jitter
        self.mix_spatial_noise = mix_spatial_noise
        self.harris_window = harris_window

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
            self.test_fonts = [
                font for font in self.test_fonts if self._is_clean_font(font)
            ]

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

    def train_set(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        return AllGidsDataset(self.train_fonts)

    def test_set(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        return AllGidsDataset(self.test_fonts)

    def _is_clean_font(self, font: object) -> bool:
        display_score_fn = getattr(font, "display_score", None)
        if display_score_fn is None or not callable(display_score_fn):
            return True

        display_score = display_score_fn()
        if not isinstance(display_score, (int, float)):
            return True
        score = float(display_score)
        return score <= self.clean_font_display_score_threshold

    def _edge_magnitude(self, images: torch.Tensor) -> torch.Tensor:
        grayscale = images.mean(dim=1, keepdim=True)
        gx = grayscale[:, :, :, 1:] - grayscale[:, :, :, :-1]
        gy = grayscale[:, :, 1:, :] - grayscale[:, :, :-1, :]
        gx = F.pad(gx, (0, 1, 0, 0), mode="replicate")
        gy = F.pad(gy, (0, 0, 0, 1), mode="replicate")
        magnitude = torch.sqrt(gx * gx + gy * gy + 1e-8)
        return magnitude

    # ------------------------------------------------------------------
    # Curvature-weighted Gaussian blur (Tier 2)
    # ------------------------------------------------------------------

    @staticmethod
    def _gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply separable Gaussian blur across all channels.

        When *sigma* < 0.5 the blur is negligible, so the input is returned
        unchanged to avoid unnecessary compute.
        """
        if sigma < 0.5:
            return image

        kernel_size = int(2 * (3.0 * sigma) + 1) | 1  # ensure odd
        half = kernel_size // 2
        x = torch.arange(-half, half + 1, dtype=torch.float32, device=image.device)
        g1d = torch.exp(-0.5 * (x / sigma) ** 2)
        g1d = g1d / g1d.sum()
        g1d = g1d.view(1, 1, -1, 1)  # horizontal kernel: (1, 1, K, 1)

        B, C, H, W = image.shape
        flat = image.reshape(B * C, 1, H, W)

        # Horizontal pass
        flat = F.conv2d(flat, g1d, padding=(0, half))
        # Vertical pass
        flat = F.conv2d(flat, g1d.permute(0, 1, 3, 2), padding=(half, 0))

        return flat.reshape(B, C, H, W)

    def _corner_response(self, image: torch.Tensor) -> torch.Tensor:
        """Compute a normalised Harris corner-response map.

        Returns a ``(B, H, W)`` tensor where values near 1.0 indicate
        corners, stroke terminals and joins; values near 0.0 indicate
        straight edges or homogeneous regions.

        The Harris detector uses the structure tensor
        :math:`M = \\begin{bmatrix} I_x^2 & I_x I_y \\\\ I_x I_y & I_y^2 \\end{bmatrix}`
        smoothed with a box filter, then computes
        :math:`R = \\det(M) - k \\cdot \\operatorname{tr}(M)^2`.
        """
        gray = image.mean(dim=1, keepdim=True)  # (B, 1, H, W)

        # Central-difference gradients
        gx = gray[:, :, :, 2:] - gray[:, :, :, :-2]
        gx = F.pad(gx, (1, 1, 0, 0), mode="replicate")
        gy = gray[:, :, 2:, :] - gray[:, :, :-2, :]
        gy = F.pad(gy, (0, 0, 1, 1), mode="replicate")

        # Structure tensor elements
        gx2 = gx * gx
        gy2 = gy * gy
        gxy = gx * gy

        # Smooth with a box filter (fast and sufficient for corner detection)
        w = self.harris_window
        gx2 = F.avg_pool2d(gx2, w, stride=1, padding=w // 2)
        gy2 = F.avg_pool2d(gy2, w, stride=1, padding=w // 2)
        gxy = F.avg_pool2d(gxy, w, stride=1, padding=w // 2)

        k = 0.04
        det = gx2 * gy2 - gxy * gxy
        trace = gx2 + gy2
        R = det - k * trace * trace  # (B, 1, H, W)

        # Normalise to [0, 1] per image
        R_min = R.amin(dim=(2, 3), keepdim=True)
        R_max = R.amax(dim=(2, 3), keepdim=True)
        R = (R - R_min) / (R_max - R_min + 1e-8)

        return R.squeeze(1)  # (B, H, W)

    def _corrupt_high_res_for_conformance(self, high_res: torch.Tensor) -> torch.Tensor:
        """Apply curvature-weighted Gaussian blur to edge regions.

        1. Compute an edge mask from gradient magnitude.
        2. Compute a Harris corner-response map: high at
           terminals/corners/joins, low along straight edges.
        3. Blend two Gaussian blurs — strong at corners, mild on
           straight edges — weighted by the corner response.
           Each batch sees randomly jittered sigma values to prevent
           the model from learning to invert one specific blur kernel.
        4. Mix the blended blur with the original image, gated by
           the edge mask, a per-sample random factor, and per-pixel
           spatial noise so the corruption pattern varies across the
           glyph surface.
        5. Optionally inject additive Gaussian noise on edge regions
           for robustness to generation artifacts.

        The clean ``high_res`` tensor is still used as the training
        target — only the low-res input sees the corruption.
        """
        B, C, H, W = high_res.shape
        device = high_res.device
        dtype = high_res.dtype

        # ── Edge mask ──────────────────────────────────────────────────
        edge = self._edge_magnitude(high_res)  # (B, 1, H, W)
        edge_norm = edge / (edge.amax(dim=(2, 3), keepdim=True) + 1e-6)
        edge_mask = (edge_norm >= self.outline_noise_edge_threshold).to(dtype)

        # ── Corner response ────────────────────────────────────────────
        corner = self._corner_response(high_res)  # (B, H, W)

        # ── Per-batch sigma jitter ─────────────────────────────────────
        # Each batch sees a different blur kernel strength, sampled
        # uniformly in [base*(1-jitter), base*(1+jitter)].  This forces
        # the model to learn a general sharpening capability rather than
        # memorising a specific inverse filter.
        jitter = (
            (torch.rand(1, device=device) * 2.0 - 1.0)
            * self.blur_sigma_jitter
        )
        eff_terminal = max(self.terminal_blur_sigma * (1.0 + jitter.item()), 0.1)
        eff_stem = max(self.stem_blur_sigma * (1.0 + jitter.item()), 0.1)

        # ── Curvature-weighted blur blend ──────────────────────────────
        blurred_strong = self._gaussian_blur(high_res, eff_terminal)
        blurred_mild = self._gaussian_blur(high_res, eff_stem)

        corner = corner.unsqueeze(1)  # (B, 1, H, W)
        blended_blur = blurred_mild * (1.0 - corner) + blurred_strong * corner

        # ── Mix with original ──────────────────────────────────────────
        # Per-sample random mix strength
        blur_mix = torch.rand((B, 1, 1, 1), dtype=dtype, device=device)
        blur_mix = (
            self.blur_mix_min
            + blur_mix * (self.blur_mix_max - self.blur_mix_min)
        )

        # Per-pixel spatial noise on mix weight so the blur pattern
        # varies across the glyph surface within a single sample.
        spatial_noise = (
            torch.rand((B, 1, H, W), dtype=dtype, device=device) * 2.0 - 1.0
        ) * self.mix_spatial_noise

        mix_weight = (edge_mask * blur_mix + spatial_noise).clamp(0.0, 1.0)

        mixed = high_res * (1.0 - mix_weight) + blended_blur * mix_weight

        # ── Additive outline noise (optional) ──────────────────────────
        if self.outline_noise_std > 0.0:
            noise = torch.randn_like(high_res) * self.outline_noise_std
            mixed = mixed + noise * edge_mask

        return mixed.clamp(0.0, 1.0)

    def _inject_monochrome_low_res_noise(self, low_res: torch.Tensor) -> torch.Tensor:
        replace_mask = (
            torch.rand(
                (low_res.shape[0], 1, low_res.shape[2], low_res.shape[3]),
                dtype=low_res.dtype,
                device=low_res.device,
            )
            < self.low_res_noise_std
        )
        replacement_gray = torch.rand(
            (low_res.shape[0], 1, low_res.shape[2], low_res.shape[3]),
            dtype=low_res.dtype,
            device=low_res.device,
        )
        replacement = replacement_gray.expand_as(low_res)
        return torch.where(replace_mask.expand_as(low_res), replacement, low_res)

    @staticmethod
    def _sample_style_gids(font: object, exclude_gid: int, count: int) -> list[int]:
        """Sample *count* valid (non-empty) gids from *font*, excluding *exclude_gid*.

        Uses rejection sampling with a budget of 10×*count* attempts.
        If not enough distinct gids are found, the list is padded by
        repeating the last-found gid.
        """
        glyph_count: int = font.hb_face.glyph_count  # type: ignore[union-attr]
        gids: list[int] = []
        attempts = 0
        budget = count * 10
        while len(gids) < count and attempts < budget:
            candidate = int(torch.randint(1, max(2, glyph_count), ()).item())
            attempts += 1
            if candidate == exclude_gid:
                continue
            # Quick render at low res to check if the glyph has drawable geometry.
            raster = font.render_gid(candidate, size=64)  # type: ignore[union-attr]
            if not np.allclose(raster, 1.0, atol=1e-2):
                gids.append(candidate)
        # Pad with the last-found gid (or exclude_gid as absolute fallback).
        fallback = gids[-1] if gids else exclude_gid
        while len(gids) < count:
            gids.append(fallback)
        return gids

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

        # --- Style references ---
        style_refs: list[torch.Tensor] = []
        if self.style_reference_count > 0:
            for item in batch:
                ref_gids = self._sample_style_gids(
                    item["font"], item["gid"], self.style_reference_count
                )
                ref_rasters = [
                    torch.tensor(
                        item["font"].render_gid(g, size=self.high_res_size),
                        dtype=torch.float32,
                    )
                    for g in ref_gids
                ]
                style_refs.append(torch.stack(ref_rasters))  # (K, 3, H, W)

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
            low_res = self._inject_monochrome_low_res_noise(low_res)

        result = {
            "gid": gids,
            "low_res": low_res,
            "high_res": high_res,
        }
        if style_refs:
            result["style_references"] = torch.stack(style_refs)  # (B, K, 3, H, W)
        return result


__all__ = ["UpscalerDatasetMaker"]
