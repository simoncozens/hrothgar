"""Tokeniser reconstruction quality validator for G-Tok.

Evaluates whether a trained G-Tok tokeniser can faithfully reconstruct style
glyphs before the AR generator is trained.  Poor reconstruction of fine
details (corners, serifs, curves) strongly suggests the generator will
struggle, so this module acts as a gate — similar to the linear probing and
autocorrelation gates already in ``hrothgar.gtok``.

Reports min / avg / max across four complementary metrics:

* **L1** — per-pixel absolute error, sensitive to overall intensity mismatch.
* **LPIPS** — learned perceptual similarity (Zhang et al.), correlates well
  with human judgement of fine-structure fidelity.
* **SSIM** — structural similarity (Wang et al.), sensitive to local
  luminance, contrast, and structure changes (1.0 = perfect).
* **VGG** — MSE between VGG-19 feature maps (conv2_2), a coarser perceptual
  distance.

Usage as a module::

    from hrothgar.gtok.validator import GtokValidator, ValidatorConfig
    config = ValidatorConfig(
        gtok_model_path="models/gtok_model.pth",
        dataset_path="/path/to/google/fonts",
        font_family="EB Garamond",
    )
    validator = GtokValidator(config)
    results = validator.run()
    if results["pass"]:
        print("✓ Tokeniser reconstruction quality is sufficient")
    else:
        print("✗ Tokeniser reconstruction quality is insufficient")

CLI::

    python -m hrothgar.gtok.validator \\
        --gtok-model-path models/gtok_model.pth \\
        --dataset-path $GOOGLE_FONTS_REPO \\
        --font-family "EB Garamond"
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure

from hrothgar.googlefonts import Font, GoogleFonts
from hrothgar.gtok.llamagen_lpips import LPIPS
from hrothgar.gtok.model import load_model
from hrothgar.gtok.vgg_loss import VGG
from hrothgar.utils import torch_setup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ValidatorConfig:
    """Configuration for G-Tok reconstruction quality validation.

    Attributes:
        gtok_model_path: Path to a trained G-Tok ``.pth`` weights file.
            A ``.conf.json`` sidecar must exist alongside it.
        dataset_path: Path to the Google Fonts repository.
        font_family: Optional specific font family to validate.  If
            ``None``, samples ``max_fonts`` families randomly.
        max_fonts: Maximum number of font families to validate when no
            specific family is requested.
        probe_chars: Characters to use for validation.  These are the
            "style glyphs" whose reconstruction quality is being assessed.
        batch_size: Batch size for model inference.
        seed: RNG seed for reproducibility.

        l1_max_threshold: Maximum acceptable average L1 error across all
            glyphs (pass/fail gate).
        lpips_max_threshold: Maximum acceptable average LPIPS distance.
        ssim_min_threshold: Minimum acceptable average SSIM.
        vgg_max_threshold: Maximum acceptable average VGG distance.
    """

    gtok_model_path: str = "models/gtok_model.pth"
    dataset_path: str = os.environ.get("GOOGLE_FONTS_REPO", "")
    font_family: Optional[str] = None
    max_fonts: int = 20
    probe_chars: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    batch_size: int = 16
    seed: int = 42

    # Pass / fail thresholds (applied to the per-family *mean*).
    l1_max_threshold: float = 0.15
    lpips_max_threshold: float = 0.30
    ssim_min_threshold: float = 0.85
    vgg_max_threshold: float = 1.0


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


class L1Metric:
    """Mean absolute (L1) pixel error between two normalised images."""

    @torch.no_grad()
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute per-sample scalar L1 error.

        Args:
            a, b: ``(B, C, H, W)`` tensors in [0, 1].

        Returns:
            Scalar tensor: mean L1 across the batch.
        """
        return torch.mean(torch.abs(a - b))


class LPIPSMetric:
    """Learned Perceptual Image Patch Similarity (Zhang et al.).

    Lower is better.  Values typically in [0, 1] for natural images.
    Returns per-sample values.
    """

    def __init__(self, device: torch.device) -> None:
        self._lpips = LPIPS().to(device)
        self._lpips.eval()

    @torch.no_grad()
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute per-sample LPIPS distances.

        Args:
            a, b: ``(B, C, H, W)`` tensors in [0, 1].

        Returns:
            ``(B,)`` tensor of per-sample LPIPS values.
        """
        # LPIPS expects input in [-1, 1]; our images are [0, 1].
        a_norm = a * 2.0 - 1.0
        b_norm = b * 2.0 - 1.0
        # self._lpips returns (B, 1, 1, 1) — squeeze to (B,).
        return self._lpips(a_norm, b_norm).reshape(-1)


class VGGMetric:
    """VGG-19 perceptual distance (conv2_2 features).

    Lower is better.  Unbounded — values depend on image content.
    """

    def __init__(self, device: torch.device) -> None:
        self._vgg = VGG(conv_index="22").to(device)
        self._vgg.eval()

    @torch.no_grad()
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute per-sample VGG feature distance.

        Args:
            a, b: ``(B, C, H, W)`` tensors in [0, 1].

        Returns:
            Scalar tensor: mean VGG MSE across the batch.
        """
        return self._vgg(a, b)


class SSIMMetric:
    """Structural Similarity Index (Wang et al.).

    Higher is better.  Values in [-1, 1]; 1.0 is perfect reconstruction.
    """

    def __init__(self, device: torch.device) -> None:
        self._ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    @torch.no_grad()
    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute per-sample SSIM.

        Args:
            a, b: ``(B, C, H, W)`` tensors in [0, 1].

        Returns:
            Scalar tensor: mean SSIM across the batch.
        """
        return self._ssim(a, b)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ReconstructionDataset(torch.utils.data.Dataset):
    """Dataset that renders glyphs for reconstruction quality assessment.

    Each item is a ``(font, codepoint)`` tuple.  The collate function
    renders images on the fly.
    """

    def __init__(self, samples: List[Tuple[Font, int]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def _collate_render(batch: List[Tuple[Font, int]], *, image_size: int) -> torch.Tensor:
    """Render a batch of (font, codepoint) samples into a stacked tensor.

    Returns:
        ``(B, 3, image_size, image_size)`` float32 tensor in [0, 1].
    """
    images = [
        torch.tensor(font.render(cp, size=image_size), dtype=torch.float32)
        for font, cp in batch
    ]
    return torch.stack(images, dim=0)


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------


class GtokValidator:
    """Validate G-Tok reconstruction quality on a range of style glyphs.

    For each selected font family, renders ``probe_chars`` glyphs, passes
    them through the frozen G-Tok tokeniser, and compares input vs
    reconstruction using L1, LPIPS, SSIM, and VGG metrics.

    Reports per-family min / avg / max for each metric and a final
    pass / fail gate decision.
    """

    def __init__(self, config: ValidatorConfig) -> None:
        self.config = config
        self.device = torch_setup()

        # --- Load G-Tok ---
        gtok, gtok_config = load_model(Path(config.gtok_model_path), device=self.device)
        self.gtok = gtok
        self.gtok_config = gtok_config
        self.image_size: int = gtok_config.image_size
        self.gtok.eval()
        for param in self.gtok.parameters():
            param.requires_grad = False

        # --- Initialise metrics ---
        self._metrics: Dict[str, object] = {
            "l1": L1Metric(),
            "lpips": LPIPSMetric(self.device),
            "ssim": SSIMMetric(self.device),
            "vgg": VGGMetric(self.device),
        }

        # --- Probe characters ---
        self._probe_codepoints: List[int] = [ord(c) for c in config.probe_chars]

        # --- Build datasets ---
        self._datasets: Dict[str, ReconstructionDataset] = {}
        self._build_datasets()

    def _build_datasets(self) -> None:
        """Select font families and build per-family reconstruction datasets."""
        cfg = self.config
        rng = np.random.RandomState(cfg.seed)

        gf = GoogleFonts(cfg.dataset_path)
        # Apply the same display filter as GTok training.
        gf.fonts = [font for font in gf.fonts if font.display_score() < 60.0]

        # Group fonts by family.
        family_to_fonts: Dict[str, List[Font]] = {}
        for font in gf.fonts:
            family_to_fonts.setdefault(font.family, []).append(font)

        # Select families.
        if cfg.font_family:
            if cfg.font_family not in family_to_fonts:
                raise ValueError(
                    f"Font family {cfg.font_family!r} not found in the repository. "
                    f"Available families include: "
                    f"{', '.join(sorted(family_to_fonts.keys())[:20])}..."
                )
            selected_families = [cfg.font_family]
        else:
            all_families = sorted(family_to_fonts.keys())
            rng.shuffle(all_families)
            selected_families = all_families[: cfg.max_fonts]

        print(f"Validating {len(selected_families)} font families...")
        print(
            f"Probe characters: {cfg.probe_chars}  ({len(self._probe_codepoints)} glyphs)"
        )

        # Build per-family sample lists.
        for family in selected_families:
            fonts = family_to_fonts[family]
            # Use the first available font file that has all probe characters.
            usable_font: Optional[Font] = None
            for font in fonts:
                if all(font.has_codepoint(cp) for cp in self._probe_codepoints):
                    usable_font = font
                    break
            if usable_font is None:
                print(f"  ⚠  Skipping {family!r}: no single font has all probe chars")
                continue

            samples: List[Tuple[Font, int]] = [
                (usable_font, cp) for cp in self._probe_codepoints
            ]
            self._datasets[family] = ReconstructionDataset(samples)

    def _make_loader(self, dataset: ReconstructionDataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=lambda batch: _collate_render(batch, image_size=self.image_size),
            num_workers=0,  # GPU encoding prohibits forking.
            pin_memory=False,
        )

    def _compute_family_metrics(
        self, dataset: ReconstructionDataset
    ) -> Dict[str, Dict[str, float]]:
        """Reconstruct all glyphs in a family and compute per-metric stats.

        Returns:
            Dict mapping metric names to ``{"min": float, "avg": float, "max": float}``.
        """
        loader = self._make_loader(dataset)

        # Accumulate per-sample metric values.
        all_values: Dict[str, List[float]] = {name: [] for name in self._metrics}

        for batch in tqdm.tqdm(loader, desc="  Reconstructing", leave=False):
            images = batch.to(self.device)  # (B, 3, H, W)

            with torch.no_grad():
                reconstructed, _loss_info = self.gtok(images)

            # Compute each metric.
            # L1 and LPIPS both return per-sample tensors directly.
            # SSIM and VGG return a scalar average; we compute per-sample
            # in a loop (acceptable for a one-off validation gate).
            for name, metric in self._metrics.items():
                if name == "l1":
                    per_sample = (
                        (images - reconstructed)
                        .abs()
                        .reshape(images.shape[0], -1)
                        .mean(dim=1)
                    )
                    all_values[name].extend(per_sample.cpu().tolist())
                elif name == "lpips":
                    per_sample = metric(images, reconstructed)  # (B,)
                    all_values[name].extend(per_sample.cpu().tolist())
                else:
                    # SSIM / VGG: compute one sample at a time.
                    vals: List[float] = []
                    for i in range(images.shape[0]):
                        vals.append(
                            metric(images[i : i + 1], reconstructed[i : i + 1]).item()
                        )
                    all_values[name].extend(vals)

        # Build per-metric stats.
        stats: Dict[str, Dict[str, float]] = {}
        for name, values in all_values.items():
            arr = np.array(values)
            stats[name] = {
                "min": float(np.min(arr)),
                "avg": float(np.mean(arr)),
                "max": float(np.max(arr)),
            }

        return stats

    def run(self) -> Dict[str, object]:
        """Run validation across all selected font families.

        Returns:
            Dict with keys:

            * ``"per_family"``: ``Dict[str, Dict[str, Dict[str, float]]]`` —
              per-family → per-metric → ``{min, avg, max}``.
            * ``"overall"``: ``Dict[str, Dict[str, float]]`` —
              aggregate stats across all families.
            * ``"pass"``: ``bool`` — whether all families pass the thresholds.
            * ``"failures"``: ``List[str]`` — list of failing families with
              details.
        """
        per_family: Dict[str, Dict[str, Dict[str, float]]] = {}
        all_avgs: Dict[str, List[float]] = {name: [] for name in self._metrics}

        for family, dataset in self._datasets.items():
            print(f"\n--- {family} ---")
            stats = self._compute_family_metrics(dataset)
            per_family[family] = stats

            for name in self._metrics:
                all_avgs[name].append(stats[name]["avg"])

            # Print per-family summary.
            l1_str = f"L1:    min={stats['l1']['min']:.4f}  avg={stats['l1']['avg']:.4f}  max={stats['l1']['max']:.4f}"
            lpips_str = f"LPIPS: min={stats['lpips']['min']:.4f}  avg={stats['lpips']['avg']:.4f}  max={stats['lpips']['max']:.4f}"
            ssim_str = f"SSIM:  min={stats['ssim']['min']:.4f}  avg={stats['ssim']['avg']:.4f}  max={stats['ssim']['max']:.4f}"
            vgg_str = f"VGG:   min={stats['vgg']['min']:.4f}  avg={stats['vgg']['avg']:.4f}  max={stats['vgg']['max']:.4f}"
            print(f"  {l1_str}")
            print(f"  {lpips_str}")
            print(f"  {ssim_str}")
            print(f"  {vgg_str}")

        # --- Overall stats ---
        overall: Dict[str, Dict[str, float]] = {}
        for name in self._metrics:
            arr = np.array(all_avgs[name])
            overall[name] = {
                "min": float(np.min(arr)),
                "avg": float(np.mean(arr)),
                "max": float(np.max(arr)),
            }

        # --- Pass / fail gate ---
        cfg = self.config
        failures: List[str] = []

        for family, stats in per_family.items():
            family_failures: List[str] = []
            if stats["l1"]["avg"] > cfg.l1_max_threshold:
                family_failures.append(
                    f"L1={stats['l1']['avg']:.4f} > {cfg.l1_max_threshold}"
                )
            if stats["lpips"]["avg"] > cfg.lpips_max_threshold:
                family_failures.append(
                    f"LPIPS={stats['lpips']['avg']:.4f} > {cfg.lpips_max_threshold}"
                )
            if stats["ssim"]["avg"] < cfg.ssim_min_threshold:
                family_failures.append(
                    f"SSIM={stats['ssim']['avg']:.4f} < {cfg.ssim_min_threshold}"
                )
            if stats["vgg"]["avg"] > cfg.vgg_max_threshold:
                family_failures.append(
                    f"VGG={stats['vgg']['avg']:.4f} > {cfg.vgg_max_threshold}"
                )
            if family_failures:
                failures.append(f"{family}: " + ", ".join(family_failures))

        passed = len(failures) == 0

        # --- Print summary ---
        print(f"\n{'=' * 60}")
        print("Overall reconstruction quality")
        print(f"{'=' * 60}")
        for name, label in [
            ("l1", "L1   "),
            ("lpips", "LPIPS"),
            ("ssim", "SSIM "),
            ("vgg", "VGG  "),
        ]:
            s = overall[name]
            print(
                f"  {label}:  min={s['min']:.4f}  avg={s['avg']:.4f}  max={s['max']:.4f}"
            )

        print(f"\n{'=' * 60}")
        if passed:
            print("✓ ALL FAMILIES PASS — tokeniser reconstruction is sufficient")
        else:
            print("✗ SOME FAMILIES FAIL — tokeniser may need improvement")
            for failure in failures:
                print(f"  • {failure}")
        print(f"{'=' * 60}")
        return {
            "per_family": per_family,
            "overall": overall,
            "pass": passed,
            "failures": failures,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruction quality validator for G-Tok tokeniser"
    )
    parser.add_argument(
        "--gtok-model-path",
        type=str,
        default="models/gtok_model.pth",
        help="Path to trained G-Tok weights (.pth); .conf.json must exist beside it",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=os.environ.get("GOOGLE_FONTS_REPO", ""),
        help="Path to the Google Fonts repository",
    )
    parser.add_argument(
        "--font-family",
        type=str,
        default=None,
        help="Specific font family to validate (default: sample randomly)",
    )
    parser.add_argument(
        "--max-fonts",
        type=int,
        default=20,
        help="Maximum font families to validate when no --font-family is given",
    )
    parser.add_argument(
        "--probe-chars",
        type=str,
        default="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        help="Characters to use for reconstruction validation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for model inference",
    )
    parser.add_argument(
        "--l1-threshold",
        type=float,
        default=0.15,
        help="Maximum acceptable average L1 error",
    )
    parser.add_argument(
        "--lpips-threshold",
        type=float,
        default=0.30,
        help="Maximum acceptable average LPIPS distance",
    )
    parser.add_argument(
        "--ssim-threshold",
        type=float,
        default=0.85,
        help="Minimum acceptable average SSIM",
    )
    parser.add_argument(
        "--vgg-threshold",
        type=float,
        default=1.0,
        help="Maximum acceptable average VGG distance",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed",
    )
    return parser


def main() -> None:
    """Entry point for ``python -m hrothgar.gtok.validator``."""
    parser = _build_parser()
    args = parser.parse_args()

    if not args.dataset_path:
        parser.error(
            "--dataset-path is required (or set GOOGLE_FONTS_REPO environment variable)"
        )

    config = ValidatorConfig(
        gtok_model_path=args.gtok_model_path,
        dataset_path=args.dataset_path,
        font_family=args.font_family,
        max_fonts=args.max_fonts,
        probe_chars=args.probe_chars,
        batch_size=args.batch_size,
        seed=args.seed,
        l1_max_threshold=args.l1_threshold,
        lpips_max_threshold=args.lpips_threshold,
        ssim_min_threshold=args.ssim_threshold,
        vgg_max_threshold=args.vgg_threshold,
    )

    validator = GtokValidator(config)
    results = validator.run()

    # Exit with non-zero if validation fails (for CI / scripting).
    if not results["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
