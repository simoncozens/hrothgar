"""Training script for glyph super-resolution prototype."""

from __future__ import annotations

import itertools
import os

import torch
import torch.nn.functional as F
import torchvision
import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure

from hrothgar.upscaler.dataset import UpscalerDatasetMaker
from hrothgar.upscaler.model import UpscalerConfig, UpscalerModel
from hrothgar.utils import TrainingLoop
from hrothgar.eagle_loss import EagleLoss


def _edge_map(images: torch.Tensor) -> torch.Tensor:
    """Compute finite-difference edge magnitude for edge-aware losses."""
    grayscale = images.mean(dim=1, keepdim=True)

    gx = grayscale[:, :, :, 1:] - grayscale[:, :, :, :-1]
    gy = grayscale[:, :, 1:, :] - grayscale[:, :, :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0), mode="replicate")
    gy = F.pad(gy, (0, 0, 0, 1), mode="replicate")
    magnitude = torch.sqrt(gx * gx + gy * gy + 1e-8)
    return magnitude


def _sanitize_for_bce(tensor: torch.Tensor, *, nan_fill: float) -> torch.Tensor:
    """Convert NaN/Inf to finite values and clamp to BCE's expected range."""
    return torch.nan_to_num(tensor, nan=nan_fill, posinf=1.0, neginf=0.0).clamp(
        0.0, 1.0
    )


def compute_upscaler_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    eagle_loss: EagleLoss,
    edge_weight: float = 1.0,
    eagle_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine pixel BCE with edge-aware L1 penalty."""
    predictions = _sanitize_for_bce(predictions, nan_fill=0.5)
    targets = _sanitize_for_bce(targets, nan_fill=0.0)

    bce = F.binary_cross_entropy(predictions, targets)
    pred_edges = _edge_map(predictions)
    target_edges = _edge_map(targets)
    edge_l1 = F.l1_loss(pred_edges, target_edges)
    eagle_contribution = (
        eagle_loss(predictions, targets) if eagle_loss_weight > 0 else torch.tensor(0.0)
    )

    loss = bce + edge_weight * edge_l1 + eagle_loss_weight * eagle_contribution
    return loss, {
        "bce": bce,
        "edge_l1": edge_l1,
        "eagle": eagle_contribution,
        "loss": loss,
    }


class UpscalerTrainingLoop(TrainingLoop):
    """Training loop for the glyph SR prototype."""

    def post_init(self, train_args):
        self.eagle_loss = EagleLoss(patch_size=8).to(self.device)
        config = UpscalerConfig(
            low_res_size=train_args.low_res_size,
            high_res_size=train_args.high_res_size,
            base_channels=train_args.base_channels,
            num_residual_blocks=train_args.num_residual_blocks,
            use_gtok_encoder=not train_args.disable_gtok_encoder,
            use_gtok_vit_features=not train_args.disable_gtok_vit,
            gtok_model_path=train_args.gtok_model_path,
        )
        model = UpscalerModel(config).to(self.device)

        maker = UpscalerDatasetMaker(
            train_args.dataset_path,
            batch_size=train_args.batch_size,
            low_res_size=config.low_res_size,
            high_res_size=config.high_res_size,
            style_conformance_mode=train_args.style_conformance_mode,
            clean_font_only=train_args.clean_font_only,
            clean_font_display_score_threshold=train_args.clean_font_display_score_threshold,
            outline_noise_std=train_args.outline_noise_std,
            outline_noise_edge_threshold=train_args.outline_noise_edge_threshold,
            low_res_noise_std=train_args.low_res_noise_std,
        )

        self.train_loader = maker.train_loader()
        self.test_loader = maker.test_loader()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_args.learning_rate,
            betas=(train_args.beta1, train_args.beta2),
        )
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

        self.model = model
        self.target_steps = train_args.target_steps
        self.validation_every = train_args.validation_every
        self.validation_batches = train_args.validation_batches
        self.num_epochs = (self.target_steps // len(self.train_loader)) + 1
        self.validation_direction = "higher"
        self.edge_weight = train_args.edge_weight

    def train_step(self, batch):
        low_res = batch["low_res"].to(self.device)
        high_res = batch["high_res"].to(self.device)
        descriptions = batch.get("description")
        predictions = self.model(low_res, descriptions=descriptions)
        return compute_upscaler_loss(
            predictions,
            high_res,
            edge_weight=self.edge_weight,
            eagle_loss=self.eagle_loss,
            eagle_loss_weight=0.0,
        )

    def post_train_step(self):
        if self.global_step % self.validation_every != 0:
            return

        self.model.eval()
        with torch.no_grad():
            val_ssim = []
            for val_batch in tqdm.tqdm(
                itertools.islice(self.test_loader, self.validation_batches),
                desc="Validation",
                total=self.validation_batches,
            ):
                low_res = val_batch["low_res"].to(self.device)
                high_res = val_batch["high_res"].to(self.device)
                pred = self.model(low_res, descriptions=val_batch.get("description"))
                val_ssim.append(self.ssim(pred, high_res))

            avg_ssim = torch.mean(torch.stack(val_ssim))
            self.write_scalar("Validation/SSIM", avg_ssim)
            self.checkpoint_if_best(avg_ssim)
            self.visualize()

        self.model.train()

    def visualize(self):
        val_batch = next(iter(self.test_loader))
        low_res = val_batch["low_res"].to(self.device)
        high_res = val_batch["high_res"].to(self.device)
        pred = self.model(low_res, descriptions=val_batch.get("description"))

        preview_count = min(8, low_res.shape[0])
        bicubic = F.interpolate(
            low_res[:preview_count],
            size=(high_res.shape[-2], high_res.shape[-1]),
            mode="bicubic",
            align_corners=False,
        )
        grid = torch.cat(
            [
                bicubic,
                pred[:preview_count],
                high_res[:preview_count],
            ],
            dim=0,
        )
        self.writer.add_image(
            "Upscaler/Bicubic_Pred_Target",
            torchvision.utils.make_grid(grid, nrow=preview_count),
            self.global_step,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train glyph super-resolution model")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=os.environ.get("GOOGLE_FONTS_REPO"),
        help="Path to the Google Fonts repository",
    )
    parser.add_argument("--tag", type=str, help="Tag for the training run")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow training with uncommitted changes in the git repository (not recommended)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for SR training",
    )
    parser.add_argument(
        "--low-res-size",
        type=int,
        default=128,
        help="Input glyph raster size",
    )
    parser.add_argument(
        "--high-res-size",
        type=int,
        default=512,
        help="Output glyph raster size",
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=64,
        help="Base channel width of the SR model",
    )
    parser.add_argument(
        "--num-residual-blocks",
        type=int,
        default=8,
        help="Number of residual blocks in the SR body",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="AdamW learning rate",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="AdamW beta1",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.95,
        help="AdamW beta2",
    )
    parser.add_argument(
        "--target-steps",
        type=int,
        default=200_000,
        help="Number of optimizer steps to train",
    )
    parser.add_argument(
        "--validation-every",
        type=int,
        default=1000,
        help="Run validation every N optimizer steps",
    )
    parser.add_argument(
        "--validation-batches",
        type=int,
        default=100,
        help="Validation batch count per validation pass",
    )
    parser.add_argument(
        "--disable-gtok-encoder",
        action="store_true",
        help="Disable GTok conditioning features",
    )
    parser.add_argument(
        "--disable-gtok-vit",
        action="store_true",
        help="Use GTok CNN features only (skip GTok ViT features)",
    )
    parser.add_argument(
        "--gtok-model-path",
        type=str,
        default="models/gtok_model.pth",
        help="Path to pretrained GTok weights (optional)",
    )
    parser.add_argument(
        "--style-conformance-mode",
        action="store_true",
        help=(
            "Corrupt low-res inputs with synthetic outline noise while keeping clean "
            "high-res targets, to train cleanup/conformance behavior"
        ),
    )
    parser.add_argument(
        "--clean-font-only",
        action="store_true",
        help="Filter out high-display fonts during SR training",
    )
    parser.add_argument(
        "--clean-font-display-score-threshold",
        type=float,
        default=45.0,
        help="Maximum display score to keep when --clean-font-only is set",
    )
    parser.add_argument(
        "--outline-noise-std",
        type=float,
        default=0.08,
        help="Stddev of edge-localized noise used in conformance mode",
    )
    parser.add_argument(
        "--outline-noise-edge-threshold",
        type=float,
        default=0.12,
        help="Normalized edge threshold (0-1) used to place outline noise",
    )
    parser.add_argument(
        "--low-res-noise-std",
        type=float,
        default=0.01,
        help="Per-pixel replacement probability applied after downsampling in conformance mode",
    )
    parser.add_argument(
        "--edge-weight",
        type=float,
        default=1.0,
        help="Relative weight of the edge-aware loss term compared to pixel BCE",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/upscaler_model.pth",
        help="Path to save SR model weights",
    )

    args = parser.parse_args()
    if not args.dataset_path:
        raise ValueError(
            "GOOGLE_FONTS_REPO environment variable not set, cannot run training"
        )

    loop = UpscalerTrainingLoop(args)
    loop.train()
