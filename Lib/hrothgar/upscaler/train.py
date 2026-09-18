"""Training script for glyph super-resolution."""

from __future__ import annotations

import itertools
import os

import torch
import torch.nn.functional as F
import torchvision
import tqdm

from hrothgar.upscaler.dataset import UpscalerDatasetMaker
from hrothgar.upscaler.model import UpscalerConfig, UpscalerModel
from hrothgar.utils import TrainingLoop


def compute_upscaler_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Pixel L1 loss (the standard objective for grayscale super-resolution)."""
    loss = F.l1_loss(predictions, targets)
    return loss, {"l1": loss, "loss": loss}


class UpscalerTrainingLoop(TrainingLoop):
    """Training loop for the glyph SR model."""

    def post_init(self, train_args):
        config = UpscalerConfig(
            low_res_size=train_args.low_res_size,
            high_res_size=train_args.low_res_size * train_args.upscaling_factor,
            base_channels=train_args.base_channels,
            num_residual_blocks=train_args.num_residual_blocks,
        )
        model = UpscalerModel(config).to(self.device)
        config.save_sidecar(train_args.model_path)

        maker = UpscalerDatasetMaker(
            train_args.dataset_path,
            batch_size=train_args.batch_size,
            low_res_size=config.low_res_size,
            high_res_size=config.high_res_size,
        )

        self.train_loader = maker.train_loader()
        self.test_loader = maker.test_loader()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_args.learning_rate,
            betas=(train_args.beta1, train_args.beta2),
        )

        self.model = model
        self.target_steps = train_args.target_steps
        self.validation_every = train_args.validation_every
        self.validation_batches = train_args.validation_batches
        self.num_epochs = (train_args.target_steps // len(self.train_loader)) + 1
        self.validation_direction = "lower"

    def train_step(self, batch):
        low_res = batch["low_res"].to(self.device)
        high_res = batch["high_res"].to(self.device)
        predictions = self.model(low_res)
        return compute_upscaler_loss(predictions, high_res)

    def post_train_step(self):
        if self.global_step % self.validation_every != 0:
            return

        self.model.eval()
        with torch.no_grad():
            val_l1 = []
            for val_batch in tqdm.tqdm(
                itertools.islice(self.test_loader, self.validation_batches),
                desc="Validation",
                total=self.validation_batches,
            ):
                low_res = val_batch["low_res"].to(self.device)
                high_res = val_batch["high_res"].to(self.device)
                pred = self.model(low_res)
                val_l1.append(F.l1_loss(pred, high_res))

            avg_l1 = torch.mean(torch.stack(val_l1))
            self.write_scalar("Validation/L1", avg_l1)
            self.checkpoint_if_best(avg_l1)
            self.visualize()

        self.model.train()

    def visualize(self):
        val_batch = next(iter(self.test_loader))
        low_res = val_batch["low_res"].to(self.device)
        high_res = val_batch["high_res"].to(self.device)
        pred = self.model(low_res)

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
        "--upscaling-factor",
        type=int,
        default=4,
        help="Output glyph raster size is upscaling_factor * low_res_size",
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
