import itertools
import os

import torch
import torchvision
import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure

from hrothgar.gtok import compute_gtok_loss
from hrothgar.gtok.dataset import GTokDatasetMaker
from hrothgar.gtok.llamagen_lpips import LPIPS
from hrothgar.gtok.model import GtokConfig, GtokModel
from hrothgar.gtok.vgg_loss import VGG
from hrothgar.utils import TrainingLoop


def _parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated integer list from CLI input."""
    items = [part.strip() for part in value.split(",") if part.strip()]
    if not items:
        raise ValueError("Expected a comma-separated list of integers")
    return [int(item) for item in items]


class GtokTrainingLoop(TrainingLoop):
    def post_init(self, train_args):
        gtok_config_kwargs = {
            "image_size": train_args.image_size,
        }
        if train_args.cnn_channel_multipliers is not None:
            gtok_config_kwargs["cnn_channel_multipliers"] = (
                train_args.cnn_channel_multipliers
            )
        if train_args.cnn_latent_channels is not None:
            gtok_config_kwargs["cnn_latent_channels"] = train_args.cnn_latent_channels
        if train_args.quantizer_codebook_size is not None:
            gtok_config_kwargs["quantizer_codebook_size"] = (
                train_args.quantizer_codebook_size
            )
        if train_args.quantizer_code_dim is not None:
            gtok_config_kwargs["quantizer_code_dim"] = train_args.quantizer_code_dim
        if train_args.quantizer_entropy_loss_ratio is not None:
            gtok_config_kwargs["quantizer_entropy_loss_ratio"] = (
                train_args.quantizer_entropy_loss_ratio
            )

        config = GtokConfig(**gtok_config_kwargs)
        model = GtokModel(config).to(self.device)
        # Batch size 16 / LR 1e-4 / AdamW are specified in paper, don't mess with them.
        maker = GTokDatasetMaker(
            train_args.dataset_path, batch_size=16, image_size=config.image_size
        )
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        self.train_loader = maker.train_loader()
        self.test_loader = maker.test_loader()
        self.perceptual_loss_fn = VGG().to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips = LPIPS().to(self.device)
        self.model = model
        self.target_steps = 200_000  # Specified in paper
        if train_args.canary != 0:
            # Run for ten epochs, i.e. 10 * len(train) steps
            train_loader = list(itertools.islice(self.train_loader, train_args.canary))
            self.target_steps = 10 * len(train_loader)

        self.num_epochs = (self.target_steps // len(self.train_loader)) + 1
        self.validation_direction = "higher"  # We want to maximize SSIM

    def train_step(self, batch):
        gt_images = batch["rendering"].to(self.device)
        recon_images, vq_loss_info = self.model(gt_images)
        loss, loss_info = compute_gtok_loss(
            recon_images,
            gt_images,
            vq_loss_info,
            perceptual_loss_fn=self.perceptual_loss_fn,
        )
        return loss, loss_info

    def post_train_step(self):
        # Do some validation every 1000 steps
        if self.global_step % 1000 != 0:
            return
        self.model.eval()
        with torch.no_grad():
            # Compute SSIM and LPIPS on the validation set and log them to TensorBoard
            val_metrics = {"ssim": [], "lpips": []}
            # Just do 100 batches
            for val_batch in tqdm.tqdm(
                itertools.islice(self.test_loader, 100),
                desc="Validation",
                total=100,
            ):
                val_gt_images = val_batch["rendering"].to(self.device)
                val_recon_images, _ = self.model(val_gt_images)
                val_metrics["ssim"].append(self.ssim(val_recon_images, val_gt_images))
                val_metrics["lpips"].append(self.lpips(val_recon_images, val_gt_images))
            avg_ssim = torch.mean(torch.stack(val_metrics["ssim"]))
            avg_lpips = torch.mean(torch.stack(val_metrics["lpips"]))
            self.write_scalar("Validation/SSIM", avg_ssim)
            self.write_scalar("Validation/LPIPS", avg_lpips)
            self.checkpoint_if_best(avg_ssim)
            self.visualize()
        self.model.train()

    def visualize(self):
        # Also display some pretty pictures
        val_batch = next(iter(self.test_loader))
        val_gt_images = val_batch["rendering"].to(self.device)
        val_recon_images, _ = self.model(val_gt_images)
        # Log a grid of reconstructed vs. target images
        recon_grid = torch.cat([val_gt_images[:16], val_recon_images[:16]], dim=0)
        self.writer.add_image(
            "Reconstruction/GT_vs_Recon",
            torchvision.utils.make_grid(recon_grid, nrow=16),
            self.global_step,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train GTok model")
    parser.add_argument(
        "--canary",
        type=int,
        default=0,
        help="If nonzero, run a canary test instead of full training",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=os.environ.get("GOOGLE_FONTS_REPO"),
        help="Path to the Google Fonts repository",
    )
    parser.add_argument(
        "--tag",
        type=str,
        help="Tag for the training run",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Square glyph raster size for GTok training.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to save the trained model",
        default="gtok_model.pth",
    )
    parser.add_argument(
        "--cnn-channel-multipliers",
        type=_parse_int_list,
        default=None,
        help=(
            "Optional comma-separated CNN channel multipliers for the tokenizer "
            "pyramid (for example: 1,2,2,4,4)."
        ),
    )
    parser.add_argument(
        "--cnn-latent-channels",
        type=int,
        default=None,
        help="Optional tokenizer latent channel count override.",
    )
    parser.add_argument(
        "--quantizer-codebook-size",
        type=int,
        default=None,
        help="Optional VQ codebook size override.",
    )
    parser.add_argument(
        "--quantizer-code-dim",
        type=int,
        default=None,
        help="Optional VQ code dimensionality override.",
    )
    parser.add_argument(
        "--quantizer-entropy-loss-ratio",
        type=float,
        default=None,
        help="Optional entropy regularization weight override for the VQ quantizer.",
    )
    args = parser.parse_args()
    if not args.dataset_path:
        raise ValueError(
            "GOOGLE_FONTS_REPO environment variable not set, cannot run training"
        )
    loop = GtokTrainingLoop(args)
    loop.train()
