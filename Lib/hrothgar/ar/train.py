import itertools
import os

import torch
import torchvision
import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure

from hrothgar.ar.dataset import ARPhase1DatasetMaker
from hrothgar.ar.losses import ARLossWeights, compute_ar_loss
from hrothgar.ar.model import ARModel, ARModelConfig
from hrothgar.gtok.llamagen_lpips import LPIPS
from hrothgar.gtok.model import GtokModel, GtokConfig
from hrothgar.utils import TrainingLoop


class ARVisualTrainingLoop(TrainingLoop):
    """Visual-only AR stage training loop.

    This matches the GAR-Font phase-1 setup: AdamW with paper betas, one
    reference-font content glyph, and configurable N_s style references.
    """

    def post_init(self, train_args):
        config = ARModelConfig(image_size=train_args.image_size)
        if not os.path.exists(train_args.gtok_model_path):
            raise ValueError(
                f"G-Tok model not found at {train_args.gtok_model_path}, cannot run AR training"
            )
        gtok = GtokModel(GtokConfig())
        gtok.load(train_args.gtok_model_path, device=self.device)
        model = ARModel(config, gtok_model=gtok).to(self.device)

        maker = ARPhase1DatasetMaker(
            train_args.dataset_path,
            batch_size=32,
            image_size=config.image_size,
            style_glyph_count=train_args.style_glyph_count,
        )

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.95),
        )
        self.train_loader = maker.train_loader()
        self.test_loader = maker.test_loader()

        self.loss_weights = ARLossWeights()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips = LPIPS().to(self.device)

        self.model = model
        self.target_steps = train_args.target_steps
        self.validation_every = train_args.validation_every
        self.validation_batches = train_args.validation_batches

        if train_args.canary != 0:
            self.train_loader = list(
                itertools.islice(self.train_loader, train_args.canary)
            )
            if len(self.train_loader) == 0:
                raise ValueError("Canary mode produced an empty train loader")
            # Run for ten epochs over the canary slice.
            self.target_steps = 10 * len(self.train_loader)

        self.num_epochs = (self.target_steps // len(self.train_loader)) + 1
        self.validation_direction = "higher"  # Maximize SSIM.

    def train_step(self, batch):
        target_images = batch["target_rendering"].to(self.device)
        content_images = batch["content_rendering"].to(self.device)
        style_reference_images = batch["style_renderings"].to(self.device)

        model_output = self.model(
            content_images,
            style_reference_images,
            target_images=target_images,
        )
        loss, loss_info = compute_ar_loss(
            model_output,
            target_images,
            weights=self.loss_weights,
        )
        return loss, loss_info

    def post_train_step(self):
        if self.global_step % self.validation_every != 0:
            return

        self.model.eval()
        with torch.no_grad():
            val_metrics = {"ssim": [], "lpips": []}
            for val_batch in tqdm.tqdm(
                itertools.islice(self.test_loader, self.validation_batches),
                desc="Validation",
                total=self.validation_batches,
            ):
                val_target_images = val_batch["target_rendering"].to(self.device)
                val_content_images = val_batch["content_rendering"].to(self.device)
                val_style_images = val_batch["style_renderings"].to(self.device)

                val_output = self.model(
                    val_content_images,
                    val_style_images,
                    target_images=val_target_images,
                )
                val_metrics["ssim"].append(
                    self.ssim(val_output.reconstructed_images, val_target_images)
                )
                val_metrics["lpips"].append(
                    self.lpips(val_output.reconstructed_images, val_target_images)
                )

            avg_ssim = torch.mean(torch.stack(val_metrics["ssim"]))
            avg_lpips = torch.mean(torch.stack(val_metrics["lpips"]))
            self.write_scalar("Validation/SSIM", avg_ssim)
            self.write_scalar("Validation/LPIPS", avg_lpips)
            self.checkpoint_if_best(avg_ssim)
            self.visualize()

        self.model.train()

    def visualize(self):
        val_batch = next(iter(self.test_loader))
        val_target_images = val_batch["target_rendering"].to(self.device)
        val_content_images = val_batch["content_rendering"].to(self.device)
        val_style_images = val_batch["style_renderings"].to(self.device)

        val_output = self.model(
            val_content_images,
            val_style_images,
            target_images=val_target_images,
        )

        preview_count = min(8, val_target_images.shape[0])
        first_style = val_style_images[:preview_count, 0]
        recon_grid = torch.cat(
            [
                val_content_images[:preview_count],
                first_style,
                val_target_images[:preview_count],
                val_output.reconstructed_images[:preview_count],
            ],
            dim=0,
        )
        self.writer.add_image(
            "Reconstruction/content_style_target_recon",
            torchvision.utils.make_grid(recon_grid, nrow=preview_count),
            self.global_step,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train AR visual-only model")
    parser.add_argument(
        "--canary",
        type=int,
        default=0,
        help="If nonzero, use this many train batches and run a short canary loop",
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
        help="Square glyph raster size for AR training",
    )
    parser.add_argument(
        "--style-glyph-count",
        type=int,
        default=8,
        help="Number of style glyph references N_s (paper default: 8)",
    )
    parser.add_argument(
        "--target-steps",
        type=int,
        default=600_000,
        help="Training iterations (paper: 600k for small set, 1M for large set)",
    )
    parser.add_argument(
        "--validation-every",
        type=int,
        default=1000,
        help="Run validation every N optimization steps",
    )
    parser.add_argument(
        "--validation-batches",
        type=int,
        default=100,
        help="Number of validation batches per validation pass",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to save the trained model",
        default="model/ar_visual_model.pth",
    )
    parser.add_argument(
        "--gtok-model-path",
        type=str,
        help="Path to load the trained tokenizer model",
        default="models/gtok_model.pth",
    )

    args = parser.parse_args()
    if not args.dataset_path:
        raise ValueError(
            "GOOGLE_FONTS_REPO environment variable not set, cannot run training"
        )

    loop = ARVisualTrainingLoop(args)
    loop.train()
