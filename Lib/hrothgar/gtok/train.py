import itertools
import os
from pathlib import Path
from typing import Dict, Optional, Set

import torch
import torchvision
import tqdm
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure

from glyphloss import GlyphReconstructionLoss
from hrothgar.googlefonts import GoogleFonts
from hrothgar.gtok import compute_gtok_loss
from hrothgar.gtok.config import GtokConfig, GtokLossWeights
from hrothgar.gtok.dataset import GTokAxisDataset, GTokDatasetMaker
from hrothgar.gtok.health import GtokHealthCheck, HealthCheckConfig
from hrothgar.gtok.llamagen_lpips import LPIPS
from hrothgar.gtok.model import GtokModel
from hrothgar.gtok.vgg_loss import VGG
from hrothgar.utils import TrainingLoop


def _parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated integer list from CLI input."""
    items = [part.strip() for part in value.split(",") if part.strip()]
    if not items:
        raise ValueError("Expected a comma-separated list of integers")
    return [int(item) for item in items]


def _read_targeted_validation_families(path: Optional[str]) -> Set[str]:
    if not path:
        return set()
    families: Set[str] = set()
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            family = line.strip()
            if not family or family.startswith("#"):
                continue
            families.add(family)
    return families


class GtokTrainingLoop(TrainingLoop):
    def post_init(self, train_args):
        gtok_config_kwargs = {
            "image_size": train_args.image_size,
        }
        config = GtokConfig(**gtok_config_kwargs)
        config.save_sidecar(train_args.model_path)

        model = GtokModel(config).to(self.device)
        self.loss_weights = GtokLossWeights()
        # Batch size 16 / LR 1e-4 / AdamW are specified in paper, don't mess with them.
        maker = GTokDatasetMaker(
            train_args.dataset_path,
            batch_size=16,
            image_size=config.image_size,
            class_balanced=True,
            max_display_score=train_args.max_display_score,
            gtok_config=config,
            render_time_augmentation=True,
        )
        self._maker = maker
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Register font classes from the dataset for the font classifier head.
        font_classes = sorted({f.classification() for f in maker.googlefonts.fonts})
        model.register_font_classes(font_classes)

        self.train_loader = maker.train_loader()
        self.test_loader = maker.test_loader()
        self.targeted_test_loader = None
        self.targeted_validation_families = _read_targeted_validation_families(
            train_args.targeted_validation_families_file
        )
        if self.targeted_validation_families:
            gf = GoogleFonts(train_args.dataset_path)
            filtered_fonts = [
                font
                for font in gf.fonts
                if getattr(font, "family", None) in self.targeted_validation_families
            ]
            if filtered_fonts:
                dataset = GTokAxisDataset(
                    filtered_fonts,
                    codepoint_filter_fn=maker.test_codepoint_filter,
                    axis_splits=maker.axis_splits,
                    max_axis_positions_per_font=maker.max_axis_positions_per_font,
                )
                if len(dataset) > 0:
                    self.targeted_test_loader = DataLoader(
                        dataset,
                        batch_size=16,
                        shuffle=True,
                        drop_last=False,
                        collate_fn=maker.collate_fn,
                        num_workers=12,
                        pin_memory=True,
                    )
                    print(
                        "Enabled targeted validation for "
                        f"{len(filtered_fonts)} families from "
                        f"{train_args.targeted_validation_families_file}"
                    )
                else:
                    print(
                        "Targeted validation families matched fonts but produced "
                        "no validation samples after filtering; skipping targeted validation."
                    )
            else:
                print(
                    "Targeted validation families file did not match any test families; "
                    "skipping targeted validation."
                )
        self.perceptual_loss_fn = VGG().to(self.device)
        self.glyphloss_fn = GlyphReconstructionLoss(lambda_pixel=0.0).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips = LPIPS().to(self.device)
        self.model = model
        self.target_steps = 1_000_000  # 500k specified in paper for 64
        if train_args.canary != 0:
            # Run for ten epochs, i.e. 10 * len(train) steps
            train_loader = list(itertools.islice(self.train_loader, train_args.canary))
            self.target_steps = 10 * len(train_loader)

        self.num_epochs = (self.target_steps // len(self.train_loader)) + 1
        self.validation_direction = "higher"  # We want to maximize SSIM

        # Health checks — disabled in canary mode to keep it fast.
        if train_args.canary == 0:
            self.health_check = GtokHealthCheck(
                HealthCheckConfig(
                    dataset_path=train_args.dataset_path,
                )
            )
        else:
            self.health_check = None

    def train_step(self, batch):
        gt_images = batch["rendering"].to(self.device)
        codepoints = batch["char"].to(self.device)
        font_labels = None
        if self.model._font_class_map:
            fm = self.model._font_class_map
            font_labels = torch.tensor(
                [fm.get(c, 0) for c in batch["classification"]],
                device=self.device,
                dtype=torch.long,
            )
        recon_images, vq_loss_info = self.model(
            gt_images, codepoints=codepoints, font_labels=font_labels
        )
        loss, loss_info = compute_gtok_loss(
            recon_images,
            gt_images,
            vq_loss_info,
            perceptual_loss_fn=self.perceptual_loss_fn,
            glyphloss_fn=self.glyphloss_fn,
            weights=self.loss_weights,
        )
        return loss, loss_info

    def _run_validation_pass(
        self,
        loader,
        *,
        metric_prefix: str,
        recon_image_tag: str,
        max_batches: int = 100,
    ) -> Optional[torch.Tensor]:
        val_metrics: Dict[str, list[torch.Tensor]] = {"ssim": [], "lpips": []}
        bucket_metrics: Dict[str, Dict[str, list[torch.Tensor]]] = {}

        for val_batch in tqdm.tqdm(
            itertools.islice(loader, max_batches),
            desc=metric_prefix,
            total=max_batches,
        ):
            val_gt_images = val_batch["rendering"].to(self.device)
            val_recon_images, _ = self.model(
                val_gt_images,
            )
            classifications = val_batch.get(
                "classification", ["UNKNOWN"] * val_gt_images.shape[0]
            )

            for idx, cls in enumerate(classifications):
                cls_name = cls or "UNKNOWN"
                ssim_val = self.ssim(
                    val_recon_images[idx : idx + 1], val_gt_images[idx : idx + 1]
                )
                lpips_val = self.lpips(
                    val_recon_images[idx : idx + 1], val_gt_images[idx : idx + 1]
                )
                val_metrics["ssim"].append(ssim_val)
                val_metrics["lpips"].append(lpips_val)
                if cls_name not in bucket_metrics:
                    bucket_metrics[cls_name] = {"ssim": [], "lpips": []}
                bucket_metrics[cls_name]["ssim"].append(ssim_val)
                bucket_metrics[cls_name]["lpips"].append(lpips_val)

        if not val_metrics["ssim"]:
            return None

        avg_ssim = torch.mean(torch.stack(val_metrics["ssim"]))
        avg_lpips = torch.mean(torch.stack(val_metrics["lpips"]))
        self.write_scalar(f"{metric_prefix}/SSIM", avg_ssim)
        self.write_scalar(f"{metric_prefix}/LPIPS", avg_lpips)

        bucket_ssim_scalars: Dict[str, float] = {}
        bucket_lpips_scalars: Dict[str, float] = {}
        for cls_name, cls_metrics in bucket_metrics.items():
            # Keep labels flat in TensorBoard for easy cross-class comparison.
            label = cls_name.replace("/", "_")
            bucket_ssim_scalars[label] = float(
                torch.mean(torch.stack(cls_metrics["ssim"])).detach().cpu()
            )
            bucket_lpips_scalars[label] = float(
                torch.mean(torch.stack(cls_metrics["lpips"])).detach().cpu()
            )

        if bucket_ssim_scalars:
            self.writer.add_scalars(
                f"{metric_prefix}/Buckets/SSIM",
                bucket_ssim_scalars,
                self.global_step,
            )
        if bucket_lpips_scalars:
            self.writer.add_scalars(
                f"{metric_prefix}/Buckets/LPIPS",
                bucket_lpips_scalars,
                self.global_step,
            )

        self.visualize(loader=loader, image_tag=recon_image_tag)
        return avg_ssim

    def post_train_step(self):
        # Log encoder/codebook gradient norms if due.
        if (
            self.health_check is not None
            and self.global_step % self.health_check.config.grad_norm_every == 0
        ):
            GtokHealthCheck.log_gradient_norms(
                self.model, self.writer, self.global_step
            )

        # Do some validation every 1000 steps
        if self.global_step % 1000 != 0:
            return
        self.model.eval()
        with torch.no_grad():
            avg_ssim = self._run_validation_pass(
                self.test_loader,
                metric_prefix="Validation",
                recon_image_tag="Reconstruction/GT_vs_Recon",
            )
            if avg_ssim is not None:
                self.checkpoint_if_best(avg_ssim)
            if self.global_step % 10_000 == 0:
                self.model.save(
                    self.model_path.replace(".pth", f"_step{self.global_step}.pth")
                )
            if self.targeted_test_loader is not None:
                self._run_validation_pass(
                    self.targeted_test_loader,
                    metric_prefix="Validation/Targeted",
                    recon_image_tag="Reconstruction/Targeted_GT_vs_Recon",
                )
        # Health checks (linear probing, autocorrelation, oracle AR).
        # These run at their own configured intervals and may switch the
        # model between train/eval modes internally.
        if self.health_check is not None:
            self.health_check.maybe_run(
                gtok=self.model,
                image_size=self.model.config.image_size,
                global_step=self.global_step,
                writer=self.writer,
            )
        self.model.train()

    def visualize(self, loader, image_tag: str):
        # Also display some pretty pictures
        val_batch = next(iter(loader))
        val_gt_images = val_batch["rendering"].to(self.device)
        val_recon_images, _ = self.model(val_gt_images)
        # Log a grid of reconstructed vs. target images
        recon_grid = torch.cat([val_gt_images[:16], val_recon_images[:16]], dim=0)
        self.writer.add_image(
            image_tag,
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
        default="models/gtok.pth",
    )
    parser.add_argument(
        "--targeted-validation-families-file",
        type=str,
        default=None,
        help=(
            "Optional path to a newline-delimited text file of font family names. "
            "When provided, logs additional targeted validation metrics and "
            "reconstruction previews using only these families."
        ),
    )
    parser.add_argument(
        "--max-display-score",
        type=int,
        default=50,
        help=(
            "Exclude fonts with display_score() above this threshold. "
            "Display fonts have extreme stylistic variation that a shared "
            "codebook struggles to represent.  Set to 0 to disable."
        ),
    )
    args = parser.parse_args()
    if not args.dataset_path:
        raise ValueError(
            "GOOGLE_FONTS_REPO environment variable not set, cannot run training"
        )
    loop = GtokTrainingLoop(args)
    loop.train()
