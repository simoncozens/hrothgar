import datetime
import itertools
import os
import random
import subprocess
from typing import Dict

import tqdm
import pkbar
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torch.utils.tensorboard import SummaryWriter
import torchvision

from hrothgar.dataset import DatasetMaker
from hrothgar.gtok import compute_gtok_loss
from hrothgar.gtok.model import GtokConfig, GtokModel
from hrothgar.gtok.vgg_loss import VGG
from hrothgar.gtok.llamagen_lpips import LPIPS


def torch_setup() -> torch.device:
    """Set random seeds and configure torch for reproducibility and performance."""
    random.seed(1234)
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.set_float32_matmul_precision("high")
    return device


def _progress_values(loss_info: Dict[str, torch.Tensor]):
    """Convert scalar loss tensors to pkbar's expected ``[(name, value), ...]`` format."""
    return [(key, float(value.detach().cpu())) for key, value in loss_info.items()]


def check_git_clean_and_get_commit_hash() -> str:
    """Check that the git repository is clean and return the current commit hash."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        if result.stdout.strip():
            raise RuntimeError(
                "Git repository has uncommitted changes. Please commit or stash them before training."
            )
        commit_hash_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return commit_hash_result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Git command failed: {e.stderr.strip()}") from e


def train(train_args):
    # Batch size 16 / LR 1e-4 / AdamW are specified in paper, don't mess with them.
    device = torch_setup()
    config = GtokConfig(image_size=train_args.image_size)
    model = GtokModel(config).to(device)
    maker = DatasetMaker(
        train_args.dataset_path, batch_size=16, image_size=config.image_size
    )
    train_loader = maker.train_loader()
    test_loader = maker.test_loader()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    perceptual_loss_fn = VGG().to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LPIPS().to(device)
    model.load(train_args.model_path)

    git_tag = check_git_clean_and_get_commit_hash()
    run_id = f"logs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{git_tag}"
    if train_args.tag:
        run_id += f"-{train_args.tag}"

    writer = SummaryWriter(log_dir=run_id)

    epoch = 0
    global_step = 0
    best_ssim = 0
    target_steps = 200_000  # Specified in paper
    if train_args.canary != 0:
        # Run for ten epochs, i.e. 10 * len(train) steps
        train_loader = list(itertools.islice(train_loader, train_args.canary))
        target_steps = 10 * len(train_loader)
    num_epochs = (target_steps // len(train_loader)) + 1
    if len(train_loader) == 0:
        raise ValueError("Training loader is empty; cannot start training.")
    try:
        while global_step < target_steps:
            model.train()
            kbar = pkbar.Kbar(
                target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=8
            )
            for i, batch in enumerate(train_loader):
                if global_step >= target_steps:
                    break

                optimizer.zero_grad(set_to_none=True)
                gt_images = batch["rendering"].to(device)
                recon_images, vq_loss_info = model(gt_images)
                loss, loss_info = compute_gtok_loss(
                    recon_images,
                    gt_images,
                    vq_loss_info,
                    perceptual_loss_fn=perceptual_loss_fn,
                )
                loss.backward()
                optimizer.step()

                global_step += 1
                kbar.update(i, values=_progress_values(loss_info))
                for key, value in loss_info.items():
                    writer.add_scalar(
                        "Losses/" + key, float(value.detach().cpu()), global_step
                    )
                writer.flush()
                # Do some validation every 1000 steps
                if global_step % 1000 != 0:
                    continue
                model.eval()
                with torch.no_grad():
                    # Compute SSIM and LPIPS on the validation set and log them to TensorBoard
                    val_metrics = {"ssim": [], "lpips": []}
                    # Just do 100 batches
                    for val_batch in tqdm.tqdm(
                        itertools.islice(test_loader, 100),
                        desc="Validation",
                        total=100,
                    ):
                        val_gt_images = val_batch["rendering"].to(device)
                        val_recon_images, _ = model(val_gt_images)
                        val_metrics["ssim"].append(
                            ssim(val_recon_images, val_gt_images)
                        )
                        val_metrics["lpips"].append(
                            lpips(val_recon_images, val_gt_images)
                        )
                    avg_ssim = torch.mean(torch.stack(val_metrics["ssim"]))
                    avg_lpips = torch.mean(torch.stack(val_metrics["lpips"]))
                    writer.add_scalar("Validation/SSIM", avg_ssim.item(), global_step)
                    writer.add_scalar("Validation/LPIPS", avg_lpips.item(), global_step)
                    if avg_ssim > best_ssim:
                        best_ssim = avg_ssim
                        model.save(train_args.model_path)

                    # Also display some pretty pictures
                    val_batch = next(iter(test_loader))
                    val_gt_images = val_batch["rendering"].to(device)
                    val_recon_images, _ = model(val_gt_images)
                    # Log a grid of reconstructed vs. target images
                    recon_grid = torch.cat(
                        [val_gt_images[:16], val_recon_images[:16]], dim=0
                    )
                    writer.add_image(
                        "Reconstruction/GT_vs_Recon",
                        torchvision.utils.make_grid(recon_grid, nrow=16),
                        global_step,
                    )
                model.train()
            epoch += 1
    finally:
        writer.close()


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
    args = parser.parse_args()
    if not args.dataset_path:
        raise ValueError(
            "GOOGLE_FONTS_REPO environment variable not set, cannot run training"
        )
    train(args)
