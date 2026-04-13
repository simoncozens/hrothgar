import datetime
import itertools
import os
import random
import subprocess
from typing import Dict

import pkbar
import torch
from torch.utils.tensorboard import SummaryWriter

from hrothgar.dataset import DatasetMaker
from hrothgar.gtok import compute_gtok_loss
from hrothgar.gtok.model import GtokConfig, GtokModel
from hrothgar.gtok.vgg_loss import VGG


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
    model = GtokModel(GtokConfig()).to(device)
    maker = DatasetMaker(train_args.dataset_path, batch_size=16)
    train_loader = maker.train_loader()
    # test = maker.test_loader()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    perceptual_loss_fn = VGG().to(device)

    git_tag = check_git_clean_and_get_commit_hash()
    run_id = f"logs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{git_tag}"
    if train_args.tag:
        run_id += f"-{train_args.tag}"

    writer = SummaryWriter(log_dir=run_id)

    epoch = 0
    global_step = 0
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
                    writer.add_scalar(key, float(value.detach().cpu()), global_step)
            writer.flush()
            epoch += 1
            # Do validation here later
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
    args = parser.parse_args()
    if not args.dataset_path:
        raise ValueError(
            "GOOGLE_FONTS_REPO environment variable not set, cannot run training"
        )
    train(args)
