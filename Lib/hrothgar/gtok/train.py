import datetime
import os
import random
import subprocess

import pkbar
import torch
from torch.utils.tensorboard import SummaryWriter

from hrothgar.dataset import DatasetMaker
from hrothgar.gtok.losses import compute_gtok_loss
from hrothgar.gtok.model import GtokConfig, GtokModel


def torch_setup():
    """Set random seeds and configure torch for reproducibility and performance."""
    random.seed(1234)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.set_float32_matmul_precision("high")


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
    model = GtokModel(GtokConfig())
    maker = DatasetMaker(train_args.dataset_path, batch_size=16)
    train_loader = maker.train_loader()
    # test = maker.test_loader()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

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
        loader_iter = iter(train_loader)
        train_loader = [next(loader_iter)] * train_args.canary
        target_steps = 10 * len(train_loader)
    num_epochs = (target_steps // len(train_loader)) + 1
    while global_step < target_steps:
        model.train()
        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs)
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            gt_images = batch["rendering"]
            recon_images, vq_loss_info = model(gt_images)
            loss, loss_info = compute_gtok_loss(recon_images, gt_images, vq_loss_info)
            loss.backward()
            optimizer.step()
            global_step += 1
            kbar.update(i, values=loss_info.items())
            for key, value in loss_info.items():
                writer.add_scalar(key, value.item(), global_step)
        epoch += 1
        # Do validation here later


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
