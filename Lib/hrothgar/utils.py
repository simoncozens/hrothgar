import datetime
import random
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


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


def progress_values(loss_info: Dict[str, torch.Tensor]):
    """Convert scalar loss tensors to pkbar's expected ``[(name, value), ...]`` format."""
    return [(key, float(value.detach().cpu())) for key, value in loss_info.items()]


def check_git_clean_and_get_commit_hash(train_args) -> str:
    """Check that the git repository is clean and return the current commit hash."""
    try:
        allow_dirty = bool(getattr(train_args, "allow_dirty", False))
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        if result.stdout.strip() and not allow_dirty:
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


class SaveLoadModel(torch.nn.Module):
    """A simple interface for models that can be saved and loaded from disk."""

    def save(self, path: str):
        """Save the model to the given path."""
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: torch.device):
        """Load the model from the given path."""
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)


class TrainingLoop:
    """A little mini Keras"""

    target_steps: Optional[int] = None  # Set in post_init based on canary mode
    num_epochs: int
    model_path: str
    validation_metric: float
    validation_direction: str  # Set in post_init
    model: SaveLoadModel  # Set in post_init
    optimizer: torch.optim.Optimizer  # Set in post_init
    train_loader: torch.utils.data.DataLoader  # Set in post_init
    test_loader: torch.utils.data.DataLoader  # Set in post_init

    def __init__(self, train_args):
        from torch.utils.tensorboard import SummaryWriter

        self.device = torch_setup()
        git_tag = check_git_clean_and_get_commit_hash(train_args)
        run_id = f"logs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{git_tag}"
        if train_args.tag:
            run_id += f"-{train_args.tag}"

        self.writer = SummaryWriter(log_dir=run_id)
        self.epoch = 0
        self.global_step = 0
        self.model_path = train_args.model_path
        self.post_init(train_args)
        if self.validation_direction == "higher":
            self.validation_metric = float("-inf")
        else:
            self.validation_metric = float("inf")
        if Path(self.model_path).exists():
            print(
                f"Model file {self.model_path} already exists, loading it before training."
            )
            self.model.load(self.model_path, device=self.device)

    def must_stop(self):
        if self.target_steps is None:
            return False
        return self.global_step >= self.target_steps

    def checkpoint_if_best(self, validation_returned: torch.Tensor):
        is_best = False
        if self.validation_direction == "higher":
            is_best = validation_returned > self.validation_metric
        else:
            is_best = validation_returned < self.validation_metric
        if is_best:
            self.validation_metric = validation_returned.item()
            self.model.save(self.model_path)

    def post_train_step(self):
        pass

    def post_train_epoch(self):
        pass

    def validation(self):
        pass

    def train_step(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Perform a single training step on a batch of data. Must be implemented by subclass.

        Returns:
            loss: The scalar loss tensor to backpropagate.
            loss_info: A dictionary of individual loss terms for logging.
        """
        raise NotImplementedError("train_step must be implemented by subclass")

    def post_init(self, train_args):
        """Set up the model, data loaders, optimizer, and other training components. This is separated from __init__ so that it can be overridden in canary mode to use a smaller dataset and fewer steps."""
        raise NotImplementedError("post_init must be implemented by subclass")

    def train(self):
        if len(self.train_loader) == 0:
            raise ValueError("Training loader is empty; cannot start training.")
        import pkbar

        try:
            while not self.must_stop():
                kbar = pkbar.Kbar(
                    target=len(self.train_loader),
                    epoch=self.epoch,
                    num_epochs=self.num_epochs,
                )
                self.model.train()
                for i, batch in enumerate(self.train_loader):
                    if self.must_stop():
                        break
                    self.optimizer.zero_grad(set_to_none=True)
                    loss, loss_info = self.train_step(batch)
                    loss.backward()
                    self.optimizer.step()

                    self.global_step += 1
                    kbar.update(i, values=progress_values(loss_info))
                    for key, value in loss_info.items():
                        self.write_scalar("Losses/" + key, value)
                    if self.global_step % 100 == 0:
                        self.writer.flush()
                    self.post_train_step()
                self.post_train_epoch()
                self.epoch += 1
                self.validation()
        finally:
            self.writer.close()

    def write_scalar(self, name: str, value: torch.Tensor):
        self.writer.add_scalar(
            name, float(value.detach().cpu()), global_step=self.global_step
        )


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
