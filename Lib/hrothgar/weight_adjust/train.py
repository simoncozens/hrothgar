"""Training script for the weight-adjustment model (deterministic CNN)."""

from __future__ import annotations

import itertools
import os

import torch
import torch.nn.functional as F
import torchvision
import tqdm

from glyphloss import glyph_reconstruction_loss
from hrothgar.dataset_constants import LATIN_CORE, LATIN_KERNEL, LGC_ALL
from hrothgar.utils import TrainingLoop
from hrothgar.weight_adjust.config import WeightAdjustConfig
from hrothgar.weight_adjust.dataset import WeightAdjustDatasetMaker
from hrothgar.weight_adjust.model import WeightAdjustModel

CHARACTER_SETS = {
    "kernel": LATIN_KERNEL,
    "core": LATIN_CORE,
    "lgc": LGC_ALL,
}


class WeightAdjustTrainingLoop(TrainingLoop):
    """Training loop for the weight-adjustment model."""

    def post_init(self, train_args):
        self.maker = WeightAdjustDatasetMaker(
            train_args.dataset_path,
            train_args.batch_size,
            image_size=train_args.image_size,
            num_exemplars=train_args.num_exemplars,
            regular_weight=train_args.regular_weight,
            bold_weight=train_args.bold_weight,
            canary_size=train_args.canary_size,
            samples_per_font=train_args.samples_per_font,
            val_fraction=train_args.val_fraction,
            character_set=CHARACTER_SETS[train_args.character_set],
            val_codepoint_fraction=train_args.val_codepoint_fraction,
            weight_min=train_args.weight_min,
            weight_max=train_args.weight_max,
            include_static=not train_args.no_static,
            split_seed=train_args.split_seed,
        )

        self.config = WeightAdjustConfig(
            image_size=train_args.image_size,
            num_exemplars=train_args.num_exemplars,
            style_dim=train_args.style_dim,
            style_base_channels=train_args.style_base_channels,
            decoder_base_channels=train_args.decoder_base_channels,
            decoder_num_blocks=train_args.decoder_num_blocks,
            dropout=train_args.dropout,
            regular_weight=train_args.regular_weight,
            bold_weight=train_args.bold_weight,
            learning_rate=train_args.learning_rate,
            weight_decay=train_args.weight_decay,
            advance_loss_weight=train_args.advance_loss_weight,
        )
        self.config.save_sidecar(train_args.model_path)

        self.model = WeightAdjustModel(self.config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_args.learning_rate,
            weight_decay=train_args.weight_decay,
        )

        self.train_loader = self.maker.train_loader()
        self.test_loader = self.maker.val_loader()

        self.target_steps = train_args.target_steps
        self.validation_every = train_args.validation_every
        self.validation_batches = train_args.validation_batches
        self.num_epochs = (train_args.target_steps // max(len(self.train_loader), 1)) + 1
        self.validation_direction = "lower"

    def train_step(self, batch):
        regular = batch["regular"].to(self.device)
        bold = batch["bold"].to(self.device)
        exemplars = batch["exemplars"].to(self.device)
        regular_advance = batch["regular_advance"].to(self.device)
        advance_delta = batch["advance_delta"].to(self.device)
        weight = batch["weight"].to(self.device)

        bold_pred, delta_pred = self.model(regular, exemplars, regular_advance, weight)
        glyph_loss = glyph_reconstruction_loss(bold_pred, bold)
        advance_loss = F.mse_loss(delta_pred, advance_delta)
        loss = glyph_loss + self.config.advance_loss_weight * advance_loss
        return loss, {
            "loss": loss.detach(),
            "glyphloss": glyph_loss.detach(),
            "advance": advance_loss.detach(),
        }

    def post_train_step(self):
        if self.global_step % self.validation_every != 0:
            return

        self.model.eval()
        with torch.no_grad():
            glyphlosses, advance_errors = [], []
            for batch in tqdm.tqdm(
                itertools.islice(self.test_loader, self.validation_batches),
                desc="Validation",
                total=self.validation_batches,
            ):
                regular = batch["regular"].to(self.device)
                bold = batch["bold"].to(self.device)
                exemplars = batch["exemplars"].to(self.device)
                regular_advance = batch["regular_advance"].to(self.device)
                advance_delta = batch["advance_delta"].to(self.device)
                weight = batch["weight"].to(self.device)

                bold_pred, delta_pred = self.model(
                    regular, exemplars, regular_advance, weight
                )
                glyphlosses.append(glyph_reconstruction_loss(bold_pred, bold))
                advance_errors.append(F.mse_loss(delta_pred, advance_delta))

            val_glyphloss = torch.stack(glyphlosses).mean()
            val_advance = torch.stack(advance_errors).mean()
            self.write_scalar("Validation/glyphloss", val_glyphloss)
            self.write_scalar("Validation/advance_mse", val_advance)
            self.checkpoint_if_best(val_glyphloss)
            self.visualize()

        self.model.train()

    def visualize(self):
        batch = self.maker.random_val_batch(8)
        regular = batch["regular"].to(self.device)
        bold = batch["bold"].to(self.device)
        exemplars = batch["exemplars"].to(self.device)
        regular_advance = batch["regular_advance"].to(self.device)
        weight = batch["weight"].to(self.device)

        bold_pred, _ = self.model(regular, exemplars, regular_advance, weight)
        n = min(8, regular.shape[0])
        grid = torch.cat([regular[:n], bold_pred[:n], bold[:n]], dim=0)
        self.writer.add_image(
            "WeightAdjust/Regular_Pred_Bold",
            torchvision.utils.make_grid(grid, nrow=n),
            self.global_step,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train weight-adjustment model")
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
        help="Allow training with uncommitted changes in the git repository",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-exemplars", type=int, default=5)
    parser.add_argument("--style-dim", type=int, default=128)
    parser.add_argument("--style-base-channels", type=int, default=32)
    parser.add_argument("--decoder-base-channels", type=int, default=64)
    parser.add_argument("--decoder-num-blocks", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--regular-weight", type=int, default=400)
    parser.add_argument("--bold-weight", type=int, default=700)
    parser.add_argument(
        "--weight-min", type=int, default=500,
        help="Lower bound of the jittered target weight during training",
    )
    parser.add_argument(
        "--weight-max", type=int, default=800,
        help="Upper bound of the jittered target weight during training",
    )
    parser.add_argument(
        "--no-static",
        action="store_true",
        help="Exclude static-font regular/bold pairs (variable fonts only)",
    )
    parser.add_argument("--advance-loss-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--target-steps", type=int, default=200_000)
    parser.add_argument("--validation-every", type=int, default=1000)
    parser.add_argument("--validation-batches", type=int, default=50)
    parser.add_argument("--samples-per-font", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--character-set",
        type=str,
        default="kernel",
        choices=sorted(CHARACTER_SETS),
        help="Glyphset to draw targets from; a random subset is held out for validation",
    )
    parser.add_argument(
        "--val-codepoint-fraction",
        type=float,
        default=0.25,
        help="Fraction of the character set held out of training as validation targets",
    )
    parser.add_argument(
        "--canary-size",
        type=int,
        default=None,
        help="Limit to the first N fonts scanned (fast canary mode)",
    )
    parser.add_argument("--split-seed", type=int, default=1234)
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/weight_adjust.pth",
        help="Path to save model weights",
    )
    args = parser.parse_args()
    if not args.dataset_path:
        raise ValueError(
            "GOOGLE_FONTS_REPO environment variable not set, cannot run training"
        )

    loop = WeightAdjustTrainingLoop(args)
    loop.train()
