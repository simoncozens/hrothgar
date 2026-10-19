"""Training loop for the instance-membership model."""

from __future__ import annotations

import argparse

import torch

from hrothgar.set_membership.config import SetMembershipConfig
from hrothgar.set_membership.dataset import SetMembershipDatasetMaker
from hrothgar.set_membership.evaluation import (
    build_membership_panel,
    membership_breakdown,
)
from hrothgar.set_membership.losses import glyph_contrastive_loss
from hrothgar.set_membership.model import SetMembershipModel
from hrothgar.utils import TrainingLoop, progress_values

LEARNING_RATE = 1e-4
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
WEIGHT_DECAY = 1e-3
GRAD_CLIP_NORM = 1.0


class SetMembershipTrainingLoop(TrainingLoop):
    """Training loop for the instance-membership model."""

    def post_init(self, train_args) -> None:
        config = SetMembershipConfig(
            glyph_size=getattr(train_args, "glyph_size", 128),
            codepoints_per_instance=getattr(train_args, "codepoints_per_instance", 8),
            instances_per_family=getattr(train_args, "instances_per_family", 2),
        )
        self.config = config
        self.model = SetMembershipModel(config).to(self.device)
        config.save_sidecar(train_args.model_path)
        print(
            f"Trainable parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )

        maker = SetMembershipDatasetMaker(
            train_args.dataset_path,
            train_args.batch_size,
            config,
            canary_size=train_args.limit_dataset_size,
        )
        self.train_loader = maker.train_loader()
        self.test_loader = maker.test_loader()
        print(
            f"Train batches: {len(self.train_loader)}, "
            f"Test batches: {len(self.test_loader)}"
        )

        # Fixed validation panel (multi-instance families first) for the
        # overall / same-family / same-category membership breakdowns.
        self.eval_panel = None
        val_instances = getattr(train_args, "val_embed_instances", 0)
        if val_instances and maker.test_instances:
            self.eval_panel = build_membership_panel(
                maker.test_instances, config, max_instances=val_instances
            )
            print(f"Membership eval panel: {len(self.eval_panel.families)} instances")

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=LEARNING_RATE,
            betas=(ADAM_BETA1, ADAM_BETA2),
            weight_decay=getattr(train_args, "weight_decay", WEIGHT_DECAY),
        )

        self.target_steps = train_args.target_steps
        self.num_epochs = getattr(train_args, "num_epochs", 100)
        self.validation_direction = "higher"
        self.validation_every = train_args.validation_every
        self.validation_batches = train_args.validation_batches

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
                for _i, batch in enumerate(self.train_loader):
                    if self.must_stop():
                        break
                    self.optimizer.zero_grad(set_to_none=True)
                    loss, loss_info = self.train_step(batch)
                    loss.backward()
                    if GRAD_CLIP_NORM > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), GRAD_CLIP_NORM
                        )
                    self.optimizer.step()

                    self.global_step += 1
                    kbar.update(_i, values=progress_values(loss_info))
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

    def train_step(self, batch):
        images = batch["images"].to(self.device)
        ids = batch["instance_ids"].to(self.device)

        z = self.model(images)
        loss = glyph_contrastive_loss(
            z, ids, temperature=self.model.config.contrastive_temperature
        )
        return loss, {"loss": loss.detach(), "contrastive": loss.detach()}

    def post_train_step(self):
        if self.global_step % self.validation_every != 0:
            return
        if self.eval_panel is None:
            return

        self.model.eval()
        breakdown = membership_breakdown(self.model, self.eval_panel, self.device)
        for scope, metric in breakdown.items():
            for name, value in metric.items():
                self.write_scalar(f"Validation/{scope}/{name}", torch.tensor(value))
        self.checkpoint_if_best(torch.tensor(breakdown["overall"]["r1"]))
        self.model.train()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train instance-membership model")
    p.add_argument("--dataset-path", type=str, required=True)
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--target-steps", type=int, default=None)
    p.add_argument("--num-epochs", type=int, default=100)
    p.add_argument("--validation-every", type=int, default=500)
    p.add_argument("--validation-batches", type=int, default=10)
    p.add_argument("--glyph-size", type=int, default=128)
    p.add_argument("--codepoints-per-instance", type=int, default=8)
    p.add_argument("--instances-per-family", type=int, default=2)
    p.add_argument(
        "--val-embed-instances",
        type=int,
        default=200,
        help="Size of the held-out membership eval panel (0 disables validation).",
    )
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--limit-dataset-size", type=int, default=None, help="canary mode")
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--allow-dirty", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    loop = SetMembershipTrainingLoop(args)
    loop.train()


if __name__ == "__main__":
    main()
