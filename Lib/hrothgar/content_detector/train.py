"""Training loop for the content detector."""

from __future__ import annotations

import argparse

import torch

from hrothgar.content_detector.config import ContentDetectorConfig
from hrothgar.content_detector.dataset import ContentDatasetMaker
from hrothgar.content_detector.evaluation import build_content_panel, content_report
from hrothgar.content_detector.losses import (
    height_loss,
    margin_hinges,
    ovr_bce_loss,
    quality_margin_hinge,
)
from hrothgar.content_detector.model import ContentDetector
from hrothgar.fontquant import load_default_or_none
from hrothgar.utils import TrainingLoop, progress_values

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
GRAD_CLIP_NORM = 1.0


class ContentDetectorTrainingLoop(TrainingLoop):
    """Training loop for the content detector."""

    def post_init(self, train_args) -> None:
        config = ContentDetectorConfig(
            glyph_size=getattr(train_args, "glyph_size", 256),
            anchors_per_batch=getattr(train_args, "anchors_per_batch", 4),
            fonts_per_anchor=getattr(train_args, "fonts_per_anchor", 4),
            num_workers=getattr(train_args, "num_workers", 8),
            quality_margin=getattr(train_args, "quality_margin", 0.0),
            generated_manifest=getattr(train_args, "generated_manifest", None),
            generated_per_batch=getattr(train_args, "generated_per_batch", 16),
        )
        self.config = config
        self.model = ContentDetector(config).to(self.device)
        config.save_sidecar(train_args.model_path)
        print(
            f"Trainable parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )

        maker = ContentDatasetMaker(
            train_args.dataset_path,
            config,
            canary_size=train_args.limit_dataset_size,
            font_metadata=load_default_or_none(),
        )
        self.maker = maker
        self.train_loader = maker.train_loader()
        self.test_loader = maker.test_loader()
        print(
            f"Train batches: {len(self.train_loader)}, "
            f"Test batches: {len(self.test_loader)}"
        )

        self.panel = build_content_panel(
            maker.test_fonts,
            config,
            max_fonts=getattr(train_args, "panel_fonts", config.panel_fonts),
        )
        print(f"Content eval panel: {len(self.panel.codepoints)} glyphs")

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=(ADAM_BETA1, ADAM_BETA2),
            weight_decay=getattr(train_args, "weight_decay", config.weight_decay),
        )

        self.target_steps = train_args.target_steps
        self.num_epochs = getattr(train_args, "num_epochs", 100)
        self.validation_direction = "higher"
        self.validation_every = train_args.validation_every

    def train(self):
        if len(self.train_loader) == 0:
            raise ValueError("Training loader is empty; cannot start training.")
        import pkbar

        try:
            while not self.must_stop():
                self.maker.set_epoch(self.epoch)
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
        targets = batch["targets"].to(self.device)
        broken = batch["broken"].to(self.device)
        bce_mask = batch["bce_mask"].to(self.device)
        height_target = batch["height"].to(self.device)
        height_mask = batch["height_mask"].to(self.device)

        logits, cosine, predicted_height = self.model.forward_with_aux(images)
        bce = ovr_bce_loss(logits, targets, mask=bce_mask)
        ranking, positive = margin_hinges(
            cosine,
            targets,
            ranking_margin=self.config.ranking_margin,
            positive_margin=self.config.positive_margin,
        )
        quality = quality_margin_hinge(
            cosine,
            targets,
            broken,
            quality_margin=self.config.quality_margin,
        )
        aux = height_loss(predicted_height, height_target, height_mask)
        loss = bce + ranking + positive + quality + self.config.height_head_weight * aux
        return loss, {
            "loss": loss.detach(),
            "bce": bce.detach(),
            "rank": ranking.detach(),
            "pos": positive.detach(),
            "quality": quality.detach(),
            "height": aux.detach(),
        }

    def post_train_step(self):
        # The learned logit scale and anchor biases are the operating point; log
        # them often enough to see them drift (they are cheap scalars).
        if self.global_step % 100 == 0:
            self._log_model_parameters()

        if self.global_step % self.validation_every != 0:
            return
        if not len(self.panel.codepoints):
            return

        self.model.eval()
        report = content_report(self.model, self.panel, self.config, self.device)
        for name in (
            "auc_hard",
            "auc_any",
            "positive_recall",
            "positive_recall_macro",
            "recall_at_far",
            "offsample_reject_rate_hard",
            "offsample_reject_rate_easy",
            "anchor_margin_cosine",
            "anchor_rank_margin_cosine",
        ):
            self.write_scalar(f"Validation/{name}", torch.tensor(report[name]))
        self.checkpoint_if_best(torch.tensor(report["auc_hard"]))
        self.model.train()

    def _log_model_parameters(self) -> None:
        """Log the learned logit scale and anchor bias spread to TensorBoard."""
        self.write_scalar("Model/logit_scale", self.model.logit_scale.detach())
        bias = self.model.anchor_bias.detach()
        self.write_scalar("Model/anchor_bias_min", bias.min())
        self.write_scalar("Model/anchor_bias_mean", bias.mean())
        self.write_scalar("Model/anchor_bias_max", bias.max())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the content detector")
    p.add_argument("--dataset-path", type=str, required=True)
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--target-steps", type=int, default=None)
    p.add_argument("--num-epochs", type=int, default=100)
    p.add_argument("--validation-every", type=int, default=500)
    p.add_argument("--glyph-size", type=int, default=256)
    p.add_argument("--anchors-per-batch", type=int, default=4)
    p.add_argument("--fonts-per-anchor", type=int, default=4)
    p.add_argument(
        "--panel-fonts",
        type=int,
        default=100,
        help="Held-out fonts in the validation panel (0 disables validation).",
    )
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument(
        "--quality-margin",
        type=float,
        default=0.0,
        help="Within-anchor quality margin: correct glyphs must beat broken "
        "(bad-skeleton) glyphs by this much in cosine space (0 disables)",
    )
    p.add_argument(
        "--generated-manifest",
        type=str,
        default=None,
        help="build_critic_dataset.py manifest; routes acceptable/ambiguous/"
        "bad-skeleton generated glyphs into training (off if unset)",
    )
    p.add_argument(
        "--generated-per-batch",
        type=int,
        default=16,
        help="Generated glyphs sampled into each training batch",
    )
    p.add_argument("--limit-dataset-size", type=int, default=None, help="canary mode")
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--allow-dirty", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    loop = ContentDetectorTrainingLoop(args)
    loop.train()


if __name__ == "__main__":
    main()
