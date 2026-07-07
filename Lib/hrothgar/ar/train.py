"""MaskGIT glyph generation training loop.

Policy (hyperparameters, generation mode) is stored here in module
constants rather than on the CLI.  Only truly variable parameters
(datasets, paths, scheduling) remain as switches.  The git commit
hash at train time uniquely identifies which values were used.
"""

from __future__ import annotations

import itertools
import os
from contextlib import nullcontext
from pathlib import Path
from typing import List

import torch
import torchvision
import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure

from hrothgar.ar.dataset import ARPhase1DatasetMaker
from hrothgar.ar.losses import ARLossWeights, compute_ar_loss
from hrothgar.ar.model import ARModel, ARModelConfig
from hrothgar.gtok.llamagen_lpips import LPIPS
from hrothgar.gtok.model import load_model as load_gtok_model
from hrothgar.utils import TrainingLoop

# ── Policy constants (not CLI switches — tracked by git hash) ────────────
LEARNING_RATE: float = 1e-4
ADAM_BETA1: float = 0.9
ADAM_BETA2: float = 0.95
MAX_GRAD_NORM: float = 1.0
MASKGIT_NUM_INFERENCE_STEPS: int = 8
MASKGIT_TEMPERATURE: float = 1.0
# ──────────────────────────────────────────────────────────────────────────


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def _parse_codepoint(value: str) -> List[int]:
    return [ord(c) for c in value]


class MaskGITTrainingLoop(TrainingLoop):
    """MaskGIT glyph generation training loop (visual-pretraining stage)."""

    def post_init(self, train_args) -> None:
        # ── Infer image size from the G-Tok checkpoint ────────────────
        gtok, gtok_config = load_gtok_model(
            Path(train_args.gtok_model_path),
            device=self.device,
        )
        image_size = gtok_config.image_size

        config = ARModelConfig(
            image_size=image_size,
            use_maskgit=True,
            maskgit_num_inference_steps=MASKGIT_NUM_INFERENCE_STEPS,
            maskgit_temperature=MASKGIT_TEMPERATURE,
        )
        model = ARModel(config, gtok_model=gtok).to(self.device)

        # ── Data ──────────────────────────────────────────────────────
        if train_args.style_glyph_count < len(train_args.style_characters or []):
            train_args.style_glyph_count = len(train_args.style_characters or [])

        maker = ARPhase1DatasetMaker(
            train_args.dataset_path,
            batch_size=train_args.batch_size,
            image_size=config.image_size,
            style_glyph_count=train_args.style_glyph_count,
            common_style_codepoints=train_args.style_characters,
            target_codepoints=train_args.target_characters,
            target_codepoint_oversample_factor=(
                train_args.target_character_oversample_factor
            ),
            class_balanced=True,  # always on
            split_seed=train_args.split_seed,
            canary_size=train_args.limit_dataset_size,
            target_only=train_args.target_only,
        )

        # ── Optimiser ─────────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            betas=(ADAM_BETA1, ADAM_BETA2),
        )
        self.train_loader = maker.train_loader()
        self.test_loader = maker.test_loader()

        # ── Loss & metrics ────────────────────────────────────────────
        self.loss_weights = ARLossWeights()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips = LPIPS().to(self.device)

        # ── Training bookkeeping ──────────────────────────────────────
        self.model = model
        self.target_steps = train_args.target_steps
        self.validation_every = train_args.validation_every
        self.validation_batches = train_args.validation_batches
        self.grad_accum_steps = getattr(train_args, "grad_accum_steps", 1)
        self.canary_batches = train_args.canary

        if self.grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be positive")

        # ── AMP ───────────────────────────────────────────────────────
        self.use_amp = train_args.precision in {"bf16", "fp16"}
        if train_args.precision == "bf16":
            self.amp_dtype = torch.bfloat16
        elif train_args.precision == "fp16":
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = None

        if self.use_amp and self.device.type != "cuda":
            raise ValueError(
                f"precision={train_args.precision} requires CUDA, "
                f"got device {self.device}"
            )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp and self.amp_dtype == torch.float16,
        )

        # ── Canary mode ───────────────────────────────────────────────
        if self.canary_batches != 0:
            if self.canary_batches > len(self.train_loader):
                raise ValueError(
                    "canary exceeds train loader length "
                    f"({self.canary_batches} > {len(self.train_loader)})"
                )
            self.target_steps = 10 * self.canary_batches

        if self.target_steps is None:
            raise ValueError("target_steps must not be None for MaskGITTrainingLoop")

        self.num_epochs = (self.target_steps // len(self.train_loader)) + 1
        self.validation_direction = "higher"  # Maximise SSIM.

    # ------------------------------------------------------------------
    # AMP helper
    # ------------------------------------------------------------------

    def _autocast_context(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, batch):
        target_images = batch["target_rendering"].to(self.device)
        content_images = batch["content_rendering"].to(self.device)
        style_reference_images = batch["style_renderings"].to(self.device)
        target_codepoints = batch["char"].to(self.device)

        model_output = self.model(
            content_images,
            style_reference_images,
            target_images=target_images,
            target_codepoints=target_codepoints,
            global_step=self.global_step,
        )
        loss, loss_info = compute_ar_loss(
            model_output,
            target_images,
            weights=self.loss_weights,
            lpips_metric=self.lpips,
        )
        loss_info["content_only"] = torch.tensor(
            float(self.model._content_only_step),
            device=target_images.device,
        )
        loss_info["style_only"] = torch.tensor(
            float(self.model._style_only_step),
            device=target_images.device,
        )
        return loss, loss_info

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

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
                self.optimizer.zero_grad(set_to_none=True)

                train_iterable = self.train_loader
                if self.canary_batches != 0:
                    train_iterable = itertools.islice(
                        self.train_loader, self.canary_batches
                    )

                steps_in_epoch = (
                    self.canary_batches
                    if self.canary_batches != 0
                    else len(self.train_loader)
                )

                for i, batch in enumerate(train_iterable):
                    if self.must_stop():
                        break

                    with self._autocast_context():
                        loss, loss_info = self.train_step(batch)
                        scaled_loss = loss / self.grad_accum_steps

                    if self.scaler.is_enabled():
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                    should_step = (i + 1) % self.grad_accum_steps == 0 or (
                        i + 1 == steps_in_epoch
                    )

                    if should_step:
                        if MAX_GRAD_NORM > 0:
                            if self.scaler.is_enabled():
                                self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                max_norm=MAX_GRAD_NORM,
                            )
                        if self.scaler.is_enabled():
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)

                        self.global_step += 1
                        kbar.update(
                            i,
                            values=[
                                (k, float(v.detach().cpu()))
                                for k, v in loss_info.items()
                            ],
                        )
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

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def post_train_step(self):
        if self.global_step % self.validation_every != 0:
            return

        use_maskgit = getattr(self.model.config, "use_maskgit", False)
        if use_maskgit:
            ctx_label = "BidirectionalContext"
            gen_label = "IterativeDecode"
        else:
            ctx_label = "TeacherForced"
            gen_label = "FreeRunning"

        self.model.eval()
        with torch.no_grad():
            # ── Full-context quality ──────────────────────────────────
            val_metrics = {
                "ssim": [],
                "lpips": [],
                "token_accuracy": [],
                "token_cross_entropy": [],
            }
            for val_batch in tqdm.tqdm(
                itertools.islice(self.test_loader, self.validation_batches),
                desc="Validation",
                total=self.validation_batches,
            ):
                val_target = val_batch["target_rendering"].to(self.device)
                val_content = val_batch["content_rendering"].to(self.device)
                val_style = val_batch["style_renderings"].to(self.device)
                val_cp = val_batch["char"].to(self.device)

                with self._autocast_context():
                    val_output = self.model(
                        val_content,
                        val_style,
                        target_images=val_target,
                        target_codepoints=val_cp,
                    )
                _val_loss, val_loss_info = compute_ar_loss(
                    val_output,
                    val_target,
                    weights=self.loss_weights,
                )
                recon = torch.clamp(val_output.reconstructed_images, 0.0, 1.0).float()
                target = torch.clamp(val_target, 0.0, 1.0).float()
                with torch.autocast(device_type=self.device.type, enabled=False):
                    val_metrics["ssim"].append(self.ssim(recon, target))
                    val_metrics["lpips"].append(self.lpips(recon, target))
                val_metrics["token_accuracy"].append(val_loss_info["token_accuracy"])
                val_metrics["token_cross_entropy"].append(
                    val_loss_info["token_cross_entropy"]
                )

            avg_ssim = torch.mean(torch.stack(val_metrics["ssim"]))
            avg_lpips = torch.mean(torch.stack(val_metrics["lpips"]))
            avg_token_acc = torch.mean(torch.stack(val_metrics["token_accuracy"]))
            avg_token_ce = torch.mean(torch.stack(val_metrics["token_cross_entropy"]))
            self.write_scalar(f"Validation/{ctx_label}_SSIM", avg_ssim)
            self.write_scalar(f"Validation/{ctx_label}_LPIPS", avg_lpips)
            self.write_scalar(f"Validation/{ctx_label}_TokenAccuracy", avg_token_acc)
            self.write_scalar(f"Validation/{ctx_label}_TokenCrossEntropy", avg_token_ce)

            # ── Generation quality ────────────────────────────────────
            fr_batches = max(1, self.validation_batches // 10)
            fr_metrics = {"ssim": [], "lpips": [], "token_accuracy": []}
            for val_batch in tqdm.tqdm(
                itertools.islice(self.test_loader, fr_batches),
                desc="Generation validation",
                total=fr_batches,
            ):
                val_target = val_batch["target_rendering"].to(self.device)
                val_content = val_batch["content_rendering"].to(self.device)
                val_style = val_batch["style_renderings"].to(self.device)
                val_cp = val_batch["char"].to(self.device)

                with self._autocast_context():
                    gen_output = self.model.generate(
                        content_images=val_content,
                        style_reference_images=val_style,
                        target_codepoints=val_cp,
                    )
                gt_tokens = self.model.target_token_indices_from_images(
                    val_target,
                )
                gen_recon = torch.clamp(
                    gen_output.reconstructed_images, 0.0, 1.0
                ).float()
                gen_target = torch.clamp(val_target, 0.0, 1.0).float()
                with torch.autocast(device_type=self.device.type, enabled=False):
                    fr_metrics["ssim"].append(self.ssim(gen_recon, gen_target))
                    fr_metrics["lpips"].append(self.lpips(gen_recon, gen_target))
                fr_metrics["token_accuracy"].append(
                    (gen_output.target_token_indices == gt_tokens).float().mean()
                )

            fr_ssim = torch.mean(torch.stack(fr_metrics["ssim"]))
            fr_lpips = torch.mean(torch.stack(fr_metrics["lpips"]))
            fr_token_acc = torch.mean(torch.stack(fr_metrics["token_accuracy"]))
            self.write_scalar(f"Validation/{gen_label}_SSIM", fr_ssim)
            self.write_scalar(f"Validation/{gen_label}_LPIPS", fr_lpips)
            self.write_scalar(f"Validation/{gen_label}_TokenAccuracy", fr_token_acc)
            gap = avg_token_acc - fr_token_acc
            gap_ratio = gap / torch.clamp(avg_token_acc, min=1e-8)
            self.write_scalar("Validation/TokenAccuracy_Gap_Absolute", gap)
            self.write_scalar("Validation/TokenAccuracy_Gap_Relative", gap_ratio)

            self.checkpoint_if_best(fr_ssim)
            self.visualize()

        self.model.train()

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize(self):
        val_batch = next(iter(self.test_loader))
        val_target = val_batch["target_rendering"].to(self.device)
        val_content = val_batch["content_rendering"].to(self.device)
        val_style = val_batch["style_renderings"].to(self.device)
        val_cp = val_batch["char"].to(self.device)

        with self._autocast_context():
            # Full-context reconstruction (teacher-forced / bidirectional).
            val_output = self.model(
                val_content,
                val_style,
                target_images=val_target,
                target_codepoints=val_cp,
            )
            # Generation (free-running / iterative decode).
            gen_output = self.model.generate(
                content_images=val_content,
                style_reference_images=val_style,
                target_codepoints=val_cp,
            )

        preview_count = min(8, val_target.shape[0])
        first_style = val_style[:preview_count, 0]
        recon_grid = torch.cat(
            [
                val_content[:preview_count],
                first_style,
                val_target[:preview_count],
                val_output.reconstructed_images[:preview_count],
                gen_output.reconstructed_images[:preview_count],
            ],
            dim=0,
        )
        self.writer.add_image(
            "Reconstruction/content_style_target_recon_gen",
            torchvision.utils.make_grid(recon_grid, nrow=preview_count),
            self.global_step,
        )


# ══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MaskGIT glyph generator")
    parser.add_argument(
        "--canary",
        type=int,
        default=0,
        help="If nonzero, use this many train batches for a short canary run",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow training with uncommitted changes (not recommended)",
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
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--style-glyph-count",
        type=int,
        default=8,
        help="Number of style glyph references N_s (default: 8)",
    )
    parser.add_argument(
        "--style-characters",
        type=_parse_codepoint,
        help="Optional string of explicit style characters shared across items",
    )
    parser.add_argument(
        "--target-characters",
        type=_parse_codepoint,
        help=(
            "Optional string of extra target characters to add to the "
            "train/test datasets when present in a font.  Oversampled in "
            "the training set rather than restricting the dataset."
        ),
    )
    parser.add_argument(
        "--target-character-oversample-factor",
        type=int,
        default=8,
        help="Oversampling factor for --target-characters (default: 8)",
    )
    parser.add_argument(
        "--target-steps",
        type=int,
        default=600_000,
        help="Training iterations (default: 600k)",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=1234,
        help="Seed for train/test font and character splits",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        default="bf16",
        help="Numerical precision for training",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--validation-every",
        type=int,
        default=1000,
        help="Run validation every N steps",
    )
    parser.add_argument(
        "--validation-batches",
        type=int,
        default=100,
        help="Number of validation batches per pass",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/maskgit_glyph_gen.pth",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--gtok-model-path",
        type=str,
        default="models/gtok.pth",
        help="Path to the trained G-Tok tokenizer",
    )
    parser.add_argument(
        "--limit-dataset-size",
        type=int,
        default=None,
        help="Limit dataset to this many fonts for quick canary runs",
    )
    parser.add_argument(
        "--target-only",
        action="store_true",
        help=(
            "Restrict dataset to only --target-characters (instead of "
            "oversampling them).  Stricter test of few-shot learning."
        ),
    )

    args = parser.parse_args()
    if not args.dataset_path:
        raise ValueError(
            "GOOGLE_FONTS_REPO environment variable not set, cannot run training"
        )

    loop = MaskGITTrainingLoop(args)
    loop.train()
