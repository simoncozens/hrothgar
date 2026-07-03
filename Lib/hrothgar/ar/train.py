"""DiT-based glyph generation training loop."""

from __future__ import annotations

import itertools
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure

from hrothgar.ar.dataset import ARPhase1DatasetMaker
from hrothgar.ar.losses import GlyphGenLossWeights, compute_glyph_gen_loss
from hrothgar.ar.model import GlyphGenConfig, GlyphGenerator
from hrothgar.gtok.model import load_model as load_gtok_model
from hrothgar.upstream.lpips import LPIPS
from hrothgar.utils import TrainingLoop

# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _parse_codepoint(value: str) -> list[int]:
    return [ord(c) for c in value]


def _parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


class DiTGlyphTrainingLoop(TrainingLoop):
    """DiT-based glyph generation training loop.

    Trains the GlyphDiT backbone to denoise G-Tok codebook embeddings,
    conditioned on codepoint identity and visual style features.
    """

    def post_init(self, train_args) -> None:
        config = GlyphGenConfig(
            image_size=train_args.image_size,
            dit_hidden_size=getattr(train_args, "dit_hidden_size", 832),
            dit_depth=getattr(train_args, "dit_depth", 16),
            dit_num_heads=getattr(train_args, "dit_num_heads", 16),
            dit_mlp_ratio=getattr(train_args, "dit_mlp_ratio", 4.0),
            diffusion_steps=getattr(train_args, "diffusion_steps", 1000),
            ddim_steps=getattr(train_args, "ddim_steps", 250),
            cfg_scale=getattr(train_args, "cfg_scale", 1.0),
            gumbel_temperature=getattr(train_args, "gumbel_temperature", 0.5),
        )

        gtok, _gtok_config = load_gtok_model(
            Path(train_args.gtok_model_path),
            device=self.device,
        )
        model = GlyphGenerator(config, gtok_model=gtok).to(self.device)

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
            class_balanced=train_args.class_balanced,
            split_seed=train_args.split_seed,
            canary_size=train_args.limit_dataset_size,
            target_only=train_args.target_only,
        )

        self.optimizer = torch.optim.AdamW(
            model.trainable_parameters(),
            lr=train_args.learning_rate,
            betas=(train_args.beta1, train_args.beta2),
        )
        self.train_loader = maker.train_loader()
        self.test_loader = maker.test_loader()

        self.loss_weights = GlyphGenLossWeights()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips = LPIPS().to(self.device)

        self.model = model
        self.target_steps = train_args.target_steps
        self.validation_every = train_args.validation_every
        self.validation_batches = train_args.validation_batches
        self.grad_accum_steps = getattr(train_args, "grad_accum_steps", 1)
        self.max_grad_norm = getattr(train_args, "max_grad_norm", 1.0)
        self.canary_batches = train_args.canary

        if self.grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be positive")

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

        if self.canary_batches != 0:
            if self.canary_batches > len(self.train_loader):
                raise ValueError(
                    "canary exceeds train loader length "
                    f"({self.canary_batches} > {len(self.train_loader)})"
                )
            self.target_steps = 10 * self.canary_batches

        if self.target_steps is None:
            raise ValueError("target_steps must not be None for DiTGlyphTrainingLoop")

        self.num_epochs = (self.target_steps // len(self.train_loader)) + 1
        self.validation_direction = "higher"  # Maximise SSIM.

    def _autocast_context(self):
        if not self.use_amp:
            from contextlib import nullcontext

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
        )
        loss, loss_info = compute_glyph_gen_loss(
            model_output,
            target_images,
            weights=self.loss_weights,
            lpips_metric=self.lpips,
        )
        return loss, loss_info

    # ------------------------------------------------------------------
    # Training loop (with grad accumulation + AMP)
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
                        if self.max_grad_norm > 0:
                            if self.scaler.is_enabled():
                                self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                max_norm=self.max_grad_norm,
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

        self.model.eval()
        with torch.no_grad():
            # --- Reconstruction quality (predict x₀ from noise) ---------
            val_metrics: dict[str, list[torch.Tensor]] = {
                "ssim": [],
                "lpips": [],
                "noise_mse": [],
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
                _loss, val_info = compute_glyph_gen_loss(
                    val_output,
                    val_target,
                    weights=self.loss_weights,
                    lpips_metric=self.lpips,
                )
                recon = torch.clamp(val_output.reconstructed_images, 0.0, 1.0).float()
                target = torch.clamp(val_target, 0.0, 1.0).float()
                with torch.autocast(device_type=self.device.type, enabled=False):
                    val_metrics["ssim"].append(self.ssim(recon, target))
                    val_metrics["lpips"].append(self.lpips(recon, target))
                val_metrics["noise_mse"].append(val_info["noise_mse"])

            avg_ssim = torch.mean(torch.stack(val_metrics["ssim"]))
            avg_lpips = torch.mean(torch.stack(val_metrics["lpips"]))
            avg_noise_mse = torch.mean(torch.stack(val_metrics["noise_mse"]))
            self.write_scalar("Validation/Reconstruction_SSIM", avg_ssim)
            self.write_scalar("Validation/Reconstruction_LPIPS", avg_lpips)
            self.write_scalar("Validation/Noise_MSE", avg_noise_mse)

            # --- Generation quality (DDIM sample → quantise → decode) ---
            gen_batches = max(1, self.validation_batches // 10)
            gen_metrics: dict[str, list[torch.Tensor]] = {
                "ssim": [],
                "lpips": [],
                "token_accuracy": [],
            }
            for val_batch in tqdm.tqdm(
                itertools.islice(self.test_loader, gen_batches),
                desc="Generation validation",
                total=gen_batches,
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
                gt_tokens = self.model.target_token_indices_from_images(val_target)
                gen_recon = torch.clamp(
                    gen_output.reconstructed_images, 0.0, 1.0
                ).float()
                gen_target = torch.clamp(val_target, 0.0, 1.0).float()
                with torch.autocast(device_type=self.device.type, enabled=False):
                    gen_metrics["ssim"].append(self.ssim(gen_recon, gen_target))
                    gen_metrics["lpips"].append(self.lpips(gen_recon, gen_target))
                gen_metrics["token_accuracy"].append(
                    (gen_output.token_indices == gt_tokens).float().mean()
                )

            gen_ssim = torch.mean(torch.stack(gen_metrics["ssim"]))
            gen_lpips = torch.mean(torch.stack(gen_metrics["lpips"]))
            gen_token_acc = torch.mean(torch.stack(gen_metrics["token_accuracy"]))
            self.write_scalar("Validation/Generation_SSIM", gen_ssim)
            self.write_scalar("Validation/Generation_LPIPS", gen_lpips)
            self.write_scalar("Validation/Generation_TokenAccuracy", gen_token_acc)

            self.checkpoint_if_best(gen_ssim)
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
            # Reconstruction (predict x₀ from noise → decode).
            val_output = self.model(
                val_content,
                val_style,
                target_images=val_target,
                target_codepoints=val_cp,
            )
            # Generation (DDIM → quantise → decode).
            gen_output = self.model.generate(
                content_images=val_content,
                style_reference_images=val_style,
                target_codepoints=val_cp,
            )

        preview_count = min(8, val_target.shape[0])
        first_style = val_style[:preview_count, 0]

        import torchvision.utils

        grid = torch.cat(
            [
                val_content[:preview_count],
                first_style,
                val_target[:preview_count],
                val_output.reconstructed_images[:preview_count],
                gen_output.reconstructed_images[:preview_count],
            ],
            dim=0,
        )
        grid_image = torchvision.utils.make_grid(
            grid, nrow=preview_count, normalize=True, value_range=(0, 1)
        )
        self.writer.add_image(
            "Visual/DiT_Preview",
            grid_image,
            global_step=self.global_step,
        )
        self.writer.flush()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DiT-based glyph generator")
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
        "--image-size",
        type=int,
        default=128,
        help="Square glyph raster size",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--class-balanced",
        action="store_true",
        help="Enable batch-level class-balanced sampling",
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
        help="Optional string of explicit style characters",
    )
    parser.add_argument(
        "--target-characters",
        type=_parse_codepoint,
        help="Optional string of extra target characters to oversample",
    )
    parser.add_argument(
        "--target-character-oversample-factor",
        type=int,
        default=8,
        help="Oversampling factor for --target-characters",
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
        "--learning-rate",
        type=float,
        default=1e-4,
        help="AdamW learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="AdamW beta1",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.95,
        help="AdamW beta2",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
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
        default="models/dit_glyph_model.pth",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--gtok-model-path",
        type=str,
        default="models/gtok_model.pth",
        help="Path to the trained G-Tok tokenizer",
    )
    parser.add_argument(
        "--target-only",
        action="store_true",
        help="Restrict dataset to only --target-characters",
    )
    parser.add_argument(
        "--limit-dataset-size",
        type=int,
        default=None,
        help="Limit dataset to this many fonts for quick canary runs",
    )
    # DiT-specific args.
    parser.add_argument(
        "--dit-hidden-size",
        type=int,
        default=832,
        help="DiT hidden dimension (default: 832)",
    )
    parser.add_argument(
        "--dit-depth",
        type=int,
        default=16,
        help="DiT transformer depth (default: 16)",
    )
    parser.add_argument(
        "--dit-num-heads",
        type=int,
        default=16,
        help="DiT attention heads (default: 16)",
    )
    parser.add_argument(
        "--dit-mlp-ratio",
        type=float,
        default=4.0,
        help="DiT MLP hidden ratio (default: 4.0)",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=1000,
        help="Number of diffusion timesteps (default: 1000)",
    )
    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=250,
        help="Number of DDIM sampling steps (default: 250)",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale (default: 1.0 = no CFG)",
    )
    parser.add_argument(
        "--gumbel-temperature",
        type=float,
        default=0.5,
        help="Gumbel-softmax temperature for auxiliary image loss",
    )

    args = parser.parse_args()
    if not args.dataset_path:
        raise ValueError(
            "GOOGLE_FONTS_REPO environment variable not set, cannot run training"
        )

    loop = DiTGlyphTrainingLoop(args)
    loop.train()
