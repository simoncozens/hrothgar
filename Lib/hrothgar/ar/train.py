import itertools
import os
from contextlib import nullcontext
from typing import List, Optional

import torch
import torchvision
import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure

from hrothgar.ar.dataset import ARPhase1DatasetMaker
from hrothgar.ar.losses import (
    ARAdaptationLossWeights,
    ARLossWeights,
    compute_ar_adaptation_loss,
    compute_ar_loss,
)
from hrothgar.ar.model import ARModel, ARModelConfig
from hrothgar.ar.multimodal import (
    HashedDescriptionEncoder,
    HashedDescriptionEncoderConfig,
    TextStyleAdapter,
    TextStyleAdapterConfig,
)
from hrothgar.gtok.llamagen_lpips import LPIPS
from hrothgar.gtok.model import GtokModel, GtokConfig
from hrothgar.utils import TrainingLoop


def _parse_codepoint(value: str) -> List[int]:
    return [ord(c) for c in value]


def _parse_int_list(value: str) -> List[int]:
    """Parse a comma-separated integer list from CLI input."""
    items = [part.strip() for part in value.split(",") if part.strip()]
    if not items:
        raise ValueError("Expected a comma-separated list of integers")
    return [int(item) for item in items]


def _gtok_config_kwargs_from_args(train_args, image_size: int) -> dict:
    gtok_config_kwargs = {
        "image_size": image_size,
    }
    if train_args.gtok_cnn_channel_multipliers is not None:
        gtok_config_kwargs["cnn_channel_multipliers"] = (
            train_args.gtok_cnn_channel_multipliers
        )
    if train_args.gtok_cnn_latent_channels is not None:
        gtok_config_kwargs["cnn_latent_channels"] = train_args.gtok_cnn_latent_channels
    if train_args.gtok_quantizer_codebook_size is not None:
        gtok_config_kwargs["quantizer_codebook_size"] = (
            train_args.gtok_quantizer_codebook_size
        )
    if train_args.gtok_quantizer_code_dim is not None:
        gtok_config_kwargs["quantizer_code_dim"] = train_args.gtok_quantizer_code_dim
    if train_args.gtok_quantizer_entropy_loss_ratio is not None:
        gtok_config_kwargs["quantizer_entropy_loss_ratio"] = (
            train_args.gtok_quantizer_entropy_loss_ratio
        )
    return gtok_config_kwargs


def _load_gtok_from_args(
    train_args, *, image_size: int, device: torch.device
) -> GtokModel:
    if not os.path.exists(train_args.gtok_model_path):
        raise ValueError(
            f"G-Tok model not found at {train_args.gtok_model_path}, cannot run AR training"
        )
    gtok = GtokModel(
        GtokConfig(**_gtok_config_kwargs_from_args(train_args, image_size=image_size))
    )
    gtok.load(train_args.gtok_model_path, device=device)
    return gtok


class ARVisualTrainingLoop(TrainingLoop):
    """Visual-only AR stage training loop.

    This matches the GAR-Font phase-1 setup: AdamW with paper betas, one
    reference-font content glyph, and configurable N_s style references.
    """

    def post_init(self, train_args):
        config = ARModelConfig(image_size=train_args.image_size)
        gtok = _load_gtok_from_args(
            train_args,
            image_size=config.image_size,
            device=self.device,
        )
        model = ARModel(config, gtok_model=gtok).to(self.device)
        if train_args.style_glyph_count < len(train_args.style_characters or []):
            train_args.style_glyph_count = len(train_args.style_characters or [])

        maker = ARPhase1DatasetMaker(
            train_args.dataset_path,
            batch_size=train_args.batch_size,
            image_size=config.image_size,
            style_glyph_count=train_args.style_glyph_count,
            common_style_codepoints=train_args.style_characters,
            target_codepoints=train_args.target_characters,
        )

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_args.learning_rate,
            betas=(train_args.beta1, train_args.beta2),
        )
        self.train_loader = maker.train_loader()
        self.test_loader = maker.test_loader()

        self.loss_weights = ARLossWeights()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips = LPIPS().to(self.device)

        self.model = model
        self.target_steps = train_args.target_steps
        self.validation_every = train_args.validation_every
        self.validation_batches = train_args.validation_batches
        self.grad_accum_steps = train_args.grad_accum_steps
        self.canary_batches = train_args.canary

        if self.grad_accum_steps <= 0:
            raise ValueError(
                f"grad_accum_steps must be positive, got {self.grad_accum_steps}"
            )

        self.use_amp = train_args.precision in {"bf16", "fp16"}
        if train_args.precision == "bf16":
            self.amp_dtype = torch.bfloat16
        elif train_args.precision == "fp16":
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = None

        if self.use_amp and self.device.type != "cuda":
            raise ValueError(
                f"precision={train_args.precision} requires CUDA, got device {self.device}"
            )

        # GradScaler is needed for fp16, but not for bf16.
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_amp and self.amp_dtype == torch.float16
        )

        if self.canary_batches != 0:
            if self.canary_batches > len(self.train_loader):
                raise ValueError(
                    "canary exceeds train loader length "
                    f"({self.canary_batches} > {len(self.train_loader)})"
                )
            # Run for ten epochs over the canary slice.
            self.target_steps = 10 * self.canary_batches

        if self.target_steps is None:
            raise ValueError("target_steps must not be None for ARVisualTrainingLoop")

        self.num_epochs = (self.target_steps // len(self.train_loader)) + 1
        self.validation_direction = "higher"  # Maximize SSIM.

    def _autocast_context(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def train_step(self, batch):
        target_images = batch["target_rendering"].to(self.device)
        content_images = batch["content_rendering"].to(self.device)
        style_reference_images = batch["style_renderings"].to(self.device)

        model_output = self.model(
            content_images,
            style_reference_images,
            target_images=target_images,
        )
        loss, loss_info = compute_ar_loss(
            model_output,
            target_images,
            weights=self.loss_weights,
        )
        return loss, loss_info

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

    def post_train_step(self):
        if self.global_step % self.validation_every != 0:
            return

        self.model.eval()
        with torch.no_grad():
            val_metrics = {"ssim": [], "lpips": []}
            for val_batch in tqdm.tqdm(
                itertools.islice(self.test_loader, self.validation_batches),
                desc="Validation",
                total=self.validation_batches,
            ):
                val_target_images = val_batch["target_rendering"].to(self.device)
                val_content_images = val_batch["content_rendering"].to(self.device)
                val_style_images = val_batch["style_renderings"].to(self.device)

                with self._autocast_context():
                    val_output = self.model(
                        val_content_images,
                        val_style_images,
                        target_images=val_target_images,
                    )
                val_metrics["ssim"].append(
                    self.ssim(val_output.reconstructed_images, val_target_images)
                )
                val_metrics["lpips"].append(
                    self.lpips(val_output.reconstructed_images, val_target_images)
                )

            avg_ssim = torch.mean(torch.stack(val_metrics["ssim"]))
            avg_lpips = torch.mean(torch.stack(val_metrics["lpips"]))
            self.write_scalar("Validation/SSIM", avg_ssim)
            self.write_scalar("Validation/LPIPS", avg_lpips)
            self.checkpoint_if_best(avg_ssim)
            self.visualize()

        self.model.train()

    def visualize(self):
        val_batch = next(iter(self.test_loader))
        val_target_images = val_batch["target_rendering"].to(self.device)
        val_content_images = val_batch["content_rendering"].to(self.device)
        val_style_images = val_batch["style_renderings"].to(self.device)

        with self._autocast_context():
            val_output = self.model(
                val_content_images,
                val_style_images,
                target_images=val_target_images,
            )
            autoregression_output = self.model.generate(
                content_images=val_content_images,
                style_reference_images=val_style_images,
            )

        preview_count = min(8, val_target_images.shape[0])
        first_style = val_style_images[:preview_count, 0]
        recon_grid = torch.cat(
            [
                val_content_images[:preview_count],
                first_style,
                val_target_images[:preview_count],
                val_output.reconstructed_images[:preview_count],
                autoregression_output.reconstructed_images[:preview_count],
            ],
            dim=0,
        )
        self.writer.add_image(
            "Reconstruction/content_style_target_recon",
            torchvision.utils.make_grid(recon_grid, nrow=preview_count),
            self.global_step,
        )


class ARMultimodalTrainingLoop(TrainingLoop):
    """Multimodal AR adaptation stage training loop.

    Trains only the language adapter while the visual AR generator stays
    frozen. The primary objective is feature-space alignment between visual-only
    and multimodal aggregation paths, with optional decoder supervision.
    """

    def post_init(self, train_args):
        config = ARModelConfig(image_size=train_args.image_size)
        gtok = _load_gtok_from_args(
            train_args,
            image_size=config.image_size,
            device=self.device,
        )
        model = ARModel(config, gtok_model=gtok).to(self.device)

        if not train_args.base_model_path:
            raise ValueError(
                "--base-model-path is required for multimodal mode to initialise from visual pretraining."
            )
        if not os.path.exists(train_args.base_model_path):
            raise ValueError(f"Base AR model not found at {train_args.base_model_path}")
        model.load(train_args.base_model_path, device=self.device)

        self.text_encoder = HashedDescriptionEncoder(
            HashedDescriptionEncoderConfig(
                vocab_size=train_args.text_hash_vocab_size,
                embedding_dim=train_args.text_embedding_dim,
                max_tokens=train_args.text_max_tokens,
            )
        ).to(self.device)
        self.text_encoder.eval()
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = False

        language_adapter = TextStyleAdapter(
            TextStyleAdapterConfig(
                style_token_dim=config.encoder_feature_dim,
                text_embedding_dim=train_args.text_embedding_dim,
                adapter_hidden_dim=train_args.adapter_hidden_dim,
                num_layers=train_args.adapter_layers,
                num_heads=train_args.adapter_heads,
                dropout=train_args.adapter_dropout,
            )
        ).to(self.device)
        model.set_language_adapter(language_adapter)

        # Phase-3 paper setup freezes the AR generator and trains only
        # multimodal adapter components.
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.language_adapter.parameters():
            parameter.requires_grad = True

        if train_args.style_glyph_count < len(train_args.style_characters or []):
            train_args.style_glyph_count = len(train_args.style_characters or [])

        maker = ARPhase1DatasetMaker(
            train_args.dataset_path,
            batch_size=train_args.batch_size,
            image_size=config.image_size,
            style_glyph_count=train_args.style_glyph_count,
            common_style_codepoints=train_args.style_characters,
            target_codepoints=train_args.target_characters,
        )

        trainable_parameters = [p for p in model.parameters() if p.requires_grad]
        if not trainable_parameters:
            raise ValueError("No trainable parameters found for multimodal mode")
        self.optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=train_args.learning_rate,
            betas=(train_args.beta1, train_args.beta2),
        )

        self.train_loader = maker.train_loader()
        self.test_loader = maker.test_loader()
        self.model = model

        self.adaptation_loss_weights = ARAdaptationLossWeights(
            alignment_l2=train_args.adaptation_alignment_weight,
            token_cross_entropy=train_args.adaptation_token_ce_weight,
            pixel_l1=train_args.adaptation_pixel_l1_weight,
        )
        self.run_decoder = (
            self.adaptation_loss_weights.token_cross_entropy > 0.0
            or self.adaptation_loss_weights.pixel_l1 > 0.0
        )

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips = LPIPS().to(self.device)
        self.target_steps = train_args.target_steps
        self.validation_every = train_args.validation_every
        self.validation_batches = train_args.validation_batches
        self.grad_accum_steps = train_args.grad_accum_steps
        self.canary_batches = train_args.canary

        if self.grad_accum_steps <= 0:
            raise ValueError(
                f"grad_accum_steps must be positive, got {self.grad_accum_steps}"
            )

        self.use_amp = train_args.precision in {"bf16", "fp16"}
        if train_args.precision == "bf16":
            self.amp_dtype = torch.bfloat16
        elif train_args.precision == "fp16":
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = None

        if self.use_amp and self.device.type != "cuda":
            raise ValueError(
                f"precision={train_args.precision} requires CUDA, got device {self.device}"
            )

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_amp and self.amp_dtype == torch.float16
        )

        if self.canary_batches != 0:
            if self.canary_batches > len(self.train_loader):
                raise ValueError(
                    "canary exceeds train loader length "
                    f"({self.canary_batches} > {len(self.train_loader)})"
                )
            self.target_steps = 10 * self.canary_batches

        if self.target_steps is None:
            raise ValueError(
                "target_steps must not be None for ARMultimodalTrainingLoop"
            )

        self.num_epochs = (self.target_steps // len(self.train_loader)) + 1
        # Alignment is an error metric, so lower is better.
        self.validation_direction = "lower"

    def _autocast_context(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def _description_embeddings(self, descriptions: List[str]) -> torch.Tensor:
        return self.text_encoder(descriptions).to(self.device)

    def train_step(self, batch):
        target_images = batch["target_rendering"].to(self.device)
        content_images = batch["content_rendering"].to(self.device)
        style_reference_images = batch["style_renderings"].to(self.device)
        description_embeddings = self._description_embeddings(batch["description"])

        model_output = self.model.forward_adaptation(
            content_images,
            style_reference_images,
            description_embeddings,
            target_images=target_images,
            run_decoder=self.run_decoder,
        )
        loss, loss_info = compute_ar_adaptation_loss(
            model_output,
            target_images=target_images,
            weights=self.adaptation_loss_weights,
        )
        return loss, loss_info

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
                self.model.gtok.eval()
                self.text_encoder.eval()
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

    def post_train_step(self):
        if self.global_step % self.validation_every != 0:
            return

        self.model.eval()
        self.model.gtok.eval()
        self.text_encoder.eval()
        with torch.no_grad():
            val_metrics = {"alignment_l2": []}
            if self.run_decoder:
                val_metrics["ssim"] = []
                val_metrics["lpips"] = []

            for val_batch in tqdm.tqdm(
                itertools.islice(self.test_loader, self.validation_batches),
                desc="Validation",
                total=self.validation_batches,
            ):
                val_target_images = val_batch["target_rendering"].to(self.device)
                val_content_images = val_batch["content_rendering"].to(self.device)
                val_style_images = val_batch["style_renderings"].to(self.device)
                val_description_embeddings = self._description_embeddings(
                    val_batch["description"]
                )

                with self._autocast_context():
                    val_output = self.model.forward_adaptation(
                        val_content_images,
                        val_style_images,
                        val_description_embeddings,
                        target_images=val_target_images,
                        run_decoder=self.run_decoder,
                    )
                    _loss, loss_info = compute_ar_adaptation_loss(
                        val_output,
                        target_images=val_target_images,
                        weights=self.adaptation_loss_weights,
                    )
                val_metrics["alignment_l2"].append(loss_info["alignment_l2"])
                if self.run_decoder:
                    val_metrics["ssim"].append(
                        self.ssim(val_output.reconstructed_images, val_target_images)
                    )
                    val_metrics["lpips"].append(
                        self.lpips(val_output.reconstructed_images, val_target_images)
                    )

            avg_alignment = torch.mean(torch.stack(val_metrics["alignment_l2"]))
            self.write_scalar("Validation/AlignmentL2", avg_alignment)

            if self.run_decoder:
                avg_ssim = torch.mean(torch.stack(val_metrics["ssim"]))
                avg_lpips = torch.mean(torch.stack(val_metrics["lpips"]))
                self.write_scalar("Validation/SSIM", avg_ssim)
                self.write_scalar("Validation/LPIPS", avg_lpips)

            self.checkpoint_if_best(avg_alignment)
            if self.run_decoder:
                self.visualize()

        self.model.train()
        self.model.gtok.eval()
        self.text_encoder.eval()

    def visualize(self):
        val_batch = next(iter(self.test_loader))
        val_target_images = val_batch["target_rendering"].to(self.device)
        val_content_images = val_batch["content_rendering"].to(self.device)
        val_style_images = val_batch["style_renderings"].to(self.device)
        val_description_embeddings = self._description_embeddings(
            val_batch["description"]
        )

        with self._autocast_context():
            val_output = self.model.forward_adaptation(
                val_content_images,
                val_style_images,
                val_description_embeddings,
                target_images=val_target_images,
                run_decoder=True,
            )
            autoregression_output = self.model.generate_adaptation(
                content_images=val_content_images,
                style_reference_images=val_style_images,
                text_embeddings=val_description_embeddings,
            )

        preview_count = min(8, val_target_images.shape[0])
        first_style = val_style_images[:preview_count, 0]
        recon_grid = torch.cat(
            [
                val_content_images[:preview_count],
                first_style,
                val_target_images[:preview_count],
                val_output.reconstructed_images[:preview_count],
                autoregression_output.reconstructed_images[:preview_count],
            ],
            dim=0,
        )
        self.writer.add_image(
            "Reconstruction/content_style_target_recon_multimodal",
            torchvision.utils.make_grid(recon_grid, nrow=preview_count),
            self.global_step,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train AR model")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["visual", "multimodal"],
        default="visual",
        help="Training mode: visual pretraining or multimodal adaptation",
    )
    parser.add_argument(
        "--canary",
        type=int,
        default=0,
        help="If nonzero, use this many train batches and run a short canary loop",
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
        help="Square glyph raster size for AR training",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (paper default: 32)",
    )
    parser.add_argument(
        "--style-glyph-count",
        type=int,
        default=8,
        help="Number of style glyph references N_s (paper default: 8)",
    )
    parser.add_argument(
        "--style-characters",
        type=_parse_codepoint,
        help=("Optional string of explicit style characters shared across items."),
    )
    parser.add_argument(
        "--target-characters",
        type=_parse_codepoint,
        help=(
            "Optional string of target characters to emit in train/test datasets. "
            "When set, fonts are filtered to contain them and emitted chars are restricted to this set."
        ),
    )
    parser.add_argument(
        "--target-steps",
        type=int,
        default=600_000,
        help="Training iterations (paper: 600k for small set, 1M for large set)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="AdamW learning rate (paper default: 1e-4)",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="AdamW beta1 (paper default: 0.9)",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.95,
        help="AdamW beta2 (paper default: 0.95)",
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
        help="Number of micro-batches to accumulate per optimizer step",
    )
    parser.add_argument(
        "--validation-every",
        type=int,
        default=1000,
        help="Run validation every N optimization steps",
    )
    parser.add_argument(
        "--validation-batches",
        type=int,
        default=100,
        help="Number of validation batches per validation pass",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to save the trained model",
        default="models/ar_visual_model.pth",
    )
    parser.add_argument(
        "--gtok-model-path",
        type=str,
        help="Path to load the trained tokenizer model",
        default="models/gtok_model.pth",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default=None,
        help=(
            "Path to a pretrained visual AR checkpoint. Required for "
            "--mode multimodal."
        ),
    )
    parser.add_argument(
        "--gtok-cnn-channel-multipliers",
        type=_parse_int_list,
        default=None,
        help=(
            "Optional comma-separated tokenizer CNN channel multipliers used when "
            "instantiating the tokenizer for checkpoint loading (for example: 1,2,2,4,4)."
        ),
    )
    parser.add_argument(
        "--gtok-cnn-latent-channels",
        type=int,
        default=None,
        help="Optional tokenizer latent channel count used for checkpoint loading.",
    )
    parser.add_argument(
        "--gtok-quantizer-codebook-size",
        type=int,
        default=None,
        help="Optional tokenizer codebook size used for checkpoint loading.",
    )
    parser.add_argument(
        "--gtok-quantizer-code-dim",
        type=int,
        default=None,
        help="Optional tokenizer code dimensionality used for checkpoint loading.",
    )
    parser.add_argument(
        "--gtok-quantizer-entropy-loss-ratio",
        type=float,
        default=None,
        help="Optional tokenizer entropy loss ratio used for checkpoint loading.",
    )
    parser.add_argument(
        "--adaptation-alignment-weight",
        type=float,
        default=1.0,
        help="Weight for multimodal visual-text alignment L2 loss",
    )
    parser.add_argument(
        "--adaptation-token-ce-weight",
        type=float,
        default=0.0,
        help="Optional token-level CE supervision weight in multimodal mode",
    )
    parser.add_argument(
        "--adaptation-pixel-l1-weight",
        type=float,
        default=0.0,
        help="Optional pixel L1 supervision weight in multimodal mode",
    )
    parser.add_argument(
        "--text-hash-vocab-size",
        type=int,
        default=4096,
        help="Vocabulary bucket count for hashed description encoder",
    )
    parser.add_argument(
        "--text-embedding-dim",
        type=int,
        default=512,
        help="Embedding dimension for text description tokens",
    )
    parser.add_argument(
        "--text-max-tokens",
        type=int,
        default=64,
        help="Maximum tokens retained from each text description",
    )
    parser.add_argument(
        "--adapter-hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension for multimodal text-style adapter",
    )
    parser.add_argument(
        "--adapter-layers",
        type=int,
        default=6,
        help="Number of cross-attention layers in multimodal adapter",
    )
    parser.add_argument(
        "--adapter-heads",
        type=int,
        default=8,
        help="Attention heads in each multimodal adapter layer",
    )
    parser.add_argument(
        "--adapter-dropout",
        type=float,
        default=0.1,
        help="Dropout used in multimodal adapter layers",
    )

    args = parser.parse_args()
    if not args.dataset_path:
        raise ValueError(
            "GOOGLE_FONTS_REPO environment variable not set, cannot run training"
        )

    if args.mode == "multimodal" and not args.base_model_path:
        raise ValueError("--base-model-path is required when --mode multimodal")

    loop = (
        ARVisualTrainingLoop(args)
        if args.mode == "visual"
        else ARMultimodalTrainingLoop(args)
    )
    loop.train()
