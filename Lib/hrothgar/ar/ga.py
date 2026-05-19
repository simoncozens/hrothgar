"""Glyph Adaptation (GA) training loop for GAR-Font.

GA is a glyph-specialist adaptation stage: it fine-tunes decoder LoRA layers
on a single target codepoint across many fonts. The goal is to teach the
autoregressive decoder how that specific glyph is structurally constructed and
how its style varies across type families.

Usage::

    python -m hrothgar.ar.ga \
        --base-model-path models/ar_visual_model.pth \
        --gtok-model-path models/gtok_model.pth \
        --dataset-path /path/to/google/fonts \
        --target-character "₹" \
        --lora-rank 16 \
        --target-steps 5000

The best full checkpoint is saved to ``--model-path``.
The best LoRA-only checkpoint is saved to ``--lora-model-path``.
"""

from __future__ import annotations

import itertools
import os
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torchvision
import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torchmetrics.image import StructuralSimilarityIndexMeasure

from hrothgar.ar.dataset import (
    _font_has_codepoint,
    _has_non_empty_glyph,
    _is_blank_rendering,
    _sample_style_codepoints,
)
from hrothgar.ar.losses import ARLossWeights, compute_ar_loss
from hrothgar.ar.model import ARModel, ARModelConfig, LoRAConfig
from hrothgar.googlefonts import GoogleFonts
from hrothgar.gtok.llamagen_lpips import LPIPS
from hrothgar.gtok.model import load_model
from hrothgar.utils import TrainingLoop


def glyph_lora_model_path(base_model_path: str | Path, target_codepoint: int) -> Path:
    """Deterministically derive the GA LoRA checkpoint path.

    Example:
        models/ar_visual_model.pth + U+20B9 -> models/ar_visual_model.ga-20B9.pth
    """
    if target_codepoint < 0:
        raise ValueError(
            f"target_codepoint must be non-negative, got {target_codepoint}"
        )

    base_path = Path(base_model_path)
    return base_path.parent / f"{base_path.stem}.ga-{target_codepoint:04X}.pth"


class GAGlyphDataset(TorchDataset):
    """One-codepoint dataset spanning many fonts."""

    def __init__(self, fonts: Sequence, target_codepoint: int) -> None:
        self.target_codepoint = target_codepoint
        self.fonts = list(fonts)

    def __len__(self) -> int:
        return len(self.fonts)

    def __getitem__(self, idx: int) -> dict:
        return {
            "font": self.fonts[idx],
            "char": self.target_codepoint,
        }


class GADatasetMaker:
    """Creates train/val datasets for glyph adaptation.

    The dataset is built from fonts that contain a usable non-empty glyph for
    ``target_codepoint``. Fonts are split by family name to avoid leakage.
    """

    def __init__(
        self,
        dataset_path: str,
        *,
        target_codepoint: int,
        batch_size: int,
        image_size: int = 128,
        style_glyph_count: int = 8,
        common_style_codepoints: Optional[Sequence[int]] = None,
        split_seed: int = 1234,
        max_fonts: Optional[int] = None,
    ) -> None:
        if style_glyph_count <= 0:
            raise ValueError(
                f"style_glyph_count must be positive, got {style_glyph_count}"
            )
        if max_fonts is not None and max_fonts <= 1:
            raise ValueError(f"max_fonts must be > 1 when set, got {max_fonts}")

        self.target_codepoint = target_codepoint
        self.batch_size = batch_size
        self.image_size = image_size
        self.style_glyph_count = style_glyph_count
        self.common_style_codepoints = common_style_codepoints

        google_fonts = GoogleFonts(dataset_path)
        candidate_fonts = [
            font
            for font in google_fonts.fonts
            if _font_has_codepoint(font, target_codepoint)
            and _has_non_empty_glyph(font, target_codepoint)
        ]

        if max_fonts is not None and len(candidate_fonts) > max_fonts:
            generator = torch.Generator().manual_seed(split_seed)
            perm = torch.randperm(len(candidate_fonts), generator=generator).tolist()
            candidate_fonts = [candidate_fonts[i] for i in perm[:max_fonts]]

        if len(candidate_fonts) < 4:
            raise ValueError(
                "Glyph adaptation needs at least 4 candidate fonts with a usable "
                f"target glyph; found {len(candidate_fonts)}"
            )

        train_fonts, val_fonts = self._split_fonts_by_family(
            candidate_fonts,
            split_seed=split_seed,
        )

        if len(train_fonts) == 0 or len(val_fonts) == 0:
            raise ValueError(
                "Family-aware split produced an empty train or validation set for "
                f"target codepoint U+{target_codepoint:04X}"
            )

        print(
            "GA glyph U+"
            f"{target_codepoint:04X}: {len(train_fonts)} train fonts / "
            f"{len(val_fonts)} val fonts"
        )

        self.train_set = GAGlyphDataset(train_fonts, target_codepoint)
        self.val_set = GAGlyphDataset(val_fonts, target_codepoint)

    @staticmethod
    def _split_fonts_by_family(fonts: Sequence, split_seed: int) -> tuple[list, list]:
        family_to_fonts: dict[str, list] = {}
        for font in fonts:
            family = getattr(font, "family", str(getattr(font, "path", "UNKNOWN")))
            family_to_fonts.setdefault(family, []).append(font)

        families = sorted(family_to_fonts.keys())
        if len(families) < 2:
            return list(fonts), []

        train_families, val_families = train_test_split(
            families,
            test_size=0.2,
            random_state=split_seed,
        )

        train_fonts: list = []
        val_fonts: list = []
        for family in train_families:
            train_fonts.extend(family_to_fonts[family])
        for family in val_families:
            val_fonts.extend(family_to_fonts[family])

        return train_fonts, val_fonts

    def collate_fn(self, batch: list) -> dict:
        chars = torch.tensor([item["char"] for item in batch], dtype=torch.long)
        target_renderings = torch.stack(
            [
                torch.tensor(item["font"].render(item["char"], size=self.image_size))
                for item in batch
            ]
        )

        content_renderings = []
        style_renderings = []
        style_chars_list = []
        descriptions = []

        for item in batch:
            font = item["font"]
            char = item["char"]
            reference_font = font.reference_font() or font

            if not _font_has_codepoint(
                reference_font, char
            ) or not _has_non_empty_glyph(reference_font, char):
                reference_font = font

            content_render = reference_font.render(char, size=self.image_size)
            if _is_blank_rendering(content_render):
                content_render = font.render(char, size=self.image_size)
            content_renderings.append(torch.tensor(content_render))

            sampled_style_chars = _sample_style_codepoints(
                font=font,
                target_char=char,
                style_glyph_count=self.style_glyph_count,
                common_style_codepoints=self.common_style_codepoints,
            )
            rendered_styles = []
            sanitized_style_chars = []
            for cp in sampled_style_chars:
                style_render = font.render(cp, size=self.image_size)
                if _is_blank_rendering(style_render):
                    cp = char
                    style_render = font.render(cp, size=self.image_size)
                sanitized_style_chars.append(cp)
                rendered_styles.append(torch.tensor(style_render))

            style_renderings.append(torch.stack(rendered_styles))
            style_chars_list.append(
                torch.tensor(sanitized_style_chars, dtype=torch.long)
            )
            descriptions.append(font.description_with_tags_and_display())

        return {
            "target_rendering": target_renderings,
            "content_rendering": torch.stack(content_renderings),
            "style_renderings": torch.stack(style_renderings),
            "chars": chars,
            "style_chars": torch.stack(style_chars_list),
            "description": descriptions,
        }

    def train_loader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            drop_last=True,
        )

    def val_loader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            drop_last=False,
        )


class ARGlyphAdaptationTrainingLoop(TrainingLoop):
    """Glyph-specialist LoRA adaptation loop for the AR decoder."""

    def post_init(self, train_args) -> None:
        config = ARModelConfig(image_size=train_args.image_size)

        if not os.path.exists(train_args.gtok_model_path):
            raise ValueError(f"G-Tok model not found at {train_args.gtok_model_path}")
        if not os.path.exists(train_args.base_model_path):
            raise ValueError(f"Base AR model not found at {train_args.base_model_path}")
        if not train_args.dataset_path:
            raise ValueError("--dataset-path is required for glyph adaptation")

        gtok, gtok_config = load_model(Path(train_args.gtok_model_path), device=self.device)

        model = ARModel(config, gtok_model=gtok).to(self.device)
        model.load(train_args.base_model_path, device=self.device)

        lora_config = LoRAConfig(
            rank=train_args.lora_rank,
            alpha=train_args.lora_alpha,
        )
        model.enable_nfa_mode(lora_config)
        print(
            "LoRA injected. Trainable parameters: "
            f"{sum(p.numel() for p in model.trainable_parameters()):,}"
        )

        common_style_cps = train_args.style_characters
        if train_args.style_glyph_count < len(common_style_cps or []):
            train_args.style_glyph_count = len(common_style_cps)

        maker = GADatasetMaker(
            dataset_path=train_args.dataset_path,
            target_codepoint=train_args.target_codepoint,
            batch_size=train_args.batch_size,
            image_size=config.image_size,
            style_glyph_count=train_args.style_glyph_count,
            common_style_codepoints=common_style_cps,
            split_seed=train_args.split_seed,
            max_fonts=train_args.max_fonts,
        )

        self.optimizer = torch.optim.AdamW(
            model.trainable_parameters(),
            lr=train_args.learning_rate,
            betas=(train_args.beta1, train_args.beta2),
        )
        self.train_loader = maker.train_loader()
        self.test_loader = maker.val_loader()

        self.loss_weights = ARLossWeights()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips = LPIPS().to(self.device)

        self.model = model
        self.target_steps = train_args.target_steps
        self.validation_every = train_args.validation_every
        self.validation_batches = train_args.validation_batches
        self.grad_accum_steps = train_args.grad_accum_steps

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

        if self.target_steps is None:
            raise ValueError(
                "target_steps must not be None for ARGlyphAdaptationTrainingLoop"
            )

        self.num_epochs = (self.target_steps // max(len(self.train_loader), 1)) + 1
        self.validation_direction = "higher"

        if train_args.lora_model_path is None:
            self.lora_model_path = str(
                glyph_lora_model_path(
                    train_args.base_model_path,
                    train_args.target_codepoint,
                )
            )
        else:
            self.lora_model_path = train_args.lora_model_path

    def _autocast_context(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def train_step(self, batch: dict) -> tuple:
        target_images = batch["target_rendering"].to(self.device)
        content_images = batch["content_rendering"].to(self.device)
        style_images = batch["style_renderings"].to(self.device)
        descriptions = batch.get("description")

        model_output = self.model(
            content_images,
            style_images,
            target_images=target_images,
            descriptions=descriptions,
        )
        loss, loss_info = compute_ar_loss(
            model_output,
            target_images,
            weights=self.loss_weights,
        )
        return loss, loss_info

    def train(self) -> None:
        if len(self.train_loader) == 0:
            raise ValueError("Training loader is empty; cannot start GA training.")
        import pkbar

        steps_per_epoch = len(self.train_loader)

        try:
            while not self.must_stop():
                kbar = pkbar.Kbar(
                    target=steps_per_epoch,
                    epoch=self.epoch,
                    num_epochs=self.num_epochs,
                )
                self.model.train()
                self.model.gtok.eval()
                self.optimizer.zero_grad(set_to_none=True)

                for i, batch in enumerate(self.train_loader):
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
                        i + 1 == steps_per_epoch
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

    def post_train_step(self) -> None:
        if self.global_step % self.validation_every != 0:
            return

        self.model.eval()
        with torch.no_grad():
            val_metrics: dict = {"ssim": [], "lpips": []}
            for val_batch in tqdm.tqdm(
                itertools.islice(self.test_loader, self.validation_batches),
                desc="Validation",
                total=min(self.validation_batches, len(self.test_loader)),
            ):
                val_target = val_batch["target_rendering"].to(self.device)
                val_content = val_batch["content_rendering"].to(self.device)
                val_style = val_batch["style_renderings"].to(self.device)
                val_descriptions = val_batch.get("description")

                with self._autocast_context():
                    val_output = self.model(
                        val_content,
                        val_style,
                        target_images=val_target,
                        descriptions=val_descriptions,
                    )
                recon_clamped = torch.clamp(val_output.reconstructed_images, 0.0, 1.0)
                target_clamped = torch.clamp(val_target, 0.0, 1.0)
                val_metrics["ssim"].append(
                    self.ssim(recon_clamped, target_clamped).mean()
                )
                val_metrics["lpips"].append(
                    self.lpips(recon_clamped, target_clamped).mean()
                )

            avg_ssim = torch.mean(torch.stack(val_metrics["ssim"]))
            avg_lpips = torch.mean(torch.stack(val_metrics["lpips"]))
            self.write_scalar("Validation/SSIM", avg_ssim)
            self.write_scalar("Validation/LPIPS", avg_lpips)
            is_best = self.checkpoint_if_best(avg_ssim)
            if is_best:
                lora_state = self.model.token_decoder.get_lora_state_dict()
                torch.save(lora_state, self.lora_model_path)
                print(f"  Glyph LoRA checkpoint saved to {self.lora_model_path}")
            self.visualize()

        self.model.train()
        self.model.gtok.eval()

    def visualize(self) -> None:
        val_batch = next(iter(self.test_loader))
        val_target = val_batch["target_rendering"].to(self.device)
        val_content = val_batch["content_rendering"].to(self.device)
        val_style = val_batch["style_renderings"].to(self.device)
        val_descriptions = val_batch.get("description")

        with self._autocast_context():
            val_output = self.model(
                val_content,
                val_style,
                target_images=val_target,
                descriptions=val_descriptions,
            )
            autoregression_output = self.model.generate(
                content_images=val_content,
                style_reference_images=val_style,
                descriptions=val_descriptions,
            )

        preview_count = min(8, val_target.shape[0])
        first_style = val_style[:preview_count, 0]
        recon_grid = torch.cat(
            [
                val_content[:preview_count],
                first_style,
                val_target[:preview_count],
                val_output.reconstructed_images[:preview_count],
                autoregression_output.reconstructed_images[:preview_count],
            ],
            dim=0,
        )
        self.writer.add_image(
            "Reconstruction/content_style_target_recon_ga",
            torchvision.utils.make_grid(recon_grid, nrow=preview_count),
            self.global_step,
        )


def _parse_codepoint(value: str) -> List[int]:
    return [ord(c) for c in value]


def _parse_single_codepoint(value: str) -> int:
    codepoints = _parse_codepoint(value)
    if len(codepoints) != 1:
        raise ValueError(
            f"--target-character expects exactly one character, got {len(codepoints)}"
        )
    return codepoints[0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Glyph adaptation fine-tuning for AR decoder LoRA"
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        required=True,
        help="Path to the pretrained AR model checkpoint",
    )
    parser.add_argument(
        "--gtok-model-path",
        type=str,
        default="models/gtok_model.pth",
        help="Path to the trained G-Tok model",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=os.environ.get("GOOGLE_FONTS_REPO"),
        help="Path to the Google Fonts repository",
    )
    parser.add_argument(
        "--target-character",
        type=_parse_single_codepoint,
        required=True,
        help="Single target glyph character for adaptation (for example: '₹')",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/ar_ga_model.pth",
        help="Path to save the adapted full model checkpoint",
    )
    parser.add_argument(
        "--lora-model-path",
        type=str,
        default=None,
        help=(
            "Optional path to save the best LoRA-only checkpoint. If omitted, "
            "uses <base_model_stem>.ga-<CODEPOINT_HEX>.pth next to --base-model-path"
        ),
    )
    parser.add_argument(
        "--max-fonts",
        type=int,
        default=None,
        help="Optional cap on number of fonts used for glyph adaptation",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=1234,
        help="Seed for deterministic font family splitting and optional subsampling",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank r (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=16.0,
        help="LoRA scaling alpha (default: 16.0)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Square glyph raster size (must match the base model)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for glyph adaptation",
    )
    parser.add_argument(
        "--style-glyph-count",
        type=int,
        default=8,
        help="Number of style reference glyphs N_s",
    )
    parser.add_argument(
        "--style-characters",
        type=_parse_codepoint,
        help="Optional fixed style character string",
    )
    parser.add_argument(
        "--target-steps",
        type=int,
        default=5000,
        help="Total GA optimisation steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="AdamW learning rate",
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
        "--precision",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        default="bf16",
        help="Training precision",
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
        default=500,
        help="Run validation every N optimisation steps",
    )
    parser.add_argument(
        "--validation-batches",
        type=int,
        default=20,
        help="Number of validation batches per validation pass",
    )
    parser.add_argument(
        "--tag",
        type=str,
        help="Optional tag for the TensorBoard run directory",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow training with uncommitted git changes",
    )

    args = parser.parse_args()
    if not args.dataset_path:
        raise ValueError(
            "GOOGLE_FONTS_REPO environment variable not set, cannot run GA training"
        )

    if args.lora_model_path is None:
        args.lora_model_path = str(
            glyph_lora_model_path(args.base_model_path, args.target_character)
        )
        print(f"Using deterministic glyph LoRA path: {args.lora_model_path}")

    loop = ARGlyphAdaptationTrainingLoop(args)
    loop.train()
