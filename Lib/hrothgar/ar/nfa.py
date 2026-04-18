"""Novel Font Adaptation (NFA) training loop for GAR-Font.

NFA is the first post-refinement stage described in the GAR-Font paper.
It takes a pretrained AR checkpoint and a target font, then fine-tunes
only the LoRA layers of the Transformer decoder on that font's existing
glyphs.  This lightweight adaptation allows the model to better capture
the stylistic nuances of fonts that differ significantly from the
training distribution — display type, script styles, unusual weights, etc.

Usage::

    python -m hrothgar.ar.nfa \\
        --base-model-path models/ar_visual_model.pth \\
        --font-path /path/to/Font-Regular.ttf \\
        --gtok-model-path models/gtok_model.pth \\
        --lora-rank 16 \\
        --target-steps 5000

The adapted checkpoint (base weights + LoRA) is saved to ``--model-path``.
A LoRA-only checkpoint is saved alongside it with the suffix ``_lora.pth``,
for compact storage and later merging with other base checkpoints.
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
from hrothgar.dataset import LATIN_CORE
from hrothgar.googlefonts import GoogleFont, GoogleFonts, StandaloneFont
from hrothgar.gtok.llamagen_lpips import LPIPS
from hrothgar.gtok.model import GtokConfig, GtokModel
from hrothgar.utils import TrainingLoop


class NFAGlyphDataset(TorchDataset):
    """Single-font glyph dataset for NFA fine-tuning.

    Each item is a ``{"font": ..., "char": codepoint}`` dict that the
    collation function in ``NFADatasetMaker`` can render.
    """

    def __init__(self, font, codepoints: Sequence[int]) -> None:
        self.font = font
        self.order: List[int] = list(codepoints)

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, idx: int) -> dict:
        return {"font": self.font, "char": self.order[idx]}


class NFADatasetMaker:
    """Creates train/val datasets from a single font for NFA fine-tuning.

    The font's LATIN_CORE codepoints are split 80/20 into train and
    validation sets.  The collation logic mirrors ``ARPhase1DatasetMaker``:
    each batch contains a target rendering, a content rendering (from a
    reference/neutral font), and a set of style renderings drawn from the
    same font.

    Args:
        font: Any font object with ``codepoints``, ``render(char, size)``,
            ``reference_font()``, and ``has_codepoint(char)`` attributes.
            Both ``GoogleFont`` and ``StandaloneFont`` are accepted.
        batch_size: Number of items per training batch.
        image_size: Square raster size in pixels.
        style_glyph_count: Number of style reference glyphs per item (N_s).
        common_style_codepoints: If set, these codepoints are used as the
            fixed style set (same semantics as in ``ARPhase1DatasetMaker``).
        target_codepoints: If set, restrict the dataset to these codepoints
            rather than the full LATIN_CORE intersection.
    """

    def __init__(
        self,
        font,
        batch_size: int,
        image_size: int = 128,
        style_glyph_count: int = 8,
        common_style_codepoints: Optional[Sequence[int]] = None,
        target_codepoints: Optional[Sequence[int]] = None,
    ) -> None:
        self.font = font
        self.batch_size = batch_size
        self.image_size = image_size
        self.style_glyph_count = style_glyph_count
        self.common_style_codepoints = common_style_codepoints

        if target_codepoints is not None:
            candidate_codepoints = [
                cp for cp in target_codepoints if font.has_codepoint(cp)
            ]
        else:
            candidate_codepoints = [
                cp
                for cp in LATIN_CORE
                if font.has_codepoint(cp) and _has_non_empty_glyph(font, cp)
            ]

        if len(candidate_codepoints) < 2:
            raise ValueError(
                f"Font '{font.family}' has fewer than 2 usable codepoints "
                f"(found {len(candidate_codepoints)}).  NFA requires at least "
                "two codepoints so that style and target glyphs can differ."
            )

        train_cps, val_cps = train_test_split(
            candidate_codepoints, test_size=0.2, random_state=42
        )
        print(f"NFA font '{font.family}': {len(train_cps)} train / {len(val_cps)} val codepoints")
        self.train_set = NFAGlyphDataset(font, train_cps)
        self.val_set = NFAGlyphDataset(font, val_cps)

    def collate_fn(self, batch: list) -> dict:
        """Collate a list of ``{"font", "char"}`` items into model-ready tensors.

        Returns the same keys as ``ARPhase1DatasetMaker.collate_fn``:
        ``target_rendering``, ``content_rendering``, ``style_renderings``,
        ``chars``, ``style_chars``, and ``descriptions``.
        """
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

            if not _font_has_codepoint(reference_font, char) or not _has_non_empty_glyph(
                reference_font, char
            ):
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
            style_chars_list.append(torch.tensor(sanitized_style_chars, dtype=torch.long))
            descriptions.append(font.description() if hasattr(font, "description") else "")

        return {
            "target_rendering": target_renderings,
            "content_rendering": torch.stack(content_renderings),
            "style_renderings": torch.stack(style_renderings),
            "chars": chars,
            "style_chars": torch.stack(style_chars_list),
            "descriptions": descriptions,
        }

    def train_loader(self) -> DataLoader:
        """DataLoader for the training split."""
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            drop_last=True,
        )

    def val_loader(self) -> DataLoader:
        """DataLoader for the validation split."""
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            drop_last=False,
        )


class ARNFATrainingLoop(TrainingLoop):
    """Novel Font Adaptation fine-tuning training loop.

    Loads a pretrained AR checkpoint, injects LoRA into the decoder, and
    fine-tunes only the LoRA parameters using the same CE + pixel-L1 loss
    as the visual pretraining stage.

    The adapted full checkpoint is saved to ``model_path``; a compact LoRA-
    only checkpoint is saved to the same path with a ``_lora.pth`` suffix.
    """

    def post_init(self, train_args) -> None:
        config = ARModelConfig(image_size=train_args.image_size)

        if not os.path.exists(train_args.gtok_model_path):
            raise ValueError(
                f"G-Tok model not found at {train_args.gtok_model_path}"
            )
        if not os.path.exists(train_args.base_model_path):
            raise ValueError(
                f"Base AR model not found at {train_args.base_model_path}"
            )

        gtok = GtokModel(GtokConfig())
        gtok.load(train_args.gtok_model_path, device=self.device)

        model = ARModel(config, gtok_model=gtok).to(self.device)
        model.load(train_args.base_model_path, device=self.device)

        # Load the target font.
        font = _load_font(
            font_path=train_args.font_path,
            dataset_path=train_args.dataset_path,
        )

        lora_config = LoRAConfig(
            rank=train_args.lora_rank,
            alpha=train_args.lora_alpha,
        )
        model.enable_nfa_mode(lora_config)
        print(
            f"LoRA injected.  Trainable parameters: "
            f"{sum(p.numel() for p in model.trainable_parameters()):,}"
        )

        common_style_cps = train_args.style_characters
        if train_args.style_glyph_count < len(common_style_cps or []):
            train_args.style_glyph_count = len(common_style_cps)

        maker = NFADatasetMaker(
            font=font,
            batch_size=train_args.batch_size,
            image_size=config.image_size,
            style_glyph_count=train_args.style_glyph_count,
            common_style_codepoints=common_style_cps,
            target_codepoints=train_args.target_characters,
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
            raise ValueError("target_steps must not be None for ARNFATrainingLoop")

        self.num_epochs = (self.target_steps // max(len(self.train_loader), 1)) + 1
        self.validation_direction = "higher"

        # LoRA-only checkpoint path derived from model_path.
        stem = Path(self.model_path).stem
        parent = Path(self.model_path).parent
        self.lora_model_path = str(parent / f"{stem}_lora.pth")

    def _autocast_context(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def train_step(self, batch: dict) -> tuple:
        target_images = batch["target_rendering"].to(self.device)
        content_images = batch["content_rendering"].to(self.device)
        style_images = batch["style_renderings"].to(self.device)

        model_output = self.model(
            content_images,
            style_images,
            target_images=target_images,
        )
        loss, loss_info = compute_ar_loss(
            model_output,
            target_images,
            weights=self.loss_weights,
        )
        return loss, loss_info

    def train(self) -> None:
        if len(self.train_loader) == 0:
            raise ValueError("Training loader is empty; cannot start NFA training.")
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
                # G-Tok must always stay in eval.
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

                with self._autocast_context():
                    val_output = self.model(
                        val_content, val_style, target_images=val_target
                    )
                val_metrics["ssim"].append(
                    self.ssim(val_output.reconstructed_images, val_target)
                )
                val_metrics["lpips"].append(
                    self.lpips(val_output.reconstructed_images, val_target)
                )

            avg_ssim = torch.mean(torch.stack(val_metrics["ssim"]))
            avg_lpips = torch.mean(torch.stack(val_metrics["lpips"]))
            self.write_scalar("Validation/SSIM", avg_ssim)
            self.write_scalar("Validation/LPIPS", avg_lpips)
            is_best = self.checkpoint_if_best(avg_ssim)
            if is_best:
                # Also save the compact LoRA-only checkpoint alongside the full one.
                lora_state = self.model.token_decoder.get_lora_state_dict()
                torch.save(lora_state, self.lora_model_path)
                print(f"  LoRA checkpoint saved to {self.lora_model_path}")
            self.visualize()

        self.model.train()
        self.model.gtok.eval()

    def visualize(self) -> None:
        val_batch = next(iter(self.test_loader))
        val_target = val_batch["target_rendering"].to(self.device)
        val_content = val_batch["content_rendering"].to(self.device)
        val_style = val_batch["style_renderings"].to(self.device)

        with self._autocast_context():
            val_output = self.model(val_content, val_style, target_images=val_target)
            autoregression_output = self.model.generate(
                content_images=val_content,
                style_reference_images=val_style,
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
            "Reconstruction/content_style_target_recon",
            torchvision.utils.make_grid(recon_grid, nrow=preview_count),
            self.global_step,
        )


def _load_font(
    font_path: Optional[str],
    dataset_path: Optional[str],
) -> "StandaloneFont | GoogleFont":
    """Load the adaptation target font.

    If ``font_path`` is given it is loaded as a ``StandaloneFont``.  When
    ``dataset_path`` is also provided, Noto Sans from the Google Fonts repo
    is used as the reference font for content conditioning.

    Raises ``ValueError`` if neither argument is provided.
    """
    if font_path is None:
        raise ValueError("--font-path is required for NFA training.")

    reference: Optional[StandaloneFont | GoogleFont] = None
    if dataset_path is not None:
        # Load a lightweight GoogleFonts instance just to get Noto Sans.
        try:
            gf = GoogleFonts(dataset_path)
            reference = gf.families_by_name.get("Noto Sans")
        except Exception as exc:
            print(f"Warning: could not load reference font from repo ({exc}); "
                  "content glyphs will fall back to the target font itself.")

    return StandaloneFont(font_path, reference=reference)


def _parse_codepoint(value: str) -> List[int]:
    return [ord(c) for c in value]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Novel Font Adaptation fine-tuning for the AR model"
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        required=True,
        help="Path to the pretrained AR model checkpoint",
    )
    parser.add_argument(
        "--font-path",
        type=str,
        required=True,
        help="Path to the target font file (.ttf or .otf)",
    )
    parser.add_argument(
        "--gtok-model-path",
        type=str,
        default="models/gtok_model.pth",
        help="Path to the trained G-Tok model",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/ar_nfa_model.pth",
        help="Path to save the adapted model checkpoint",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=os.environ.get("GOOGLE_FONTS_REPO"),
        help=(
            "Optional path to the Google Fonts repo.  When provided, Noto Sans "
            "is loaded from the repo for content-glyph conditioning."
        ),
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
        help="LoRA scaling alpha (default: 16.0, giving scale=1.0 at default rank)",
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
        default=8,
        help="Batch size for NFA fine-tuning (default: 8, smaller than pretraining)",
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
        help="Optional fixed style character string (e.g. 'adhesionADHESION')",
    )
    parser.add_argument(
        "--target-characters",
        type=_parse_codepoint,
        help="Optional string of target characters to adapt to",
    )
    parser.add_argument(
        "--target-steps",
        type=int,
        default=5000,
        help="Total NFA optimisation steps (default: 5000)",
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

    args = parser.parse_args()
    loop = ARNFATrainingLoop(args)
    loop.train()
