"""End-to-end single-font glyph generation pipeline.

This script orchestrates a complete many-shot generation pass for one font:

1. Load pretrained GTok, AR, and upscaler models.
2. Build two single-font datasets:
   - all GIDs (for GTok fine-tuning)
   - Latin Core split via NFADatasetMaker (for AR NFA fine-tuning)
3. Fine-tune GTok for N epochs on all-glyph renderings.
4. Fine-tune AR in NFA (LoRA-only) mode for M epochs.
5. Generate a target glyph at 128x128.
6. Upscale the generated glyph to 512x512.
7. Save output images and adapted checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence

from hrothgar.ar.train import MASKGIT_NUM_INFERENCE_STEPS, MASKGIT_TEMPERATURE
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from hrothgar.ar.dataset import (
    _font_has_codepoint,
    _has_non_empty_glyph,
    _is_blank_rendering,
    _sample_style_codepoints,
)
from hrothgar.ar.losses import ARLossWeights, compute_ar_loss
from hrothgar.ar.model import ARModel, ARModelConfig, LoRAConfig
from hrothgar.ar.multimodal import (
    HashedDescriptionEncoder,
    HashedDescriptionEncoderConfig,
    TextStyleAdapter,
    TextStyleAdapterConfig,
)
from hrothgar.ar.ga import ARGlyphAdaptationTrainingLoop, glyph_lora_model_path
from hrothgar.ar.nfa import NFADatasetMaker
from hrothgar.googlefonts import (
    GoogleFont,
    find_google_font_by_basename,
)
from hrothgar.gtok import GtokFineTuneConfig, fine_tune_gtok_decoder_only
from hrothgar.gtok.model import load_model as load_gtok_model
from hrothgar.upscaler.model import UpscalerConfig, UpscalerModel
from hrothgar.utils import pick_device


def _parse_char(value: str) -> int:
    if value.startswith(("U+", "u+")):
        return int(value[2:], 16)
    if len(value) != 1:
        raise ValueError("Each target must be a single character or U+XXXX")
    return ord(value)


def _parse_chars(value: str) -> list[int]:
    """Parse a comma-separated string of chars/U+XXXX into a list of codepoints."""
    parts = [p.strip() for p in value.split(",")]
    if not parts or all(p == "" for p in parts):
        raise ValueError("--target-chars must specify at least one character")
    return [_parse_char(p) for p in parts]


def _parse_codepoint_string(value: Optional[str]) -> Optional[list[int]]:
    if value is None:
        return None
    return [ord(c) for c in value]


def _to_image(image_chw: np.ndarray) -> Image.Image:
    if image_chw.shape[0] != 3:
        raise ValueError(f"Expected CHW RGB image, got shape {image_chw.shape}")
    image_hwc = np.transpose(np.clip(image_chw, 0.0, 1.0), (1, 2, 0))
    image_u8 = (image_hwc * 255.0).round().astype(np.uint8)
    return Image.fromarray(image_u8)


def _checkpoint_contains_prefix(path: Path, prefix: str) -> bool:
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    return any(key.startswith(prefix) for key in state_dict)


def fine_tune_ar_nfa(
    *,
    model: ARModel,
    maker: NFADatasetMaker,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    beta1: float,
    beta2: float,
    composed_glyph_weight: float,
    composed_font_weight_final: float,
    composed_font_ramp_epochs: int,
    device: torch.device,
    description: str,
) -> None:
    train_set = maker.train_set
    if len(train_set) == 0:
        raise ValueError("NFA training split is empty; cannot fine-tune AR model.")

    loader = DataLoader(
        train_set,
        batch_size=min(batch_size, len(train_set)),
        shuffle=True,
        drop_last=False,
        collate_fn=maker.collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
    )
    loss_weights = ARLossWeights()

    print(f"AR NFA fine-tuning on {len(train_set)} samples for {epochs} epochs")
    model.train()
    model.gtok.eval()

    def _font_weight_for_epoch(epoch_index: int) -> float:
        if composed_font_ramp_epochs <= 0:
            return composed_font_weight_final
        if composed_font_ramp_epochs == 1:
            return composed_font_weight_final
        progress = min(
            max(epoch_index, 0) / float(composed_font_ramp_epochs - 1),
            1.0,
        )
        return composed_font_weight_final * progress

    for epoch in range(epochs):
        maker.set_style_schedule_epoch(epoch)
        font_weight = _font_weight_for_epoch(epoch)
        model.set_composed_lora_weights(composed_glyph_weight, font_weight)
        running_loss = 0.0
        for batch in tqdm(loader, desc=f"NFA epoch {epoch + 1}/{epochs}"):
            target_images = batch["target_rendering"].to(device)
            content_images = batch["content_rendering"].to(device)
            style_images = batch["style_renderings"].to(device)
            descriptions = [description] * target_images.shape[0]

            optimizer.zero_grad(set_to_none=True)
            output = model(
                content_images,
                style_images,
                target_images=target_images,
                # descriptions=descriptions,
            )
            loss, _ = compute_ar_loss(output, target_images, weights=loss_weights)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu())

        avg_loss = running_loss / max(len(loader), 1)
        print(
            "  NFA epoch "
            f"{epoch + 1}: avg loss={avg_loss:.5f}, "
            f"glyph_w={composed_glyph_weight:.3f}, font_w={font_weight:.3f}"
        )


def _build_generation_inputs(
    *,
    font: GoogleFont,
    target_char: int,
    style_glyph_count: int,
    image_size: int,
    common_style_codepoints: Optional[Sequence[int]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    reference_font = font.reference_font() or font
    if not _font_has_codepoint(reference_font, target_char) or not _has_non_empty_glyph(
        reference_font, target_char
    ):
        reference_font = font

    content_render = reference_font.render(target_char, size=image_size)
    if _is_blank_rendering(content_render):
        content_render = font.render(target_char, size=image_size)

    style_chars = _sample_style_codepoints(
        font=font,
        target_char=target_char,
        style_glyph_count=style_glyph_count,
        common_style_codepoints=common_style_codepoints,
    )
    style_renderings = []
    for cp in style_chars:
        style_render = font.render(cp, size=image_size)
        if _is_blank_rendering(style_render):
            style_render = font.render(target_char, size=image_size)
        style_renderings.append(torch.tensor(style_render))

    content_images = torch.tensor(content_render).unsqueeze(0).to(device)
    style_images = torch.stack(style_renderings).unsqueeze(0).to(device)
    return content_images, style_images


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Lightly fine-tune GTok + AR NFA on one font and generate one or more "
                        "upscaled glyphs"
        )
    )
    parser.add_argument("--font-path", type=Path, required=True)
    parser.add_argument("--target-chars", type=str, required=True,
                        help="Comma-separated target characters (e.g. 'A,B,C' or 'U+0041,U+0042')")

    parser.add_argument("--gtok-model-path", type=Path, required=True)
    parser.add_argument("--ar-model-path", type=Path, required=True)
    parser.add_argument("--upscaler-model-path", type=Path, required=True)

    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Google Fonts repo path used for metadata and reference glyph lookup",
    )
    parser.add_argument(
        "--reference-family",
        choices=["noto-sans", "noto-serif", "none"],
        default="noto-sans",
        help="Reference family for content glyph conditioning during NFA",
    )

    parser.add_argument("--upscale", type=int, default=4)

    parser.add_argument("--gtok-epochs", type=int, default=10)
    parser.add_argument("--nfa-epochs", type=int, default=20)

    parser.add_argument("--ga-target-steps", type=int, default=5000)
    parser.add_argument("--ga-batch-size", type=int, default=32)
    parser.add_argument("--ga-learning-rate", type=float, default=1e-4)
    parser.add_argument("--ga-beta1", type=float, default=0.9)
    parser.add_argument("--ga-beta2", type=float, default=0.95)
    parser.add_argument("--ga-validation-every", type=int, default=500)
    parser.add_argument("--ga-validation-batches", type=int, default=20)
    parser.add_argument(
        "--ga-max-fonts",
        type=int,
        default=None,
        help="Optional cap on number of fonts used by glyph adaptation",
    )

    parser.add_argument("--gtok-batch-size", type=int, default=16)
    parser.add_argument("--nfa-batch-size", type=int, default=8)

    parser.add_argument("--gtok-learning-rate", type=float, default=1e-5)
    parser.add_argument("--nfa-learning-rate", type=float, default=2e-5)
    parser.add_argument("--nfa-beta1", type=float, default=0.9)
    parser.add_argument("--nfa-beta2", type=float, default=0.95)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument(
        "--composed-glyph-weight",
        type=float,
        default=0.75,
        help="Multiplier for frozen glyph LoRA delta during composed NFA",
    )
    parser.add_argument(
        "--composed-font-weight-final",
        type=float,
        default=1.0,
        help="Final multiplier for trainable font LoRA delta during composed NFA",
    )
    parser.add_argument(
        "--composed-font-ramp-epochs",
        type=int,
        default=10,
        help="Epochs to ramp font LoRA multiplier from 0 to final value",
    )

    parser.add_argument("--style-glyph-count", type=int, default=8)
    parser.add_argument(
        "--style-characters",
        type=str,
        default=None,
        help="Optional fixed style character string (e.g. adhesionADHESION)",
    )
    parser.add_argument(
        "--style-warmup-epochs",
        type=int,
        default=0,
        help="Epochs to use only --style-characters before widening the style pool",
    )
    parser.add_argument(
        "--style-extra-per-epoch",
        type=int,
        default=0,
        help="Additional font glyphs to add to the style pool after each warm-up epoch",
    )
    parser.add_argument(
        "--style-schedule-seed",
        type=int,
        default=1234,
        help="Seed for the deterministic order of extra style glyphs",
    )

    parser.add_argument("--output-dir", type=Path, default=Path("outputs/generated"))
    parser.add_argument(
        "--finetuned-gtok-path",
        type=Path,
        default=None,
        help="Optional output path for the decoder-only fine-tuned GTok weights",
    )
    parser.add_argument(
        "--finetuned-ar-path",
        type=Path,
        default=None,
        help="Optional output path for fine-tuned AR (base + LoRA) weights",
    )
    parser.add_argument(
        "--target-char",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Also save the raw generated bitmap",
    )
    parser.add_argument(
        "--text-hash-vocab-size",
        type=int,
        default=4096,
        help="Vocabulary bucket count for hashed multimodal description encoder",
    )
    parser.add_argument(
        "--text-embedding-dim",
        type=int,
        default=512,
        help="Embedding dimension for multimodal description tokens",
    )
    parser.add_argument(
        "--text-max-tokens",
        type=int,
        default=64,
        help="Maximum tokens retained from each multimodal description",
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
        help="Attention heads per multimodal adapter layer",
    )
    parser.add_argument(
        "--adapter-dropout",
        type=float,
        default=0.1,
        help="Dropout used in multimodal adapter layers",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.composed_glyph_weight < 0.0:
        raise ValueError(
            "--composed-glyph-weight must be non-negative, got "
            f"{args.composed_glyph_weight}"
        )
    if args.composed_font_weight_final < 0.0:
        raise ValueError(
            "--composed-font-weight-final must be non-negative, got "
            f"{args.composed_font_weight_final}"
        )
    if args.composed_font_ramp_epochs < 0:
        raise ValueError(
            "--composed-font-ramp-epochs must be non-negative, got "
            f"{args.composed_font_ramp_epochs}"
        )

    if not args.font_path.exists():
        raise FileNotFoundError(f"Font file not found: {args.font_path}")
    if not args.gtok_model_path.exists():
        raise FileNotFoundError(f"GTok checkpoint not found: {args.gtok_model_path}")
    if not args.ar_model_path.exists():
        raise FileNotFoundError(f"AR checkpoint not found: {args.ar_model_path}")
    if not args.upscaler_model_path.exists():
        raise FileNotFoundError(
            f"Upscaler checkpoint not found: {args.upscaler_model_path}"
        )
    if args.dataset_path is None:
        raise ValueError("--dataset-path is required for Google Fonts metadata lookup")
    if not args.dataset_path.exists():
        raise FileNotFoundError(
            f"Google Fonts repo path not found: {args.dataset_path}"
        )

    # Backward compatibility: if --target-char is used, wrap it into a single-element list.
    if args.target_char is not None:
        target_chars = [_parse_char(args.target_char)]
    else:
        target_chars = _parse_chars(args.target_chars)

    style_codepoints = _parse_codepoint_string(args.style_characters)
    device = pick_device()
    print(f"Using device: {device}")

    matched_google_font = find_google_font_by_basename(
        args.dataset_path, args.font_path
    )
    font_description = matched_google_font.description_with_tags_and_display()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    gtok, gtok_config = load_gtok_model(Path(args.gtok_model_path), device)

    # Ensure a deterministic glyph-specialist LoRA exists for this target codepoint.
    ar_config = ARModelConfig(
        image_size=gtok_config.image_size,
        use_maskgit=True,
        maskgit_num_inference_steps=MASKGIT_NUM_INFERENCE_STEPS,
        maskgit_temperature=MASKGIT_TEMPERATURE,
    )
    ar_model = ARModel(
        ar_config,
        gtok_model=gtok,
    ).to(device)
    ar_model.load(str(args.ar_model_path), device=device)
    ar_model.gtok.load_state_dict(gtok.state_dict())
    ar_model.freeze_gtok()

    # Set up the upscaler once (shared across all target characters).
    upscaled_size = gtok_config.image_size * args.upscale
    upscaler = UpscalerModel(
        UpscalerConfig(
            low_res_size=gtok_config.image_size,
            high_res_size=upscaled_size,
            use_gtok_encoder=True,
            use_gtok_vit_features=True,
            gtok_model_path=args.gtok_model_path,
        )
    ).to(device)
    upscaler.load(str(args.upscaler_model_path), device=device)
    upscaler.eval()

    ar_model.eval()

    for target_char in target_chars:
        print(f"\n--- Generating U+{target_char:04X} ---")

        # Missing target glyphs in the adapted font are expected in this workflow.
        # We only require that at least one content source can render the target
        # character (selected reference font or the target font itself).
        reference_for_target = matched_google_font.reference_font() or matched_google_font
        if not _font_has_codepoint(
            reference_for_target, target_char
        ) and not _font_has_codepoint(matched_google_font, target_char):
            print(
                "Skipping character "
                f"U+{target_char:04X}: missing in both the target font and the "
                "selected reference font."
            )
            continue

        if not _font_has_codepoint(matched_google_font, target_char):
            print(
                "Target character "
                f"U+{target_char:04X} is missing in the target font; using "
                f"reference font content rendering from '{args.reference_family}'."
            )

        content_images, style_images = _build_generation_inputs(
            font=matched_google_font,
            target_char=target_char,
            style_glyph_count=args.style_glyph_count,
            image_size=gtok_config.image_size,
            common_style_codepoints=style_codepoints,
            device=device,
        )
        with torch.no_grad():
            generated = ar_model.generate(
                content_images=content_images,
                style_reference_images=style_images,
                target_codepoints=torch.tensor([target_char]),
                # descriptions=[font_description],
            )
            generated_lores = generated.reconstructed_images.squeeze(0).detach().cpu().numpy()

        # Upscale
        with torch.no_grad():
            low_res_tensor = torch.tensor(generated_lores, dtype=torch.float32, device=device)
            low_res_tensor = low_res_tensor.unsqueeze(0)
            upscaled = (
                upscaler(low_res_tensor, descriptions=[font_description])
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )

        codepoint_label = f"U+{target_char:04X}"
        output_upscaled_path = (
            args.output_dir / f"{args.font_path.stem}_{codepoint_label}_{upscaled_size}.png"
        )
        _to_image(upscaled).save(output_upscaled_path)
        print(f"Saved upscaled result: {output_upscaled_path}")

        if args.save_intermediate:
            output_intermediate_path = (
                args.output_dir / f"{args.font_path.stem}_{codepoint_label}_{gtok_config.image_size}.png"
            )
            _to_image(generated_lores).save(output_intermediate_path)
            print(f"Saved {gtok_config.image_size}x{gtok_config.image_size} generation: {output_intermediate_path}")


if __name__ == "__main__":
    main()
