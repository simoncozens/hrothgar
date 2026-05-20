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
        raise ValueError("--target-char must be a single character or U+XXXX")
    return ord(value)


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
                descriptions=descriptions,
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
            "Lightly fine-tune GTok + AR NFA on one font and generate an "
            "upscaled glyph"
        )
    )
    parser.add_argument("--font-path", type=Path, required=True)
    parser.add_argument("--target-char", type=str, required=True)

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

    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--output-size", type=int, default=512)

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
        "--save-intermediate-128",
        action="store_true",
        help="Also save the raw 128x128 generated bitmap",
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

    target_char = _parse_char(args.target_char)
    style_codepoints = _parse_codepoint_string(args.style_characters)
    device = pick_device()
    print(f"Using device: {device}")

    matched_google_font = find_google_font_by_basename(
        args.dataset_path, args.font_path
    )
    font_description = matched_google_font.description_with_tags_and_display()

    # Load GTok and adapt only its decoder path on Latin Core glyphs.
    gtok, gtok_config = load_gtok_model(args.gtok_model_path, device)
    if args.gtok_epochs > 0:
        fine_tune_gtok_decoder_only(
            model=gtok,
            font=matched_google_font,
            image_size=args.image_size,
            description=font_description,
            config=GtokFineTuneConfig(
                epochs=args.gtok_epochs,
                batch_size=args.gtok_batch_size,
                learning_rate=args.gtok_learning_rate,
            ),
            device=device,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    finetuned_gtok_path = args.gtok_model_path
    if args.gtok_epochs > 0:
        finetuned_gtok_path = args.finetuned_gtok_path
        if finetuned_gtok_path is None:
            finetuned_gtok_path = (
                args.output_dir / f"{args.font_path.stem}_gtok_decoder_finetuned.pth"
            )
        gtok.save(str(finetuned_gtok_path))
        gtok_config.save_sidecar(finetuned_gtok_path)
        print(f"Saved fine-tuned GTok checkpoint: {finetuned_gtok_path}")

    # Ensure a deterministic glyph-specialist LoRA exists for this target codepoint.
    glyph_lora_path = glyph_lora_model_path(args.ar_model_path, target_char)
    if glyph_lora_path.exists():
        print(f"Found existing glyph LoRA checkpoint: {glyph_lora_path}")
    else:
        print(
            "Glyph LoRA checkpoint not found; running glyph adaptation to create: "
            f"{glyph_lora_path}"
        )
        ga_model_path = glyph_lora_path.with_name(f"{glyph_lora_path.stem}.full.pth")
        ga_args = SimpleNamespace(
            allow_dirty=True,
            tag=f"GA-{args.font_path.stem}-U{target_char:04X}",
            model_path=str(ga_model_path),
            base_model_path=str(args.ar_model_path),
            gtok_model_path=str(finetuned_gtok_path),
            dataset_path=str(args.dataset_path),
            target_codepoint=target_char,
            batch_size=args.ga_batch_size,
            image_size=args.image_size,
            style_glyph_count=args.style_glyph_count,
            style_characters=style_codepoints,
            split_seed=1234,
            max_fonts=args.ga_max_fonts,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_steps=args.ga_target_steps,
            learning_rate=args.ga_learning_rate,
            beta1=args.ga_beta1,
            beta2=args.ga_beta2,
            precision="bf16" if device.type == "cuda" else "fp32",
            grad_accum_steps=1,
            validation_every=args.ga_validation_every,
            validation_batches=args.ga_validation_batches,
            lora_model_path=str(glyph_lora_path),
        )
        ga_loop = ARGlyphAdaptationTrainingLoop(ga_args)
        ga_loop.train()
        if not glyph_lora_path.exists():
            raise RuntimeError(
                "Glyph adaptation completed but LoRA checkpoint was not found at "
                f"{glyph_lora_path}"
            )
        print(f"Saved glyph LoRA checkpoint: {glyph_lora_path}")

    # Load AR, restore GTok weights into it, then run NFA.
    ar_config = ARModelConfig(image_size=args.image_size)
    checkpoint_has_language_adapter = _checkpoint_contains_prefix(
        args.ar_model_path,
        "language_adapter.",
    )
    language_adapter = None
    if checkpoint_has_language_adapter:
        print(
            "Detected language_adapter.* weights in AR checkpoint; "
            "enabling multimodal AR adapter in generate.py"
        )
        language_adapter = TextStyleAdapter(
            TextStyleAdapterConfig(
                style_token_dim=ar_config.encoder_feature_dim,
                text_embedding_dim=args.text_embedding_dim,
                adapter_hidden_dim=args.adapter_hidden_dim,
                num_layers=args.adapter_layers,
                num_heads=args.adapter_heads,
                dropout=args.adapter_dropout,
            )
        )

    ar_model = ARModel(
        ar_config,
        gtok_model=gtok,
        language_adapter=language_adapter,
    ).to(device)
    ar_model.load(str(args.ar_model_path), device=device)
    ar_model.gtok.load_state_dict(gtok.state_dict())
    ar_model.freeze_gtok()

    text_encoder = None
    if ar_model.language_adapter is not None:
        text_encoder = HashedDescriptionEncoder(
            HashedDescriptionEncoderConfig(
                vocab_size=args.text_hash_vocab_size,
                embedding_dim=args.text_embedding_dim,
                max_tokens=args.text_max_tokens,
            )
        ).to(device)
        text_encoder.eval()

    lora_config = LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha)

    # Enable composed NFA mode: the glyph LoRA (GA output) is loaded as a
    # frozen adapter, and a fresh trainable font adapter is injected on top.
    # NFA only updates the font adapter, preserving the glyph structural prior.
    print(f"Loading glyph LoRA from {glyph_lora_path} for composed NFA mode...")
    glyph_lora_state = torch.load(glyph_lora_path, map_location=device)
    ar_model.enable_composed_nfa_mode(glyph_lora_state, lora_config)
    print(
        "Composed NFA mode enabled: glyph adapter frozen, font adapter trainable"
    )

    nfa_maker = NFADatasetMaker(
        font=matched_google_font,
        batch_size=args.nfa_batch_size,
        image_size=args.image_size,
        style_glyph_count=args.style_glyph_count,
        common_style_codepoints=style_codepoints,
        style_warmup_epochs=args.style_warmup_epochs,
        style_extra_per_epoch=args.style_extra_per_epoch,
        style_schedule_seed=args.style_schedule_seed,
        target_codepoints=None,
    )
    fine_tune_ar_nfa(
        model=ar_model,
        maker=nfa_maker,
        epochs=args.nfa_epochs,
        batch_size=args.nfa_batch_size,
        learning_rate=args.nfa_learning_rate,
        beta1=args.nfa_beta1,
        beta2=args.nfa_beta2,
        composed_glyph_weight=args.composed_glyph_weight,
        composed_font_weight_final=args.composed_font_weight_final,
        composed_font_ramp_epochs=args.composed_font_ramp_epochs,
        device=device,
        description=font_description,
    )

    finetuned_ar_path = args.finetuned_ar_path
    if finetuned_ar_path is None:
        finetuned_ar_path = (
            args.output_dir / f"{args.font_path.stem}_ar_nfa_finetuned.pth"
        )
    ar_model.save(str(finetuned_ar_path))
    print(f"Saved fine-tuned AR checkpoint: {finetuned_ar_path}")

    # Also save LoRA-only adaptation weights.
    lora_only_path = finetuned_ar_path.with_name(f"{finetuned_ar_path.stem}_lora.pth")
    torch.save(ar_model.token_decoder.get_lora_state_dict(), lora_only_path)
    print(f"Saved LoRA-only checkpoint: {lora_only_path}")

    # Generate a 128x128 glyph.
    # Missing target glyphs in the adapted font are expected in this workflow.
    # We only require that at least one content source can render the target
    # character (selected reference font or the target font itself).
    reference_for_target = matched_google_font.reference_font() or matched_google_font
    if not _font_has_codepoint(
        reference_for_target, target_char
    ) and not _font_has_codepoint(matched_google_font, target_char):
        raise ValueError(
            "Target character "
            f"U+{target_char:04X} is missing in both the target font and the "
            "selected reference font. Provide --dataset-path with a suitable "
            "--reference-family (noto-sans or noto-serif), or choose a supported "
            "target character."
        )

    if not _font_has_codepoint(matched_google_font, target_char):
        print(
            "Target character "
            f"U+{target_char:04X} is missing in the target font; using "
            f"reference font content rendering from '{args.reference_family}'."
        )

    ar_model.eval()
    content_images, style_images = _build_generation_inputs(
        font=matched_google_font,
        target_char=target_char,
        style_glyph_count=args.style_glyph_count,
        image_size=args.image_size,
        common_style_codepoints=style_codepoints,
        device=device,
    )
    with torch.no_grad():
        if ar_model.language_adapter is not None and text_encoder is not None:
            description_embeddings = text_encoder([font_description]).to(device)
            generated = ar_model.generate_adaptation(
                content_images=content_images,
                style_reference_images=style_images,
                text_embeddings=description_embeddings,
                descriptions=[font_description],
            )
        else:
            generated = ar_model.generate(
                content_images=content_images,
                style_reference_images=style_images,
                descriptions=[font_description],
            )
        generated_128 = generated.reconstructed_images.squeeze(0).detach().cpu().numpy()

    # Upscale 128x128 -> 512x512.
    upscaler = UpscalerModel(
        UpscalerConfig(
            low_res_size=args.image_size,
            high_res_size=args.output_size,
            use_gtok_encoder=True,
            use_gtok_vit_features=True,
            gtok_model_path=str(finetuned_gtok_path),
        )
    ).to(device)
    upscaler.load(str(args.upscaler_model_path), device=device)
    upscaler.eval()

    with torch.no_grad():
        low_res_tensor = torch.tensor(generated_128, dtype=torch.float32, device=device)
        low_res_tensor = low_res_tensor.unsqueeze(0)
        upscaled_512 = upscaler(low_res_tensor).squeeze(0).detach().cpu().numpy()

    codepoint_label = f"U+{target_char:04X}"
    output_512_path = (
        args.output_dir / f"{args.font_path.stem}_{codepoint_label}_512.png"
    )
    _to_image(upscaled_512).save(output_512_path)
    print(f"Saved upscaled result: {output_512_path}")

    if args.save_intermediate_128:
        output_128_path = (
            args.output_dir / f"{args.font_path.stem}_{codepoint_label}_128.png"
        )
        _to_image(generated_128).save(output_128_path)
        print(f"Saved 128x128 generation: {output_128_path}")


if __name__ == "__main__":
    main()
