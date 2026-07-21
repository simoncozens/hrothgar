"""End-to-end single-font glyph generation pipeline.

1. Load pretrained GTok, AR, and upscaler models.
2. Optionally fine-tune AR in NFA (LoRA-only) mode on the target font's glyphs.
3. Generate target glyphs via MaskGIT iterative decoding.
4. Upscale and save.

NFA fine-tuning injects LoRA adapters into the MaskGIT transformer decoder
and fine-tunes on the target font's available Latin Core glyphs for a small
number of epochs.  This is the "many-shot" advantage: we use hundreds of
known glyphs to adapt the model before generating the few missing ones.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
import uharfbuzz as hb
from tqdm import tqdm
from torch.utils.data import DataLoader

from hrothgar.ar.dataset import (
    _font_has_codepoint,
    _has_non_empty_glyph,
    _is_blank_rendering,
    _sample_style_codepoints,
)
from hrothgar.ar.lora import LoRAConfig
from hrothgar.ar.losses import ARLossWeights, compute_ar_loss
from hrothgar.ar.model import ARModel, ARModelConfig
from hrothgar.ar.nfa import NFADatasetMaker
from hrothgar.googlefonts import (
    GoogleFont,
    find_google_font_by_basename,
)
from hrothgar.gtok.model import load_model as load_gtok_model
from hrothgar.upscaler.model import UpscalerConfig, UpscalerModel
from hrothgar.utils import pick_device

# ── NFA training policy constants ──────────────────────────────────────
NFA_LEARNING_RATE: float = 2e-5
NFA_BETA1: float = 0.9
NFA_BETA2: float = 0.95
NFA_EPOCHS_DEFAULT: int = 10
NFA_BATCH_SIZE_DEFAULT: int = 8
LORA_RANK_DEFAULT: int = 16
LORA_ALPHA_DEFAULT: float = 16.0
# ────────────────────────────────────────────────────────────────────────


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


def _build_generation_inputs(
    *,
    font: GoogleFont,
    target_char: int,
    style_glyph_count: int,
    image_size: int,
    common_style_codepoints: Optional[Sequence[int]],
    device: torch.device,
    reference_for_target: Optional[GoogleFont] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reference_font = reference_for_target or font
    if not _font_has_codepoint(reference_font, target_char) or not _has_non_empty_glyph(
        reference_font, target_char
    ):
        reference_font = font

    content_render = reference_font.render(target_char, size=image_size)
    print("Rendering content from font:", reference_font.family)
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

    # Font metrics for conditioning.
    upem = float(font.hb_face.upem)
    vm = font.vertical_metrics()
    gid = hb.Font(font.hb_face).get_nominal_glyph(target_char)
    aw = font.advance_width(gid) / upem if upem > 0 else 0.0
    metrics = torch.tensor([[
        float(vm["ascender"]) / upem if upem > 0 else 0.0,
        float(vm["descender"]) / upem if upem > 0 else 0.0,
        float(vm["x_height"]) / upem if upem > 0 else 0.0,
        float(vm["cap_height"]) / upem if upem > 0 else 0.0,
        float(vm["baseline"]) / upem if upem > 0 else 0.0,
        aw,
    ]], device=device)

    return content_images, style_images, metrics


# ---------------------------------------------------------------------------
# NFA fine-tuning
# ---------------------------------------------------------------------------


def fine_tune_ar_nfa(
    *,
    model: ARModel,
    maker: NFADatasetMaker,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    beta1: float,
    beta2: float,
    device: torch.device,
) -> None:
    """Fine-tune LoRA adapters on a single font's glyphs using MaskGIT objective.

    The model must already be in NFA mode (``enable_nfa_mode`` called).
    This function fine-tunes only the LoRA parameters using the same
    masked-prediction training objective as the base model.

    Args:
        model: ARModel in NFA mode (LoRA injected, base weights frozen).
        maker: NFADatasetMaker built from the target font.
        epochs: Number of fine-tuning epochs.
        batch_size: Training batch size (clamped to dataset size).
        learning_rate: Optimizer learning rate.
        beta1: AdamW beta1.
        beta2: AdamW beta2.
        device: Torch device.
    """
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

    print(
        f"NFA fine-tuning on {len(train_set)} glyphs from "
        f"'{maker.font.family}' for {epochs} epochs"
    )
    model.train()
    model.gtok.eval()

    for epoch in range(epochs):
        maker.set_style_schedule_epoch(epoch)
        running_loss = 0.0

        for batch in tqdm(loader, desc=f"NFA epoch {epoch + 1}/{epochs}"):
            target_images = batch["target_rendering"].to(device)
            content_images = batch["content_rendering"].to(device)
            style_images = batch["style_renderings"].to(device)
            target_codepoints = batch["chars"].to(device)
            metrics = batch.get("metrics")
            if metrics is not None:
                metrics = metrics.to(device)
            target_widths = batch.get("advance_width")
            if target_widths is not None:
                target_widths = target_widths.to(device)

            optimizer.zero_grad(set_to_none=True)
            output = model(
                content_images,
                style_images,
                target_images=target_images,
                target_codepoints=target_codepoints,
                metrics=metrics,
            )
            loss, _loss_info = compute_ar_loss(
                output,
                target_images,
                weights=loss_weights,
                target_widths=target_widths,
            )
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu())

        avg_loss = running_loss / max(len(loader), 1)
        print(f"  NFA epoch {epoch + 1}: avg loss={avg_loss:.5f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune AR (LoRA NFA) on one font and generate upscaled glyphs"
        )
    )
    parser.add_argument("--font-path", type=Path, required=True)
    parser.add_argument(
        "--target-chars", type=str, required=True,
        help="Comma-separated target characters (e.g. 'A,B,C' or 'U+0041,U+0042')",
    )

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
        choices=["noto-sans", "noto-serif", "adobe-blank", "none"],
        default="noto-sans",
        help="Reference family for content glyph conditioning",
    )

    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--style-glyph-count", type=int, default=8)

    # ── NFA fine-tuning ────────────────────────────────────────────────
    parser.add_argument(
        "--nfa-epochs",
        type=int,
        default=NFA_EPOCHS_DEFAULT,
        help=f"Number of NFA fine-tuning epochs (default: {NFA_EPOCHS_DEFAULT})",
    )
    parser.add_argument(
        "--nfa-batch-size",
        type=int,
        default=NFA_BATCH_SIZE_DEFAULT,
        help=f"NFA training batch size (default: {NFA_BATCH_SIZE_DEFAULT})",
    )
    parser.add_argument(
        "--nfa-learning-rate",
        type=float,
        default=NFA_LEARNING_RATE,
        help=f"NFA learning rate (default: {NFA_LEARNING_RATE})",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=LORA_RANK_DEFAULT,
        help=f"LoRA rank r (default: {LORA_RANK_DEFAULT})",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=LORA_ALPHA_DEFAULT,
        help=f"LoRA scaling alpha (default: {LORA_ALPHA_DEFAULT})",
    )
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
        help=(
            "Additional font glyphs to add to the style pool after each "
            "warm-up epoch"
        ),
    )
    parser.add_argument(
        "--style-schedule-seed",
        type=int,
        default=1234,
        help="Seed for the deterministic order of extra style glyphs",
    )
    # ────────────────────────────────────────────────────────────────────

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/generated"),
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
        "--zero-aggregator",
        action="store_true",
        help="Zero out the FeatureFusionModule cross-attention output, "
             "isolating the global style vector + codepoint embedding",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.lora_rank <= 0:
        raise ValueError(f"lora-rank must be positive, got {args.lora_rank}")
    if args.lora_alpha <= 0:
        raise ValueError(f"lora-alpha must be positive, got {args.lora_alpha}")
    if args.nfa_epochs < 0:
        raise ValueError(f"nfa-epochs must be non-negative, got {args.nfa_epochs}")

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

    # Backward compatibility: if --target-char is used, wrap it into a list.
    if args.target_char is not None:
        target_chars = [_parse_char(args.target_char)]
    else:
        target_chars = _parse_chars(args.target_chars)

    style_codepoints = _parse_codepoint_string(args.style_characters)

    # Load config first so we can use its metadata.
    ar_config = ARModelConfig.from_sidecar(args.ar_model_path)

    # Validate target characters against the training glyphset.
    if ar_config.target_codepoints is not None:
        for tc in target_chars:
            if tc not in ar_config.target_codepoints:
                raise ValueError(
                    f"Target character U+{tc:04X} is not in the model's "
                    f"training glyphset ({len(ar_config.target_codepoints)} "
                    "codepoints)"
                )

    device = pick_device()
    print(f"Using device: {device}")

    matched_google_font = find_google_font_by_basename(
        args.dataset_path, args.font_path
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    gtok, gtok_config = load_gtok_model(Path(args.gtok_model_path), device)

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
        )
    ).to(device)
    upscaler.load(str(args.upscaler_model_path), device=device)
    upscaler.eval()

    # ── NFA fine-tuning ─────────────────────────────────────────────────
    if args.nfa_epochs > 0:
        print(f"\n--- NFA fine-tuning: {args.nfa_epochs} epochs ---")

        # Build single-font dataset for this font.
        nfa_maker = NFADatasetMaker(
            font=matched_google_font,
            batch_size=args.nfa_batch_size,
            image_size=gtok_config.image_size,
            style_glyph_count=args.style_glyph_count,
            common_style_codepoints=style_codepoints,
            style_warmup_epochs=args.style_warmup_epochs,
            style_extra_per_epoch=args.style_extra_per_epoch,
            style_schedule_seed=args.style_schedule_seed,
        )

        # Inject LoRA and freeze base weights.
        lora_cfg = LoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha)
        ar_model.enable_nfa_mode(lora_cfg)
        print(
            f"LoRA injected (rank={args.lora_rank}, alpha={args.lora_alpha}).  "
            f"Trainable parameters: "
            f"{sum(p.numel() for p in ar_model.trainable_parameters()):,}"
        )

        fine_tune_ar_nfa(
            model=ar_model,
            maker=nfa_maker,
            epochs=args.nfa_epochs,
            batch_size=args.nfa_batch_size,
            learning_rate=args.nfa_learning_rate,
            beta1=NFA_BETA1,
            beta2=NFA_BETA2,
            device=device,
        )

        # Save fine-tuned checkpoint if requested.
        if args.finetuned_ar_path is not None:
            ar_model.save(str(args.finetuned_ar_path))
            print(f"Saved fine-tuned AR checkpoint to {args.finetuned_ar_path}")
            # Also save LoRA-only checkpoint.
            lora_path = str(args.finetuned_ar_path).replace(".pth", "_lora.pth")
            lora_sd = ar_model.maskgit_decoder.transformer.get_lora_state_dict()
            torch.save(lora_sd, lora_path)
            print(f"Saved LoRA-only checkpoint to {lora_path}")

        ar_model.eval()

    else:
        ar_model.eval()

    # ── Generation ──────────────────────────────────────────────────────
    for target_char in target_chars:
        print(f"\n--- Generating U+{target_char:04X} ---")

        # Missing target glyphs in the adapted font are expected in this workflow.
        reference_for_target = matched_google_font.reference_font(args.reference_family) or matched_google_font
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

        content_images, style_images, metrics = _build_generation_inputs(
            font=matched_google_font,
            target_char=target_char,
            style_glyph_count=args.style_glyph_count,
            image_size=gtok_config.image_size,
            common_style_codepoints=style_codepoints,
            device=device,
            reference_for_target=reference_for_target
        )
        with torch.no_grad():
            generated = ar_model.generate(
                content_images=content_images,
                style_reference_images=style_images,
                target_codepoints=torch.tensor([target_char]),
                metrics=metrics,
                zero_aggregator=args.zero_aggregator,
            )
            generated_lores = generated.reconstructed_images.squeeze(0).detach().cpu().numpy()

        # Upscale
        with torch.no_grad():
            low_res_tensor = torch.tensor(generated_lores, dtype=torch.float32, device=device)
            low_res_tensor = low_res_tensor.unsqueeze(0)
            upscaled = (
                upscaler(low_res_tensor)
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
