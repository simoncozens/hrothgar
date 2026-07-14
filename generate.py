"""End-to-end single-font glyph generation pipeline.

1. Load pretrained GTok, AR, and upscaler models.
2. Generate target glyphs.
3. Upscale and save.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image

from hrothgar.ar.dataset import (
    _font_has_codepoint,
    _has_non_empty_glyph,
    _is_blank_rendering,
    _sample_style_codepoints,
)
from hrothgar.ar.model import ARModel, ARModelConfig
from hrothgar.googlefonts import (
    GoogleFont,
    find_google_font_by_basename,
)
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

    parser.add_argument("--style-glyph-count", type=int, default=8)
    parser.add_argument(
        "--style-characters",
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

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()



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

    # Load config first so we can use its metadata.
    ar_config = ARModelConfig.from_sidecar(args.ar_model_path)
    style_codepoints = ar_config.style_codepoints

    # Validate target characters against the training glyphset.
    if ar_config.target_codepoints is not None:
        for tc in target_chars:
            if tc not in ar_config.target_codepoints:
                raise ValueError(
                    f"Target character U+{tc:04X} is not in the model's "
                    f"training glyphset ({len(ar_config.target_codepoints)} codepoints)"
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
