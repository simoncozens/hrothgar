"""Latent interpolation diagnostic for G-Tok character structure.

Encodes the same glyph from two different fonts, interpolates their
pre-quantization latents, and decodes the result.  If the interpolation
stays as a recognisable instance of the same character (just transitioning
between styles), the tokenizer's feature space is character-aligned.
If it collapses to noise or becomes a different character, the space is
not organised around character identity.

Usage::

    python -m hrothgar.gtok.interpolate \\
        --gtok-model-path models/gtok_model.pth \\
        --dataset-path $GOOGLE_FONTS_REPO \\
        --char A \\
        --font-a "Roboto" \\
        --font-b "Playfair Display" \\
        --steps 8
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torchvision
from torchvision.utils import make_grid

from hrothgar.googlefonts import GoogleFonts
from hrothgar.gtok.model import GtokModel, load_model
from hrothgar.utils import torch_setup


def _find_font(gf: GoogleFonts, family: str):
    """Find a font by family name (case-insensitive, partial match)."""
    q = family.lower()
    for font in gf.fonts:
        if q in font.family.lower():
            return font
    raise ValueError(
        f"Font '{family}' not found. Available: "
        + ", ".join(sorted({f.family for f in gf.fonts}))[:200]
    )


def interpolate_latents(
    gtok: GtokModel,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    steps: int = 8,
) -> torch.Tensor:
    """Interpolate between two glyph encodings and decode the result.

    Returns a tensor of shape ``(steps, 3, H, W)`` — the decoded images
    at each interpolation step from image_a to image_b.
    """
    with torch.no_grad():
        # Encode both images through the CNN + ViT to get pre-quant latents.
        cnn_a = gtok.cnn_encoder(image_a)
        tok_a = gtok.proj_patch(cnn_a).flatten(2).transpose(1, 2)
        vit_a = gtok.vit_encoder(tok_a)
        preq_a = gtok.vit_encoder_to_quantizer(vit_a)

        cnn_b = gtok.cnn_encoder(image_b)
        tok_b = gtok.proj_patch(cnn_b).flatten(2).transpose(1, 2)
        vit_b = gtok.vit_encoder(tok_b)
        preq_b = gtok.vit_encoder_to_quantizer(vit_b)

        batch, seq, dim = preq_a.shape
        h = gtok.token_grid_height
        w = gtok.token_grid_width

        frames: list[torch.Tensor] = []
        for t in range(steps):
            alpha = t / (steps - 1) if steps > 1 else 0.0
            interp = (1 - alpha) * preq_a + alpha * preq_b

            # Quantize the interpolated latents.
            interp_4d = interp.reshape(batch, h, w, dim).permute(0, 3, 1, 2)
            quantized_4d, _loss, _idx = gtok.quantizer(interp_4d)
            quantized = quantized_4d.permute(0, 2, 3, 1).reshape(batch, seq, dim)

            decoded = gtok.decode(quantized)
            frames.append(decoded.squeeze(0))

        return torch.stack(frames, dim=0)  # (steps, 3, H, W)


def _render(font, char: str, size: int) -> torch.Tensor:
    """Render a single glyph as a (1, 3, H, W) tensor."""
    image = torch.tensor(font.render(ord(char), size=size), dtype=torch.float32)
    return image.unsqueeze(0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="G-Tok latent interpolation diagnostic"
    )
    parser.add_argument(
        "--gtok-model-path",
        type=str,
        default="models/gtok_model.pth",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=os.environ.get("GOOGLE_FONTS_REPO", ""),
    )
    parser.add_argument(
        "--char", type=str, default="A", help="Character to interpolate"
    )
    parser.add_argument(
        "--font-a", type=str, default="Roboto", help="First font (partial name match)"
    )
    parser.add_argument(
        "--font-b",
        type=str,
        default="Playfair Display",
        help="Second font (partial name match)",
    )
    parser.add_argument(
        "--steps", type=int, default=8, help="Number of interpolation steps"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Save grid image to this path"
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.dataset_path:
        parser.error("--dataset-path is required (or set GOOGLE_FONTS_REPO)")

    device = torch_setup()
    gtok, gtok_config = load_model(Path(args.gtok_model_path), device=device)
    gtok.eval()

    gf = GoogleFonts(args.dataset_path)
    font_a = _find_font(gf, args.font_a)
    font_b = _find_font(gf, args.font_b)
    char = args.char

    print(f"Font A: {font_a.family} ({font_a.style or 'regular'})")
    print(f"Font B: {font_b.family} ({font_b.style or 'regular'})")
    print(f"Char:   '{char}' (U+{ord(char):04X})")

    img_a = _render(font_a, char, gtok_config.image_size).to(device)
    img_b = _render(font_b, char, gtok_config.image_size).to(device)

    frames = interpolate_latents(gtok, img_a, img_b, steps=args.steps)

    # Build a grid: [A | step1 | step2 | ... | B]
    grid = torch.cat([img_a.cpu(), frames.cpu(), img_b.cpu()], dim=0)
    grid_img = make_grid(grid, nrow=grid.shape[0], pad_value=0.5)

    if args.output:
        torchvision.io.write_png(
            (grid_img * 255).clamp(0, 255).to(torch.uint8), args.output
        )
        print(f"Saved to {args.output}")
    else:
        from PIL import Image

        img_np = (grid_img.permute(1, 2, 0) * 255).clamp(0, 255).to(torch.uint8)
        Image.fromarray(img_np.numpy()).show()


if __name__ == "__main__":
    main()
