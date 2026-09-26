"""Style-swap test: does the decoder actually use the style tokens?

Generates each target glyph twice — once with its own font's style tokens and
once with a different font's tokens (rolled within the batch) — and saves a
comparison grid plus the mean |own − swapped| L1.

Interpretation
--------------
- Mean L1 ≈ 0 and visually identical → the decoder is ignoring style (usage
  failure).
- Outputs differ only in weight (bold/light) → style is used coarsely (the
  representation is too coarse for fine detail).
- Outputs differ in serif/terminal too → style transfer is working.
"""

from __future__ import annotations

import argparse

import torch
import torchvision

from hrothgar.style_extraction import load_model
from hrothgar.style_extraction.dataset import StyleExtractionDatasetMaker
from hrothgar.utils import torch_setup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", default="outputs/style_swap.png")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = torch_setup()
    model, config = load_model(args.model_path, device)

    maker = StyleExtractionDatasetMaker(
        repo_url=args.dataset_path,
        batch_size=args.batch_size,
        image_size=config.image_size,
        character_set=config.character_set,
        num_evidence_glyphs=config.num_evidence_glyphs,
    )

    batch = next(iter(maker.test_loader()))
    style_images = batch["style_images"].to(device)
    style_cp = batch["style_codepoint_idx"].to(device)
    target_idx = batch["target_codepoint_idx"].to(device)
    target_images = batch["target_images"].to(device)

    with torch.no_grad():
        tokens = model.encode_style(style_images, style_codepoint_idx=style_cp)
        own = model.decode(target_idx, tokens)
        # Each sample renders with the *next* sample's style tokens.
        swapped = model.decode(target_idx, torch.roll(tokens, shifts=1, dims=0))

    l1 = torch.abs(own - swapped).mean().item()
    print(f"Mean |own - swapped| L1: {l1:.6f}")

    n = min(8, target_images.shape[0])
    grid = torch.cat([target_images[:n], own[:n], swapped[:n]], dim=0)
    torchvision.utils.save_image(grid, args.output, nrow=n)
    print(f"Saved grid to {args.output} (rows: GT / own-style / swapped-style)")
    print("If 'own' and 'swapped' look identical, the decoder is ignoring style.")


if __name__ == "__main__":
    main()
