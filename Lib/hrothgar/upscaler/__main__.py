"""CLI for visualizing glyph super-resolution outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from hrothgar.googlefonts import StandaloneFont
from hrothgar.render import render_gid
from hrothgar.upscaler.model import UpscalerConfig, UpscalerModel


def _parse_char(value: str) -> int:
    """Parse a glyph char argument.

    Accepts either a single Unicode character or a U+XXXX style codepoint.
    """
    if value.startswith(("U+", "u+")):
        return int(value[2:], 16)
    if len(value) != 1:
        raise ValueError(
            "--char must be a single Unicode character or U+XXXX codepoint"
        )
    return ord(value)


def _array_for_plot(image_chw: np.ndarray) -> np.ndarray:
    if image_chw.shape[0] != 3:
        raise ValueError(f"Expected CHW image with 3 channels, got {image_chw.shape}")
    return np.transpose(image_chw, (1, 2, 0)).clip(0.0, 1.0)


def _save_image(path: Path, image_chw: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, _array_for_plot(image_chw), vmin=0.0, vmax=1.0)


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _render_pair(
    font: StandaloneFont, *, char: int | None, gid: int | None
) -> tuple[np.ndarray, np.ndarray, str]:
    if gid is not None:
        low_res = render_gid(font.path, gid=gid, size=128)
        high_res = render_gid(font.path, gid=gid, size=512)
        label = f"gid_{gid}"
        return low_res, high_res, label

    assert char is not None
    low_res = font.render(char, size=128)
    high_res = font.render(char, size=512)
    codepoint = f"U+{char:04X}"
    label = f"cp_{char:04X}"
    print(f"Rendering character {chr(char)!r} ({codepoint})")
    return low_res, high_res, label


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render and compare native glyph rasters (128/512) and model upscaling (128->512)."
        )
    )
    parser.add_argument("font", type=Path, help="Path to a font file")

    glyph_group = parser.add_mutually_exclusive_group(required=True)
    glyph_group.add_argument(
        "--gid",
        type=int,
        help="Glyph ID to render",
    )
    glyph_group.add_argument(
        "--char",
        type=str,
        help="Unicode character or U+XXXX codepoint",
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/upscaler_model.pth"),
        help="Path to trained upscaler model weights",
    )
    parser.add_argument(
        "--gtok-model-path",
        type=Path,
        default=Path("models/gtok_model.pth"),
        help="Path to pretrained GTok weights",
    )
    parser.add_argument(
        "--disable-gtok-encoder",
        action="store_true",
        help="Disable GTok feature conditioning",
    )
    parser.add_argument(
        "--disable-gtok-vit",
        action="store_true",
        help="Use GTok CNN features only",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/upscaler_preview"),
        help="Directory to save output images",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open matplotlib preview window",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.font.exists():
        raise FileNotFoundError(f"Font file not found: {args.font}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Upscaler model file not found: {args.model_path}")

    char = _parse_char(args.char) if args.char is not None else None
    gid = args.gid

    font = StandaloneFont(args.font)
    low_res, high_res, label = _render_pair(font, char=char, gid=gid)

    device = _pick_device()
    print(f"Using device: {device}")

    config = UpscalerConfig(
        low_res_size=128,
        high_res_size=512,
        use_gtok_encoder=not args.disable_gtok_encoder,
        use_gtok_vit_features=not args.disable_gtok_vit,
        gtok_model_path=str(args.gtok_model_path),
    )
    model = UpscalerModel(config).to(device)
    model.load(str(args.model_path), device=device)
    model.eval()

    low_res_tensor = torch.tensor(
        low_res, dtype=torch.float32, device=device
    ).unsqueeze(0)
    with torch.no_grad():
        upscaled = model(low_res_tensor).squeeze(0).detach().cpu().numpy()

    stem = f"{args.font.stem}_{label}"
    low_path = args.output_dir / f"{stem}_128.png"
    high_path = args.output_dir / f"{stem}_512_native.png"
    upscaled_path = args.output_dir / f"{stem}_512_upscaled.png"
    figure_path = args.output_dir / f"{stem}_comparison.png"

    _save_image(low_path, low_res)
    _save_image(high_path, high_res)
    _save_image(upscaled_path, upscaled)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plot_items = [
        (low_res, "Native 128"),
        (high_res, "Native 512"),
        (upscaled, "Upscaled 128->512"),
    ]

    for axis, (image, title) in zip(axes, plot_items):
        axis.imshow(image[0], cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")

    fig.suptitle(f"{args.font.name} - {label}")
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150)

    print("Saved:")
    print(f"  {low_path}")
    print(f"  {high_path}")
    print(f"  {upscaled_path}")
    print(f"  {figure_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
