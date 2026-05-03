"""CLI for visualising G-Tok tokenizer reconstruction on arbitrary glyphs.

Usage::

    python -m hrothgar.gtok path/to/font.ttf --char A --model-path models/gtok_model.pth
    python -m hrothgar.gtok path/to/font.ttf --gid 5   --model-path models/gtok_model.pth

Loads the model config from the sidecar JSON (``<model_path>.conf.json``),
constructs the G-Tok model, loads its weights, renders the requested glyph
at the tokenizer's native resolution, tokenizes it and decodes it, then saves
three PNG files to the output directory:

* ``<stem>_<label>_input.png``   – the rendered input image
* ``<stem>_<label>_recon.png``   – the tokenizer reconstruction
* ``<stem>_<label>_diff.png``    – absolute pixel difference (scaled ×4)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from hrothgar.googlefonts import StandaloneFont
from hrothgar.gtok.model import GtokConfig, GtokModel


def _parse_char(value: str) -> int:
    """Parse a single Unicode character or U+XXXX codepoint string to an int."""
    if value.startswith(("U+", "u+")):
        return int(value[2:], 16)
    if len(value) != 1:
        raise ValueError(
            "--char must be a single Unicode character or a U+XXXX codepoint"
        )
    return ord(value)


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_model(model_path: Path, device: torch.device) -> tuple[GtokModel, GtokConfig]:
    """Load GtokModel from weights and its sidecar config JSON.

    Raises ``FileNotFoundError`` if either the weights or the sidecar are missing.
    """
    config_path = model_path.with_suffix(".conf.json")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Sidecar config not found: {config_path}\n"
            "Run GTok training first so the .conf.json is written alongside the .pth."
        )
    with config_path.open("r", encoding="utf-8") as fh:
        config_dict = json.load(fh)
    config = GtokConfig(**config_dict)

    model = GtokModel(config).to(device)
    model.load(str(model_path), device=device)
    model.eval()
    return model, config


def _chw_to_hwc(arr: np.ndarray) -> np.ndarray:
    """Convert (C, H, W) float32 array to (H, W, C) clipped to [0, 1]."""
    return np.transpose(arr, (1, 2, 0)).clip(0.0, 1.0)


def _save_png(path: Path, image_chw: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(str(path), _chw_to_hwc(image_chw), vmin=0.0, vmax=1.0)
    print(f"Saved {path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render a glyph, tokenize it with G-Tok, reconstruct it, and "
            "save input/reconstruction/diff PNGs."
        )
    )
    parser.add_argument("font", type=Path, help="Path to a font file (.ttf / .otf)")

    glyph_group = parser.add_mutually_exclusive_group(required=True)
    glyph_group.add_argument(
        "--char",
        type=str,
        help="Unicode character or U+XXXX codepoint to render",
    )
    glyph_group.add_argument(
        "--gid",
        type=int,
        help="Glyph ID to render",
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/gtok_model.pth"),
        help="Path to the trained G-Tok weights (.pth); sidecar .conf.json must exist beside it",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/gtok_preview"),
        help="Directory to write output PNG files",
    )
    return parser


def main() -> None:
    """Entry point for ``python -m hrothgar.gtok``."""
    parser = _build_parser()
    args = parser.parse_args()

    if not args.font.exists():
        raise FileNotFoundError(f"Font file not found: {args.font}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"G-Tok model file not found: {args.model_path}")

    device = _pick_device()
    print(f"Using device: {device}")

    model, config = _load_model(args.model_path, device)
    image_size = config.image_size
    print(f"Loaded G-Tok model (image_size={image_size})")

    font = StandaloneFont(args.font)

    if args.gid is not None:
        label = f"gid_{args.gid}"
        print(f"Rendering glyph ID {args.gid}")
        rendered = font.render_gid(args.gid, size=image_size)
    else:
        char = _parse_char(args.char)
        label = f"cp_{char:04X}"
        codepoint = f"U+{char:04X}"
        print(f"Rendering character {chr(char)!r} ({codepoint})")
        rendered = font.render(char, size=image_size)

    # rendered is (3, H, W) float32
    input_tensor = torch.tensor(rendered, dtype=torch.float32, device=device).unsqueeze(
        0
    )

    with torch.no_grad():
        reconstructed_tensor, _ = model(input_tensor)

    reconstructed = reconstructed_tensor.squeeze(0).detach().cpu().numpy()
    diff = np.abs(rendered - reconstructed) * 4.0  # amplify ×4 for visibility

    stem = f"{args.font.stem}_{label}"
    _save_png(args.output_dir / f"{stem}_input.png", rendered)
    _save_png(args.output_dir / f"{stem}_recon.png", reconstructed)
    _save_png(args.output_dir / f"{stem}_diff.png", diff)


if __name__ == "__main__":
    main()
