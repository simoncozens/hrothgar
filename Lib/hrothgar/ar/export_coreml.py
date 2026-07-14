"""Export MaskGIT generation model to Core ML format.

Produces three Core ML models:

* ``gen_encoder.mlpackage`` — content image + style refs + codepoint → conditioning map
* ``gen_transformer.mlpackage`` — token indices + cond map → logits
* ``gen_softdecoder.mlpackage`` — logits → reconstructed image

Requirements: torch, coremltools, numpy.

Usage::

    python -m hrothgar.ar.export_coreml \\
        --model-path models/ar_model.pth \\
        --gtok-model-path models/gtok_model.pth \\
        --output-dir models/coreml_gen
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Optional

import torch

from hrothgar.ar.export_wrappers import (
    _EncoderExport,
    _MaskGITTransformerExport,
    _SoftDecoderExport,
)

try:
    import coremltools as ct  # type: ignore[import-untyped]
except ImportError:
    ct = None  # type: ignore[assignment]


def _convert(
    wrapper: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    output_name: str,
    output_path: Path,
    *,
    precision: str = "float16",
) -> None:
    """Trace and export a module as a Core ML ``.mlpackage``."""
    if ct is None:
        raise RuntimeError("coremltools is required.  pip install coremltools")

    wrapper.eval()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_inputs)

    ct_inputs = [
        ct.TensorType(shape=inp.shape, name=name)
        for inp, name in zip(example_inputs, input_names)
    ]
    ct_precision = ct.precision.FLOAT16 if precision == "float16" else ct.precision.FLOAT32

    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=[ct.TensorType(name=output_name)],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        compute_precision=ct_precision,
    )
    mlmodel.save(str(output_path))
    print(f"  ✓ {output_path}")


def _compile(mlpackage_path: Path) -> Optional[Path]:
    mlmodelc_path = mlpackage_path.with_suffix(".mlmodelc")
    try:
        subprocess.run(
            ["xcrun", "coremlcompiler", "compile",
             str(mlpackage_path), str(mlpackage_path.parent)],
            check=True, capture_output=True, text=True,
        )
        print(f"  ✓ Compiled → {mlmodelc_path.name}")
        return mlmodelc_path
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"  ⚠ Compilation skipped: {exc}")
        return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export MaskGIT generator to Core ML.")
    p.add_argument("--model-path", type=Path, required=True,
                   help="Path to trained AR model .pth file.")
    p.add_argument("--gtok-model-path", type=Path, required=True,
                   help="Path to GTok model .pth file.")
    p.add_argument("--output-dir", type=Path, default=Path("models/coreml_gen"))
    p.add_argument("--precision", choices=("float32", "float16"), default="float16")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--style-reference-count", type=int, default=4,
                   help="Number of style reference glyphs (K).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    from hrothgar.ar.model import ARModelConfig, ARModel
    from hrothgar.gtok.model import load_model as load_gtok

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    # Load config from sidecar.
    config = ARModelConfig.from_sidecar(args.model_path)
    K = args.style_reference_count
    H = config.image_size

    # Load GTok.
    gtok, _gtok_config = load_gtok(args.gtok_model_path, device)

    # Load AR model.
    model = ARModel(config, gtok_model=gtok).to(device)
    model.load(str(args.model_path), device=device)
    model.eval()
    model.freeze_gtok()
    print(f"Loaded generator from {args.model_path}")
    print(f"  image_size={H}  K={K}  seq_len={model.sequence_length}")

    # Save config to output dir.
    config.save_sidecar(args.output_dir / "gen_config.pth")

    # ---- [1/3] Encoder -------------------------------------------------------
    print("\n[1/3] Exporting encoder …")
    enc = _EncoderExport(model, K).to(device)
    _convert(
        enc,
        (
            torch.randn(1, 3, H, H, device=device),
            torch.randn(1, K, 3, H, H, device=device),
            torch.zeros(1, dtype=torch.long, device=device),
        ),
        input_names=["content_image", "style_refs", "latincore_idx"],
        output_name="conditioning_map",
        output_path=args.output_dir / "gen_encoder.mlpackage",
        precision=args.precision,
    )
    if not args.no_compile:
        _compile(args.output_dir / "gen_encoder.mlpackage")

    # ---- [2/3] Transformer ---------------------------------------------------
    print("\n[2/3] Exporting transformer …")
    trans = _MaskGITTransformerExport(model.maskgit_decoder.transformer).to(device)
    N = model.sequence_length
    _convert(
        trans,
        (
            torch.zeros(1, N, dtype=torch.long, device=device),
            torch.randn(1, config.encoder_feature_dim * 2,
                        model.token_grid_height, model.token_grid_width, device=device),
        ),
        input_names=["token_indices", "conditioning_map"],
        output_name="logits",
        output_path=args.output_dir / "gen_transformer.mlpackage",
        precision=args.precision,
    )
    if not args.no_compile:
        _compile(args.output_dir / "gen_transformer.mlpackage")

    # ---- [3/3] Soft Decoder --------------------------------------------------
    print("\n[3/3] Exporting soft decoder …")
    sd = _SoftDecoderExport(model).to(device)
    _convert(
        sd,
        (torch.randn(1, N, model.codebook_size, device=device),),
        input_names=["logits"],
        output_name="images",
        output_path=args.output_dir / "gen_softdecoder.mlpackage",
        precision=args.precision,
    )
    if not args.no_compile:
        _compile(args.output_dir / "gen_softdecoder.mlpackage")

    print(f"\nDone.  Exports in {args.output_dir.resolve()}/")
    for f in sorted(args.output_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
