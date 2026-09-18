"""Export the upscaler to Core ML format.

This script converts the trained PyTorch upscaler into a single Core ML model
suitable for deployment in environments without PyTorch (e.g. Glyphs.app):

* ``upscaler.mlpackage`` — the full upscaler CNN (low-res → high-res).

The upscaler is content-preserving and has no style conditioning, so there is a
single exported model with a single ``low_res`` input.

Requirements (developer machine only): torch, coremltools, numpy.

Usage::

    python -m hrothgar.upscaler.export_coreml \\
        --model-path models/upscaler_model.pth \\
        --output-dir models/coreml
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import torch

try:
    import coremltools as ct  # type: ignore[import-untyped]
except ImportError:
    ct = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Core ML conversion
# ---------------------------------------------------------------------------


def _convert(
    model: torch.nn.Module,
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

    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, example_inputs)

    ct_inputs = [
        ct.TensorType(shape=inp.shape, name=name)
        for inp, name in zip(example_inputs, input_names)
    ]
    ct_precision = (
        ct.precision.FLOAT16 if precision == "float16" else ct.precision.FLOAT32
    )

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


def _compile(mlpackage_path: Path) -> Path | None:
    """Compile ``.mlpackage`` → ``.mlmodelc`` using xcrun."""
    mlmodelc_path = mlpackage_path.with_suffix(".mlmodelc")
    try:
        subprocess.run(
            [
                "xcrun",
                "coremlcompiler",
                "compile",
                str(mlpackage_path),
                str(mlpackage_path.parent),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"  ✓ Compiled → {mlmodelc_path.name}")
        return mlmodelc_path
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"  ⚠ Compilation skipped: {exc}")
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export upscaler to Core ML.")
    p.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained upscaler .pth file.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/coreml"),
        help="Output directory for Core ML models.",
    )
    p.add_argument("--precision", choices=("float32", "float16"), default="float16")
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip coremlcompiler compilation step.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    from hrothgar.upscaler.model import UpscalerConfig, UpscalerModel

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    # Load config from sidecar, then load weights.
    config = UpscalerConfig.from_sidecar(args.model_path)
    model = UpscalerModel(config)
    model.load(str(args.model_path), device=device)
    model.eval()
    print(f"Loaded upscaler from {args.model_path}")
    print(
        f"  low_res={config.low_res_size}  high_res={config.high_res_size}"
    )
    config.save_sidecar(args.output_dir / "upscaler_config.pth")

    # -- Upscaler ------------------------------------------------------------
    print("\n[1/1] Exporting upscaler …")
    _convert(
        model,
        (torch.randn(1, 1, config.low_res_size, config.low_res_size, device=device),),
        input_names=["low_res"],
        output_name="upscaled",
        output_path=args.output_dir / "upscaler.mlpackage",
        precision=args.precision,
    )
    if not args.no_compile:
        _compile(args.output_dir / "upscaler.mlpackage")

    print(f"\nDone.  Exports in {args.output_dir.resolve()}/")
    for f in sorted(args.output_dir.iterdir()):
        size = ""
        if f.is_file():
            size = f"  ({f.stat().st_size / 1024:.0f} KB)"
        print(f"  {f.name}{size}")


if __name__ == "__main__":
    main()
