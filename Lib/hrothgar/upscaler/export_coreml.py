"""Export upscaler components to Core ML format.

This script converts the trained PyTorch upscaler into Core ML models suitable
for deployment in environments without PyTorch (e.g. Glyphs.app).  It produces:

* ``style_encoder.mlpackage`` — GlyphStyleEncoder (reference images → FiLM params)
* ``upscaler_body.mlpackage`` — main upscaler CNN (low-res + style FiLM → high-res)
* ``style_fallback.bin`` — pre-computed FiLM vector for when no style
  references are available

Requirements (developer machine only): torch, coremltools, numpy.

Usage::

    python -m hrothgar.upscaler.export_coreml \\
        --model-path models/upscaler_model.pth \\
        --output-dir models/coreml
"""

from __future__ import annotations

import argparse
import struct
import subprocess
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

try:
    import coremltools as ct  # type: ignore[import-untyped]
except ImportError:
    ct = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Traceable wrapper modules
# ---------------------------------------------------------------------------


class _StyleEncoderExport(nn.Module):
    """Wrap ``GlyphStyleEncoder`` for Core ML export.

    The original encoder reshapes ``(B, K, 3, 512, 512)`` → ``(B*K, 3, 512, 512)``
    then pools back to ``(B, ...)``.  Since ``B=1`` at inference, we can skip
    the batch-dimension bookkeeping entirely and operate directly on ``(K, ...)``
    — avoiding all dynamic integer extraction from tensors, which coremltools
    cannot trace.

    *K* is frozen at construction time and used as a plain Python ``int``.
    """

    def __init__(self, encoder: nn.Module, K: int) -> None:
        super().__init__()
        self.backbone = encoder.backbone
        self.projection = encoder.projection
        self.K = K

    def forward(self, references: torch.Tensor) -> torch.Tensor:
        # references: (K, 3, 512, 512)
        # backbone → AdaptiveAvgPool2d ensures (K, 256, 1, 1)
        features = self.backbone(references)           # (K, 256, 1, 1)
        features = features.reshape(self.K, 256)       # (K, 256)
        pooled = features.mean(dim=0, keepdim=True)    # (1, 256)
        return self.projection(pooled).squeeze(0)      # (128,)


class _UpscalerBodyExport(nn.Module):
    """Wrap the upscaler CNN so the style FiLM vector is an explicit input.

    The exported model takes a pre-computed ``style_gamma_beta`` vector
    (128 floats) produced by the style encoder (or the learned fallback)
    rather than raw style reference images.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.input_projection = model.input_projection
        self.residual_body = model.residual_body
        self.body_projection = model.body_projection
        self.upsampler = model.upsampler
        self.output_head = model.output_head

    def forward(
        self,
        low_res: torch.Tensor,
        style_gamma_beta: torch.Tensor,
    ) -> torch.Tensor:
        """Upscale a glyph raster.

        Args:
            low_res: ``(1, 3, 128, 128)`` input raster.
            style_gamma_beta: ``(1, 128)`` FiLM (γ‖β) from the style encoder
                or fallback.

        Returns:
            ``(1, 3, 512, 512)`` upscaled glyph in [0, 1].
        """
        x = self.input_projection(low_res)

        # Residual body
        x = x + self.body_projection(self.residual_body(x))

        # Style FiLM (after residual body, before upsampling)
        gamma, beta = torch.chunk(style_gamma_beta, chunks=2, dim=-1)
        x = x * (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)

        # Upsample → sigmoid
        x = self.upsampler(x)
        x = self.output_head(x)
        return torch.sigmoid(x)


# ---------------------------------------------------------------------------
# Core ML conversion
# ---------------------------------------------------------------------------


def _convert(
    wrapper: nn.Module,
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
    """Compile ``.mlpackage`` → ``.mlmodelc`` using xcrun."""
    mlmodelc_path = mlpackage_path.with_suffix(".mlmodelc")
    try:
        subprocess.run(
            [
                "xcrun", "coremlcompiler", "compile",
                str(mlpackage_path), str(mlpackage_path.parent),
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
# Data extraction
# ---------------------------------------------------------------------------


def _extract_style_fallback(model: nn.Module, output_dir: Path) -> None:
    """Pre-compute and save the no-style fallback FiLM vector.

    When no reference glyphs are available, the original PyTorch model uses a
    learned ``_no_style_embedding`` parameter passed through
    ``style_encoder.projection``.  We compute this once and save the result
    so the runtime can use it directly without running the style encoder.
    """
    if model.style_encoder is None or model._no_style_embedding is None:
        return

    with torch.no_grad():
        style_vec = model._no_style_embedding  # (1, style_embedding_dim)
        gamma_beta = model.style_encoder.projection(style_vec).squeeze(0)  # (128,)

    data = gamma_beta.cpu().numpy().tobytes()
    path = output_dir / "style_fallback.bin"
    path.write_bytes(data)
    print(f"  ✓ {path}  ({len(data)} bytes)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export upscaler to Core ML.")
    p.add_argument("--model-path", type=Path, required=True,
                   help="Path to trained upscaler .pth file.")
    p.add_argument("--output-dir", type=Path, default=Path("models/coreml"),
                   help="Output directory for Core ML models.")
    p.add_argument("--precision", choices=("float32", "float16"), default="float16")
    p.add_argument("--no-compile", action="store_true",
                   help="Skip coremlcompiler compilation step.")
    p.add_argument("--style-reference-count", type=int, default=None,
                   help="Override K (default: use training value).")
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
    if args.style_reference_count is not None:
        config.style_reference_count = args.style_reference_count
    model = UpscalerModel(config)
    model.load(str(args.model_path), device=device)
    model.eval()
    print(f"Loaded upscaler from {args.model_path}")
    print(f"  low_res={config.low_res_size}  high_res={config.high_res_size}  "
          f"K={config.style_reference_count}")
    config.save_sidecar(args.output_dir / "upscaler_config.pth")

    K = config.style_reference_count

    # -- Style encoder ---------------------------------------------------------
    if model.style_encoder is not None:
        print("\n[1/3] Exporting style encoder …")
        se = _StyleEncoderExport(model.style_encoder, K).to(device)
        _convert(
            se,
            (torch.randn(K, 3, config.high_res_size, config.high_res_size, device=device),),
            input_names=["style_references"],
            output_name="style_gamma_beta",
            output_path=args.output_dir / "style_encoder.mlpackage",
            precision=args.precision,
        )
        if not args.no_compile:
            _compile(args.output_dir / "style_encoder.mlpackage")

    # -- Upscaler body ---------------------------------------------------------
    print("\n[2/3] Exporting upscaler body …")
    body = _UpscalerBodyExport(model).to(device)
    _convert(
        body,
        (
            torch.randn(1, 3, config.low_res_size, config.low_res_size, device=device),
            torch.randn(1, config.base_channels * 2, device=device),
        ),
        input_names=["low_res", "style_gamma_beta"],
        output_name="upscaled",
        output_path=args.output_dir / "upscaler_body.mlpackage",
        precision=args.precision,
    )
    if not args.no_compile:
        _compile(args.output_dir / "upscaler_body.mlpackage")

    # -- Style fallback --------------------------------------------------------
    print("\n[3/3] Extracting style fallback …")
    _extract_style_fallback(model, args.output_dir)

    print(f"\nDone.  Exports in {args.output_dir.resolve()}/")
    for f in sorted(args.output_dir.iterdir()):
        size = ""
        if f.is_file():
            size = f"  ({f.stat().st_size / 1024:.0f} KB)"
        print(f"  {f.name}{size}")


if __name__ == "__main__":
    main()
