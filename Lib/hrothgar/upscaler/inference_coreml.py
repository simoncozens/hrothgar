"""Core ML inference for the upscaler using coremltools.

Uses coremltools' Python inference API — no raw pointer access, no PyObjC
version headaches.  Requires ``coremltools`` and ``numpy`` at runtime.

Usage::

    from hrothgar.upscaler.inference_coreml import UpscalerInference

    infer = UpscalerInference("models/coreml")
    upscaled = infer.upscale(
        low_res=low_res_numpy,
        style_references=style_refs_numpy,
    )
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    import coremltools as ct  # type: ignore[import-untyped]
except ImportError:
    raise ImportError(
        "coremltools is required for Core ML inference. "
        "Install with: pip install coremltools"
    )


def _load_model(model_path: Path) -> ct.models.MLModel:
    """Load a Core ML model (``.mlpackage`` or ``.mlmodelc``)."""
    return ct.models.MLModel(str(model_path))


def _numpy_bytes(arr: np.ndarray) -> bytes:
    """Convert numpy array to raw float32 bytes for the style fallback."""
    return arr.astype(np.float32).tobytes()


def _model_exists(base: Path) -> bool:
    return base.with_suffix(".mlmodelc").exists() or base.with_suffix(".mlpackage").exists()


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


class UpscalerInference:
    """Run the upscaler pipeline using exported Core ML models.

    Args:
        model_dir: Directory containing the exported Core ML model files.
    """

    def __init__(self, model_dir: Union[str, Path]) -> None:
        model_dir = Path(model_dir)

        style_base = model_dir / "style_encoder"
        body_base = model_dir / "upscaler_body"

        # Resolve to whichever extension exists (.mlmodelc preferred).
        def _find(base: Path) -> Path:
            for ext in (".mlmodelc", ".mlpackage"):
                candidate = base.with_suffix(ext)
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(f"Model not found: {base}.mlmodelc or .mlpackage")

        self._style_model: Optional[ct.models.MLModel] = None
        if _model_exists(style_base):
            self._style_model = _load_model(_find(style_base))

        self._body_model = _load_model(_find(body_base))

        # Pre-computed style fallback.
        fallback_path = model_dir / "style_fallback.bin"
        if fallback_path.exists():
            self._fallback_style_gb = np.frombuffer(
                fallback_path.read_bytes(), dtype=np.float32
            ).copy()
        else:
            self._fallback_style_gb = np.zeros(128, dtype=np.float32)

    def upscale(
        self,
        low_res: np.ndarray,
        style_references: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Upscale a low-resolution glyph raster.

        Args:
            low_res: ``(3, 128, 128)`` float32 numpy array, CHW, values in [0, 1].
            style_references: Optional ``(K, 3, 512, 512)`` float32 array of
                reference glyphs for style encoding.  ``None`` uses the
                learned fallback.

        Returns:
            ``(3, 512, 512)`` float32 numpy array, CHW, values in [0, 1].
        """
        # Style gamma_beta.
        if style_references is not None and self._style_model is not None:
            result = self._style_model.predict(
                {"style_references": style_references.astype(np.float32)}
            )
            style_gb = result["style_gamma_beta"]  # (128,)
        else:
            style_gb = self._fallback_style_gb  # (128,)

        # Add batch dims for the upscaler body.
        low_res_b = low_res[np.newaxis, ...].astype(np.float32)          # (1, 3, 128, 128)
        style_gb_b = style_gb[np.newaxis, ...].astype(np.float32)        # (1, 128)

        result = self._body_model.predict(
            {
                "low_res": low_res_b,
                "style_gamma_beta": style_gb_b,
            }
        )
        upscaled = result["upscaled"]  # (1, 3, 512, 512)
        return upscaled.squeeze(0).astype(np.float32)


__all__ = ["UpscalerInference"]
