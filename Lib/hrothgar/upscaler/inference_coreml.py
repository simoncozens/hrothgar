"""Core ML inference for the upscaler using coremltools.

Uses coremltools' Python inference API — no raw pointer access, no PyObjC
version headaches.  Requires ``coremltools`` and ``numpy`` at runtime.

Usage::

    from hrothgar.upscaler.inference_coreml import UpscalerInference

    infer = UpscalerInference("models/coreml")
    upscaled = infer.upscale(low_res=low_res_numpy)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import coremltools as ct  # type: ignore[import-untyped]
except ImportError:
    raise ImportError(
        "coremltools is required for Core ML inference. "
        "Install with: pip install coremltools"
    )


def _load_model(model_path: Path) -> ct.models.MLModel:
    return ct.models.MLModel(str(model_path))


# Anchors: keep input arrays alive so Python doesn't GC their buffers
# before CoreML's async cleanup thread releases them (MLE5ExecutionStream
# resetQueue).  Without this, libcoremlpython.so double-frees the buffer.
_ANCHORS: list[np.ndarray] = []
_MAX_ANCHORS = 20


def _predict(model: ct.models.MLModel, **inputs: np.ndarray) -> dict:
    """Call model.predict() with anchored input copies."""
    anchored = {}
    for k, v in inputs.items():
        a = np.ascontiguousarray(v, dtype=np.float32)
        _ANCHORS.append(a)
        anchored[k] = a
    result = model.predict(anchored)
    # Trim old anchors to bound memory.
    while len(_ANCHORS) > _MAX_ANCHORS:
        _ANCHORS.pop(0)
    return result


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


class UpscalerInference:
    """Run the upscaler pipeline using the exported Core ML model.

    Args:
        model_dir: Directory containing the exported Core ML model files.
    """

    def __init__(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)

        base = model_dir / "upscaler"

        def _find(path_base: Path) -> Path:
            for ext in (".mlmodelc", ".mlpackage"):
                candidate = path_base.with_suffix(ext)
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(
                f"Model not found: {path_base}.mlmodelc or .mlpackage"
            )

        self._model = _load_model(_find(base))

    def upscale(self, low_res: np.ndarray) -> np.ndarray:
        """Upscale a low-resolution glyph raster.

        Args:
            low_res: ``(1, H, W)`` float32 numpy array, CHW, values in [0, 1].

        Returns:
            ``(1, H*f, W*f)`` float32 numpy array, CHW, values in [0, 1], where
            ``f`` is the upscale factor of the exported model.
        """
        low_res_b = low_res[np.newaxis, ...].astype(np.float32)
        result = _predict(self._model, low_res=low_res_b)
        upscaled = result["upscaled"]
        return upscaled.squeeze(0).astype(np.float32)


__all__ = ["UpscalerInference"]
