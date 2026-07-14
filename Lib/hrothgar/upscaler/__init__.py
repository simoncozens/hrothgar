"""Glyph super-resolution components.

In training mode (torch available), re-exports all public symbols.
In inference mode (no torch), imports only what's needed.
"""

try:
    from hrothgar.upscaler.dataset import UpscalerDatasetMaker
except ImportError:
    UpscalerDatasetMaker = None  # type: ignore

from hrothgar.upscaler.model import UpscalerConfig, UpscalerModel

__all__ = ["UpscalerDatasetMaker", "UpscalerConfig", "UpscalerModel"]
