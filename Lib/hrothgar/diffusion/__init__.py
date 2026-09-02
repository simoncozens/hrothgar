"""Diffusion glyph generator (VecFusion-style, raster-only).

Phase 1 is a **class-conditional** diffusion model: it renders a grayscale glyph
from a class id that folds codepoint and style-axis position.  It exists to
answer one question fast — *is diffusion capable of reproducing fine style detail
(terminal roundness, serifs, stroke modulation) that the v1–v5 autoencoders could
not?* — before we invest in the exemplar-conditional UNet + cross-attention of
Phase 2 (the many-shot mode that maps onto the real problem).

The diffusion process itself is delegated to
``denoising_diffusion_pytorch.classifier_free_guidance``; this package configures
it, wraps it in a clean facade, and provides the datasets/metrics/training glue.
"""

from hrothgar.diffusion.config import (
    DiffusionConfig,
    DiffusionLossWeights,
)
from hrothgar.diffusion.dataset import (
    ClassConditionalGlyphDataset,
    RONDVocab,
    build_rond_dataset,
    materialize,
)
from hrothgar.diffusion.losses import (
    AxisHead,
    diag_off,
    mean_abs_diff,
    save_montage,
)
from hrothgar.diffusion.model import DiffusionGlyphModel, build_diffusion_model
from hrothgar.diffusion.train import DiffusionTrainer

__all__ = [
    "DiffusionConfig",
    "DiffusionLossWeights",
    "DiffusionGlyphModel",
    "build_diffusion_model",
    "DiffusionTrainer",
    "ClassConditionalGlyphDataset",
    "RONDVocab",
    "build_rond_dataset",
    "materialize",
    "AxisHead",
    "diag_off",
    "mean_abs_diff",
    "save_montage",
]
