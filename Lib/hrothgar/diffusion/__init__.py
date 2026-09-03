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
    ExemplarDiffusionConfig,
    FontIdDiffusionConfig,
)
from hrothgar.diffusion.dataset import (
    ClassConditionalGlyphDataset,
    RONDVocab,
    build_exemplar_rond_data,
    build_fontid_rond_data,
    build_rond_dataset,
    materialize,
)
from hrothgar.diffusion.losses import (
    AxisHead,
    attention_health,
    diag_off,
    mean_abs_diff,
    save_montage,
)
from hrothgar.diffusion.model import DiffusionGlyphModel, build_diffusion_model
from hrothgar.diffusion.exemplar import (
    ExemplarDiffusionModel,
    build_exemplar_model,
)
from hrothgar.diffusion.fontid import (
    FontIdDiffusionModel,
    build_fontid_model,
)
from hrothgar.diffusion.dataset_fontid import FontIdDatasetMaker
from hrothgar.diffusion.train import DiffusionTrainer

__all__ = [
    "DiffusionConfig",
    "DiffusionLossWeights",
    "ExemplarDiffusionConfig",
    "FontIdDiffusionConfig",
    "DiffusionGlyphModel",
    "build_diffusion_model",
    "DiffusionTrainer",
    "ExemplarDiffusionModel",
    "build_exemplar_model",
    "FontIdDiffusionModel",
    "build_fontid_model",
    "FontIdDatasetMaker",
    "ClassConditionalGlyphDataset",
    "RONDVocab",
    "build_rond_dataset",
    "build_exemplar_rond_data",
    "build_fontid_rond_data",
    "materialize",
    "AxisHead",
    "attention_health",
    "diag_off",
    "mean_abs_diff",
    "save_montage",
]
