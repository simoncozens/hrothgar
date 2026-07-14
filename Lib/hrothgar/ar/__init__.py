"""MaskGIT glyph generator package.

In training mode (torch available), re-exports all public symbols.
In inference mode (no torch), imports only what's needed without errors.
"""

# Lazy imports — only succeed if the dependencies are available.
try:
    from hrothgar.ar.dataset import ARPhase1DatasetMaker
except ImportError:
    ARPhase1DatasetMaker = None  # type: ignore

try:
    from hrothgar.ar.losses import ARLossWeights, compute_ar_loss
except ImportError:
    ARLossWeights = None  # type: ignore
    compute_ar_loss = None  # type: ignore

try:
    from hrothgar.ar.model import ARModel, ARModelOutput
except ImportError:
    ARModel = None  # type: ignore
    ARModelOutput = None  # type: ignore

from hrothgar.ar.config import ARModelConfig

try:
    from hrothgar.ar.maskgit import (
        MaskGITConfig,
        MaskGITDecoder,
        MaskGITLossWeights,
        MaskGITTransformer,
        compute_maskgit_loss,
    )
except ImportError:
    MaskGITConfig = None  # type: ignore
    MaskGITDecoder = None  # type: ignore
    MaskGITLossWeights = None  # type: ignore
    MaskGITTransformer = None  # type: ignore
    compute_maskgit_loss = None  # type: ignore

try:
    from hrothgar.ar.train import MaskGITTrainingLoop
except ImportError:
    MaskGITTrainingLoop = None  # type: ignore

__all__ = [
    "ARPhase1DatasetMaker",
    "ARLossWeights",
    "compute_ar_loss",
    "ARModel",
    "ARModelConfig",
    "ARModelOutput",
    "MaskGITTrainingLoop",
    "MaskGITConfig",
    "MaskGITDecoder",
    "MaskGITLossWeights",
    "MaskGITTransformer",
    "compute_maskgit_loss",
]
