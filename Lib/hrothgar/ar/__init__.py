"""Autoregressive generator package exports."""

from hrothgar.ar.dataset import ARPhase1DatasetMaker
from hrothgar.ar.losses import (
    ARAdaptationLossWeights,
    ARLossWeights,
    compute_ar_adaptation_loss,
    compute_ar_loss,
)
from hrothgar.ar.model import (
    ARAdaptationOutput,
    ARModel,
    ARModelConfig,
    ARModelOutput,
    LoRAConfig,
    LoRALinear,
)
from hrothgar.ar.multimodal import (
    HashedDescriptionEncoder,
    HashedDescriptionEncoderConfig,
    TextStyleAdapter,
    TextStyleAdapterConfig,
)
from hrothgar.ar.nfa import ARNFATrainingLoop, NFADatasetMaker, NFAGlyphDataset
from hrothgar.ar.train import ARMultimodalTrainingLoop, ARVisualTrainingLoop

__all__ = [
    "ARPhase1DatasetMaker",
    "ARAdaptationLossWeights",
    "ARLossWeights",
    "compute_ar_adaptation_loss",
    "compute_ar_loss",
    "ARAdaptationOutput",
    "ARModel",
    "ARModelConfig",
    "ARModelOutput",
    "LoRAConfig",
    "LoRALinear",
    "HashedDescriptionEncoder",
    "HashedDescriptionEncoderConfig",
    "TextStyleAdapter",
    "TextStyleAdapterConfig",
    "ARNFATrainingLoop",
    "ARMultimodalTrainingLoop",
    "NFADatasetMaker",
    "NFAGlyphDataset",
    "ARVisualTrainingLoop",
]
