"""Style extraction — a *rich, complete* font-style representation.

Unlike :mod:`hrothgar.style_embedding` (a compact contrastive embedding for
similarity/recommendation), this module learns an autoencoding style
representation whose contract is **reconstructability**: the embedding must be
decodable back into high-fidelity glyphs.  Reconstruction (L1 + glyphloss +
LPIPS, optionally adversarial) is both the training objective and the
acceptance signal.
"""

from hrothgar.style_extraction.config import (
    StyleExtractionConfig,
    StyleExtractionLossWeights,
)
from hrothgar.style_extraction.dataset import StyleExtractionDatasetMaker
from hrothgar.style_extraction.losses import (
    NLayerDiscriminator,
    adversarial_discriminator_loss,
    adversarial_generator_loss,
    ink_coverage_loss,
    reconstruction_loss,
)
from hrothgar.style_extraction.model import StyleExtractionModel
from hrothgar.style_extraction.train import StyleExtractionTrainingLoop

__all__ = [
    "StyleExtractionModel",
    "StyleExtractionConfig",
    "StyleExtractionLossWeights",
    "StyleExtractionTrainingLoop",
    "StyleExtractionDatasetMaker",
    "NLayerDiscriminator",
    "reconstruction_loss",
    "ink_coverage_loss",
    "adversarial_generator_loss",
    "adversarial_discriminator_loss",
]
