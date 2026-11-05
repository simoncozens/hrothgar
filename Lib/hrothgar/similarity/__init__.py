"""Font style embedding module.

Provides a ``FontStyleEmbedder`` that renders a fixed set of glyphs in a font
and produces a compact global style embedding.  The embedding is trained with
contrastive and tag-prediction objectives, making it suitable for:

- Similar-font search
- Plagiarism detection
- Conditioning the AR glyph generator with a semantically meaningful style signal
"""

from hrothgar.similarity.config import (
    FontStyleEmbedderConfig,
    FontStyleEmbeddingLossWeights,
)
from hrothgar.similarity.dataset import FontStyleDatasetMaker
from hrothgar.similarity.losses import (
    category_loss,
    compute_losses,
    contrastive_loss,
    tag_prediction_loss,
)
from hrothgar.similarity.model import FontStyleEmbedder
from hrothgar.similarity.train import FontStyleEmbeddingTrainingLoop

__all__ = [
    "FontStyleDatasetMaker",
    "FontStyleEmbedder",
    "FontStyleEmbedderConfig",
    "FontStyleEmbeddingLossWeights",
    "FontStyleEmbeddingTrainingLoop",
    "category_loss",
    "compute_losses",
    "contrastive_loss",
    "tag_prediction_loss",
]
