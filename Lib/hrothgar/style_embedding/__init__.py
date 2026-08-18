"""Font style embedding module.

Provides a ``FontStyleEmbedder`` that renders a fixed set of glyphs in a font
and produces a compact global style embedding.  The embedding is trained with
contrastive and tag-prediction objectives, making it suitable for:

- Similar-font search
- Plagiarism detection
- Conditioning the AR glyph generator with a semantically meaningful style signal
"""

from hrothgar.style_embedding.config import (
    FontStyleEmbedderConfig,
    FontStyleEmbeddingLossWeights,
)
from hrothgar.style_embedding.model import FontStyleEmbedder
from hrothgar.style_embedding.dataset import (
    FontStyleDatasetMaker,
)
from hrothgar.style_embedding.losses import (
    category_loss,
    contrastive_loss,
    tag_prediction_loss,
    compute_losses,
)
from hrothgar.style_embedding.train import FontStyleEmbeddingTrainingLoop

__all__ = [
    "FontStyleEmbedder",
    "FontStyleEmbedderConfig",
    "FontStyleEmbeddingLossWeights",
    "FontStyleEmbeddingTrainingLoop",
    "FontStyleDatasetMaker",
    "category_loss",
    "contrastive_loss",
    "tag_prediction_loss",
    "compute_losses",
]
