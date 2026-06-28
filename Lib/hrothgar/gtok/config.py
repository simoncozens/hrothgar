import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from hrothgar.dataset import LGC_ALL


@dataclass
class GtokConfig:
    """Configuration for the G-Tok tokenizer."""

    image_size: int = 128
    character_set: List[int] = field(default_factory=lambda: list(LGC_ALL))

    # CNN encoder/decoder parameters (from LlamaGen)
    cnn_base_channels: int = 128
    cnn_num_residual_blocks: int = 2
    cnn_latent_channels: int = 256
    cnn_dropout: float = 0.0

    # ViT parameters
    vit_hidden_dim: int = 384  # Dimensionality of transformer embeddings
    vit_num_layers: int = 6  # Number of transformer layers
    vit_num_heads: int = 8  # Number of attention heads
    vit_mlp_dim: int = 512  # Dimensionality of feedforward networks
    vit_dropout: float = 0.0
    vit_attention_dropout: float = 0.0

    # Quantization parameters
    quantizer_codebook_size: int = 2048  # Size of the codebook
    quantizer_beta: float = 0.5  # Commitment loss weight
    quantizer_entropy_loss_ratio: float = 0.2  # Entropy regularization weight
    quantizer_ema_decay: float = 0.97

    def __post_init__(self):
        """Set defaults for list parameters."""
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.vit_hidden_dim % self.vit_num_heads != 0:
            raise ValueError(
                "vit_hidden_dim must be divisible by vit_num_heads "
                f"(got vit_hidden_dim={self.vit_hidden_dim}, vit_num_heads={self.vit_num_heads})"
            )
        assert self.cnn_channel_multipliers is not None

    @property
    def quantizer_code_dim(self) -> int:
        """Dimensionality of each code in the quantizer."""
        if self.image_size == 128:
            return 64
        elif self.image_size == 64:
            return 32
        else:
            raise ValueError(
                f"Unsupported image_size {self.image_size} for default quantizer_code_dim"
            )

    def save_sidecar(self, model_path: Path) -> None:
        from dataclasses import asdict

        config_path = Path(str(model_path).replace(".pth", ".conf.json"))
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"Saved GTok config to {config_path}")

    @property
    def cnn_channel_multipliers(self) -> Optional[List[int]]:
        if self.image_size == 128:
            return [1, 1, 2, 2, 4]
        elif self.image_size == 64:
            return [1, 1, 2, 4]
        else:
            raise ValueError(
                f"Unsupported image_size {self.image_size} for default cnn_channel_multipliers"
            )


@dataclass(frozen=True)
class GtokLossWeights:
    """Weights applied to each loss term in ``compute_gtok_loss``.

    Commit and entropy losses are already scaled by
    ``GtokConfig.quantizer_beta`` and
    ``GtokConfig.quantizer_entropy_loss_ratio`` inside the
    ``VectorQuantizer``, so no additional weights are needed here.
    The VQ loss is always zero with EMA codebook updates.
    """

    l1: float = 1.0
    perceptual: float = 0.1
    edge: float = 2.0
    vq: float = 1.0
    commit: float = 0.5
    entropy: float = 2.0
    aux_ar: float = 0.01
    character_ce: float = 0.5
    font_ce: float = 1.0
