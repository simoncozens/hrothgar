from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Optional


@dataclass
class GtokConfig:
    """Configuration for the G-Tok tokenizer."""

    image_size: int = 128

    # CNN encoder/decoder parameters (from LlamaGen)
    cnn_base_channels: int = 128
    cnn_num_residual_blocks: int = 2
    cnn_latent_channels: int = 256
    cnn_dropout: float = 0.0

    # ViT parameters
    vit_hidden_dim: int = 384  # Dimensionality of transformer embeddings
    vit_num_layers: int = 6  # Number of transformer layers
    vit_num_heads: int = 6  # Number of attention heads
    vit_mlp_dim: int = (
        1536  # Dimensionality of feedforward networks (typically 4x hidden_dim)
    )
    vit_dropout: float = 0.1
    vit_attention_dropout: float = 0.1

    # Quantization parameters
    quantizer_codebook_size: int = 4096  # Size of the codebook
    quantizer_code_dim: int = 8  # Dimensionality of each code
    quantizer_beta: float = 0.25  # Commitment loss weight
    quantizer_entropy_loss_ratio: float = 0.01  # Entropy regularization weight

    # Optional text conditioning via a frozen Flan-T5 encoder.
    text_conditioning_model_name: Optional[str] = "google/flan-t5-small"
    text_conditioning_max_length: int = 128

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
    """Weights applied to each loss term in ``compute_gtok_loss``."""

    l1: float = 1.0
    perceptual: float = 0.1
    edge: float = 2.0
    vq: float = 1.0
    commit: float = 1.0
    entropy: float = 1.0
