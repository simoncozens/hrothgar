"""G-Tok (Glyph Tokenizer) model implementation.

This module implements the hybrid CNN-ViT tokenizer described in the GAR-Font paper.
The architecture consists of:
  1. CNN encoder: Downsamples glyph images to feature maps (from LlamaGen)
  2. ViT encoder: 6-layer transformer encoder with 2D sinusoidal position embeddings
  3. Vector quantization: Codebook with 2048 entries and 8-dimensional codes
  4. ViT decoder: 6-layer causal transformer decoder with 2D sinusoidal position embeddings
  5. CNN decoder: Upsamples quantized codes back to glyph images (from LlamaGen)

References:
  - GAR-Font paper: https://arxiv.org/abs/2401.00141
  - LlamaGen tokenizer: https://github.com/FoundationVision/LlamaGen
"""

import json
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torchvision.models.vision_transformer import Encoder, EncoderBlock

from hrothgar.llamagen_cnn import (
    Encoder as CNNEncoder,
    Decoder as CNNDecoder,
    VectorQuantizer,
)
from hrothgar.gtok.losses import GtokLossInfo
from hrothgar.gtok.config import GtokConfig
from hrothgar.utils import SaveLoadModel


class FrozenFlanT5Conditioner(nn.Module):
    """Frozen Flan-T5 text encoder that returns pooled sentence embeddings."""

    def __init__(self, model_name: str, max_length: int = 128):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        try:
            from transformers import AutoTokenizer, T5EncoderModel
        except ImportError as exc:
            raise ImportError(
                "transformers is required for GTok text conditioning. "
                "Install with: pip install transformers sentencepiece"
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.output_dim = int(self.encoder.config.d_model)
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

    def forward(self, descriptions: List[str], device: torch.device) -> torch.Tensor:
        """Encode a batch of descriptions into pooled embeddings.

        Args:
            descriptions: List of length B with text prompts.
            device: Device where the returned tensor should live.

        Returns:
            Tensor of shape (B, output_dim).
        """
        if not descriptions:
            raise ValueError("descriptions must be non-empty")

        encoded = self.tokenizer(
            descriptions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = self.encoder(**encoded)

        # Mean-pool only over non-padding tokens.
        hidden = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * attention_mask).sum(dim=1)
        denom = attention_mask.sum(dim=1).clamp_min(1.0)
        pooled = summed / denom
        return pooled


def create_2d_sinusoidal_position_embeddings(
    sequence_length: int,
    grid_height: int,
    grid_width: int,
    embedding_dim: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Create 2D sinusoidal position embeddings for Vision Transformer.

    This follows the approach in the original Vision Transformer paper and matches
    the official get_2d_sincos_pos_embed implementation exactly.

    Args:
        sequence_length: Total number of tokens (grid_height * grid_width)
        grid_height: Height of the spatial grid
        grid_width: Width of the spatial grid
        embedding_dim: Dimensionality of the embedding
        device: Device to place the embeddings on

    Returns:
        Position embeddings of shape (sequence_length, embedding_dim)
    """
    assert (
        sequence_length == grid_height * grid_width
    ), f"sequence_length ({sequence_length}) must equal grid_height * grid_width ({grid_height * grid_width})"
    assert (
        embedding_dim % 4 == 0
    ), f"embedding_dim ({embedding_dim}) must be divisible by 4"

    # Create 1D sincos embedding helper
    def get_1d_sincos(pos: torch.Tensor, dim: int) -> torch.Tensor:
        # dim is embedding_dim // 2
        omega = torch.arange(dim // 2, dtype=torch.float32, device=device)
        omega = omega / (dim / 2.0)
        omega = 1.0 / (10000.0**omega)

        pos = pos.reshape(-1)
        out = torch.einsum("m,d->md", pos, omega)

        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)

        return torch.cat([emb_sin, emb_cos], dim=1)

    grid_h = torch.arange(grid_height, dtype=torch.float32, device=device)
    grid_w = torch.arange(grid_width, dtype=torch.float32, device=device)

    # In official code: grid[0] is column indices (x_pos), grid[1] is row indices (y_pos)
    # Flat sequence shape: (grid_height * grid_width,)
    y_pos = grid_h.unsqueeze(1).expand(grid_height, grid_width).reshape(-1)
    x_pos = grid_w.unsqueeze(0).expand(grid_height, grid_width).reshape(-1)

    # Official: emb_h = get_1d(..., grid[0]), emb_w = get_1d(..., grid[1])
    # emb = concat([emb_h, emb_w], dim=1)
    emb_x = get_1d_sincos(x_pos, embedding_dim // 2)
    emb_y = get_1d_sincos(y_pos, embedding_dim // 2)

    position_embeddings = torch.cat([emb_x, emb_y], dim=1)
    return position_embeddings


class CausalAttentionMask:
    """Helper class to create causal attention masks for autoregressive decoding."""

    _cache = {}

    @staticmethod
    def get_causal_mask(sequence_length: int, device: torch.device) -> torch.Tensor:
        """Get a causal attention mask (lower triangular matrix).

        Used to prevent the decoder from attending to future tokens during autoregressive
        generation.

        Args:
            sequence_length: Length of the sequence
            device: Device to place the mask on

        Returns:
            Causal mask of shape (sequence_length, sequence_length). Values are 0 for
            valid positions and -inf for masked positions.
        """
        key = (sequence_length, str(device))
        if key not in CausalAttentionMask._cache:
            # Float mask for nn.MultiheadAttention: 0 means attend, -inf means mask.
            mask = torch.zeros(sequence_length, sequence_length, device=device)
            mask = mask.masked_fill(
                torch.triu(
                    torch.ones(sequence_length, sequence_length, device=device),
                    diagonal=1,
                ).bool(),
                float("-inf"),
            )
            CausalAttentionMask._cache[key] = mask
        return CausalAttentionMask._cache[key]


class ViTEncoder(nn.Module):
    """Vision Transformer encoder with 2D sinusoidal position embeddings.

    Takes CNN-encoded feature maps (as a sequence of tokens) and applies self-attention
    to improve the learned representations before quantization.

    Args:
        input_dim: Dimensionality of input tokens (typically the CNN's z_channels)
        hidden_dim: Dimensionality of transformer embeddings
        num_layers: Number of transformer encoder layers
        num_heads: Number of attention heads
        mlp_dim: Dimensionality of feedforward networks
        sequence_length: Total number of tokens in the sequence
        grid_height: Height of the spatial token grid
        grid_width: Width of the spatial token grid
        dropout: Dropout probability
        attention_dropout: Dropout probability in attention layers
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_dim: int,
        sequence_length: int,
        grid_height: int,
        grid_width: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.sequence_length = sequence_length
        self.grid_height = grid_height
        self.grid_width = grid_width

        # Project input tokens to transformer embedding dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # 2D sinusoidal position embeddings
        self.register_buffer(
            "position_embeddings",
            create_2d_sinusoidal_position_embeddings(
                sequence_length, grid_height, grid_width, hidden_dim
            ),
            persistent=False,
        )

        # Learnable class token (optional, follows standard ViT)
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Transformer encoder layers
        self.encoder = Encoder(
            seq_length=sequence_length + 1,  # +1 for class token
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ViT encoder.

        Args:
            x: Input tokens of shape (batch_size, sequence_length, input_dim)

        Returns:
            Encoded tokens of shape (batch_size, sequence_length + 1, hidden_dim),
            where the first token is the class token.
        """
        batch_size, seq_length, _ = x.shape
        assert (
            seq_length == self.sequence_length
        ), f"Input sequence length {seq_length} != expected {self.sequence_length}"

        # Project to embedding dimension
        x = self.input_projection(x)  # (batch, seq_length, hidden_dim)

        # Add position embeddings
        x = x + self.position_embeddings.unsqueeze(0)  # broadcast to batch

        # Expand class tokens for the batch
        class_tokens = self.class_token.expand(
            batch_size, -1, -1
        )  # (batch, 1, hidden_dim)

        # Concatenate class token
        x = torch.cat([class_tokens, x], dim=1)  # (batch, seq_length + 1, hidden_dim)

        # Apply dropout
        x = self.dropout(x)

        # Apply transformer encoder
        x = self.encoder(x)

        return x


class CausalViTDecoder(nn.Module):
    """Causal Vision Transformer decoder for autoregressive token generation.

    Uses causal attention masking to ensure that each token can only attend to
    previously generated tokens, enabling autoregressive decoding.

    Args:
        hidden_dim: Dimensionality of transformer embeddings
        num_layers: Number of transformer decoder layers
        num_heads: Number of attention heads
        mlp_dim: Dimensionality of feedforward networks
        output_dim: Dimensionality of output tokens (typically quantizer_code_dim)
        sequence_length: Total number of tokens in the sequence
        grid_height: Height of the spatial token grid
        grid_width: Width of the spatial token grid
        dropout: Dropout probability
        attention_dropout: Dropout probability in attention layers
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_dim: int,
        output_dim: int,
        sequence_length: int,
        grid_height: int,
        grid_width: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.grid_height = grid_height
        self.grid_width = grid_width

        # 2D sinusoidal position embeddings
        self.register_buffer(
            "position_embeddings",
            create_2d_sinusoidal_position_embeddings(
                sequence_length, grid_height, grid_width, hidden_dim
            ),
            persistent=False,
        )

        # Build causal transformer layers manually to have control over masking
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        # Output projection from hidden_dim back to output_dim
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the causal ViT decoder.

        Args:
            x: Input tokens of shape (batch_size, sequence_length, hidden_dim)
            encoder_output: Optional encoder output for cross-attention (not used in base implementation)

        Returns:
            Decoded tokens of shape (batch_size, sequence_length, output_dim)
        """
        _batch_size, seq_length, _hidden_dim = x.shape
        assert (
            seq_length == self.sequence_length
        ), f"Input sequence length {seq_length} != expected {self.sequence_length}"

        # Add position embeddings
        x = x + self.position_embeddings.unsqueeze(0)  # broadcast to batch
        x = self.dropout(x)

        # Create causal attention mask
        causal_mask = CausalAttentionMask.get_causal_mask(seq_length, x.device)

        # Apply transformer decoder layers with causal masking
        for layer in self.layers:
            # Self-attention with causal mask
            h = layer.ln_1(x)
            attn_output, _ = layer.self_attention(
                h,
                h,
                h,
                need_weights=False,
                attn_mask=causal_mask,
            )
            x = x + layer.dropout(attn_output)

            # MLP
            x = x + layer.mlp(layer.ln_2(x))

        # Final layer norm
        x = self.layer_norm(x)

        # Project to output dimension
        x = self.output_projection(x)

        return x


class GtokModel(SaveLoadModel):
    """G-Tok: Hybrid CNN-ViT glyph tokenizer.

    A complete tokenizer for glyphs that:
    1. Encodes glyph images to feature maps with a CNN
    2. Applies self-attention with a ViT encoder
    3. Quantizes the representations to discrete codes
    4. Decodes using a causal ViT decoder
    5. Reconstructs glyph images with a CNN decoder

    This architecture enables learning compressed, quantized representations of glyphs
    that can be used by autoregressive generators.
    """

    def __init__(self, config: GtokConfig):
        """Initialize the G-Tok tokenizer.

        Args:
            config: GtokConfig object specifying all hyperparameters
        """
        super().__init__()
        self.config = config

        # Calculate derived parameters from the CNN pyramid.
        # Encoder downsamples at every level except the final one.
        assert (
            config.cnn_channel_multipliers is not None
        ), "this can't happen, we set it in __post_init__, but mypy doesn't know that"
        self.num_downsampling_phases = len(config.cnn_channel_multipliers) - 1
        self.downsampling_factor = 2**self.num_downsampling_phases
        if config.image_size % self.downsampling_factor != 0:
            raise ValueError(
                "image_size must be divisible by the CNN downsampling factor "
                f"(got image_size={config.image_size}, downsampling_factor={self.downsampling_factor})"
            )
        self.token_grid_height = config.image_size // self.downsampling_factor
        self.token_grid_width = config.image_size // self.downsampling_factor
        self.sequence_length = self.token_grid_height * self.token_grid_width

        self.text_conditioner: Optional[FrozenFlanT5Conditioner] = None
        if config.text_conditioning_model_name:
            self.text_conditioner = FrozenFlanT5Conditioner(
                config.text_conditioning_model_name,
                max_length=config.text_conditioning_max_length,
            )

        # CNN Encoder: Downsamples images to feature maps
        self.cnn_encoder = CNNEncoder(
            in_channels=3,
            ch=config.cnn_base_channels,
            ch_mult=tuple(config.cnn_channel_multipliers),
            num_res_blocks=config.cnn_num_residual_blocks,
            z_channels=config.cnn_latent_channels,
            dropout=config.cnn_dropout,
        )

        # ViT Encoder: self-attention over the CNN token grid.

        self.vit_encoder = ViTEncoder(
            input_dim=config.cnn_latent_channels,
            hidden_dim=config.vit_hidden_dim,
            num_layers=config.vit_num_layers,
            num_heads=config.vit_num_heads,
            mlp_dim=config.vit_mlp_dim,
            sequence_length=self.sequence_length,
            grid_height=self.token_grid_height,
            grid_width=self.token_grid_width,
            dropout=config.vit_dropout,
            attention_dropout=config.vit_attention_dropout,
        )

        # Projection to quantizer input
        self.vit_encoder_to_quantizer = nn.Linear(
            config.vit_hidden_dim, config.quantizer_code_dim
        )

        # Project text embedding dim to ViT hidden dim before affine modulation.
        text_embedding_dim = (
            self.text_conditioner.output_dim
            if self.text_conditioner is not None
            else config.vit_hidden_dim
        )
        self.encoder_text_projection = nn.Linear(
            text_embedding_dim,
            config.vit_hidden_dim,
        )
        self.encoder_text_affine = nn.Linear(
            config.vit_hidden_dim,
            config.vit_hidden_dim * 2,
        )

        # Vector Quantizer: Codebook with 2048 entries and 8-dim codes
        self.quantizer = VectorQuantizer(
            codebook_size=config.quantizer_codebook_size,
            codebook_dimensions=config.quantizer_code_dim,
            beta=config.quantizer_beta,
            entropy_loss_ratio=config.quantizer_entropy_loss_ratio,
            l2_norm=True,
            show_usage=True,
            ema_decay=0.99,
        )

        # Projection from quantizer to ViT decoder input
        self.quantizer_to_vit_decoder = nn.Linear(
            config.quantizer_code_dim, config.vit_hidden_dim
        )

        # Independent decoder-side text projection and affine modulation.
        self.decoder_text_projection = nn.Linear(
            text_embedding_dim,
            config.vit_hidden_dim,
        )
        self.decoder_text_affine = nn.Linear(
            config.vit_hidden_dim,
            config.vit_hidden_dim * 2,
        )

        # ViT Decoder: 6-layer causal transformer for autoregressive decoding
        self.vit_decoder = CausalViTDecoder(
            hidden_dim=config.vit_hidden_dim,
            num_layers=config.vit_num_layers,
            num_heads=config.vit_num_heads,
            mlp_dim=config.vit_mlp_dim,
            output_dim=config.cnn_latent_channels,
            sequence_length=self.sequence_length,
            grid_height=self.token_grid_height,
            grid_width=self.token_grid_width,
            dropout=config.vit_dropout,
            attention_dropout=config.vit_attention_dropout,
        )

        # CNN Decoder: Upsamples feature maps back to images
        self.cnn_decoder = CNNDecoder(
            z_channels=config.cnn_latent_channels,
            ch=config.cnn_base_channels,
            ch_mult=tuple(config.cnn_channel_multipliers),
            num_res_blocks=config.cnn_num_residual_blocks,
            out_channels=3,
            dropout=config.cnn_dropout,
        )

    def _description_embeddings(
        self,
        descriptions: Optional[List[str]],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if self.text_conditioner is None or descriptions is None:
            return None
        if len(descriptions) != batch_size:
            raise ValueError(
                f"description count must match batch size (got {len(descriptions)} vs {batch_size})"
            )
        return self.text_conditioner(descriptions, device=device)

    @staticmethod
    def _apply_feature_affine(
        token_features: torch.Tensor,
        text_embeddings: Optional[torch.Tensor],
        projection: nn.Linear,
        affine: nn.Linear,
    ) -> torch.Tensor:
        """Apply per-feature affine modulation: y = x * (1 + gamma) + beta."""
        if text_embeddings is None:
            return token_features
        conditioned = projection(text_embeddings)
        gamma_beta = affine(conditioned)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return token_features * (1.0 + gamma) + beta

    def encode(
        self,
        images: torch.Tensor,
        descriptions: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, GtokLossInfo]:
        """Encode glyph images to quantized codes and compute VQ losses.

        Args:
            images: Input glyph images of shape (batch_size, 3, image_size, image_size)

        Returns:
            Tuple of:
                - quantized: Quantized representations, shape (batch_size, sequence_length, code_dim)
                - loss_info: GtokLossInfo tuple containing VQ loss components and metrics
        """
        # CNN encode to a spatial token grid.
        cnn_out = self.cnn_encoder(images)

        # Reshape CNN output to sequence format
        batch_size, channels, height, width = cnn_out.shape
        cnn_seq = cnn_out.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, channels
        )

        # ViT encode (includes class token in output, we keep full output for now)
        vit_out = self.vit_encoder(
            cnn_seq
        )  # (batch, sequence_length + 1, vit_hidden_dim)
        # Remove class token for quantization
        vit_out = vit_out[:, 1:, :]  # (batch, sequence_length, vit_hidden_dim)

        text_embeddings = self._description_embeddings(
            descriptions,
            batch_size=batch_size,
            device=images.device,
        )

        # Modulate encoder token features before quantization.
        vit_out = self._apply_feature_affine(
            vit_out,
            text_embeddings,
            self.encoder_text_projection,
            self.encoder_text_affine,
        )

        # Project to quantizer input
        quant_in = self.vit_encoder_to_quantizer(
            vit_out
        )  # (batch, sequence_length, code_dim)

        # Reshape for quantizer while preserving 2D token layout.
        batch_size, _seq_length, _code_dim = quant_in.shape
        quant_in_4d = quant_in.reshape(
            batch_size,
            self.token_grid_height,
            self.token_grid_width,
            self.config.quantizer_code_dim,
        ).permute(0, 3, 1, 2)

        # Quantize
        quantized_4d, raw_loss_info, indices_info = self.quantizer(quant_in_4d)

        # Reshape back to sequence format.
        quantized = quantized_4d.permute(0, 2, 3, 1).reshape(
            batch_size, self.sequence_length, self.config.quantizer_code_dim
        )
        loss_info = GtokLossInfo(
            vq_loss=raw_loss_info[0],
            commit_loss=raw_loss_info[1],
            entropy_loss=raw_loss_info[2],
            codebook_usage=raw_loss_info[3],
            perplexity=indices_info[0],
        )

        return quantized, loss_info

    def decode(
        self,
        quantized: torch.Tensor,
        descriptions: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Decode quantized codes back to glyph images.

        Args:
            quantized: Quantized representations of shape (batch_size, sequence_length, code_dim)

        Returns:
            Reconstructed glyph images of shape (batch_size, 3, image_size, image_size)
        """
        batch_size = quantized.shape[0]
        text_embeddings = self._description_embeddings(
            descriptions,
            batch_size=batch_size,
            device=quantized.device,
        )

        # Project to ViT decoder input
        decoder_in = self.quantizer_to_vit_decoder(
            quantized
        )  # (batch, sequence_length, vit_hidden_dim)

        # Modulate decoder token stream with the same text embedding.
        decoder_in = self._apply_feature_affine(
            decoder_in,
            text_embeddings,
            self.decoder_text_projection,
            self.decoder_text_affine,
        )

        # ViT decode
        vit_out = self.vit_decoder(
            decoder_in
        )  # (batch, sequence_length, cnn_latent_channels)

        # Reshape to 4D for CNN decoder
        batch_size, _seq_length, channels = vit_out.shape
        cnn_in = vit_out.reshape(
            batch_size,
            self.token_grid_height,
            self.token_grid_width,
            channels,
        ).permute(0, 3, 1, 2)

        # CNN decode
        images = self.cnn_decoder(cnn_in)

        return images

    def forward(
        self,
        images: torch.Tensor,
        descriptions: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, GtokLossInfo]:
        """Forward pass: encode, quantize, and decode.

        Args:
            images: Input glyph images of shape (batch_size, 3, image_size, image_size)

        Returns:
            Tuple of:
                - reconstructed: Reconstructed images of shape (batch_size, 3, image_size, image_size)
                - loss_info: VQ loss information tuple
        """
        quantized, loss_info = self.encode(images, descriptions=descriptions)
        reconstructed = self.decode(quantized, descriptions=descriptions)
        return reconstructed, loss_info


def load_model(model_path: Path, device: torch.device) -> tuple[GtokModel, GtokConfig]:
    """Load GtokModel from weights and its sidecar config JSON.

    Raises ``FileNotFoundError`` if either the weights or the sidecar are missing.
    """
    config_path = model_path.with_suffix(".conf.json")
    if not config_path.exists():
        raise FileNotFoundError(
            f"Sidecar config not found: {config_path}\n"
            "Run GTok training first so the .conf.json is written alongside the .pth."
        )
    with config_path.open("r", encoding="utf-8") as fh:
        config_dict = json.load(fh)
    config = GtokConfig(**config_dict)

    model = GtokModel(config).to(device)
    model.load(str(model_path), device=device)
    model.eval()
    return model, config
