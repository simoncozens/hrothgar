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

import dataclasses
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
from hrothgar.upstream.tokenizer import ViTEncoder as UpstreamViTEncoder
from hrothgar.upstream.tokenizer import ViTDecoder as UpstreamViTDecoder
from hrothgar.utils import SaveLoadModel


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
        # We follow the upstream GAR-Font layout: Conv2d patch projection
        # before the ViT encoder (no class token, no internal projection).
        self.proj_patch = nn.Conv2d(
            config.cnn_latent_channels, config.vit_hidden_dim, kernel_size=1
        )
        self.vit_encoder = UpstreamViTEncoder(
            patch_num=self.token_grid_height,
            dim=config.vit_hidden_dim,
            depth=config.vit_num_layers,
            heads=config.vit_num_heads,
            mlp_dim=config.vit_mlp_dim,
            dim_head=config.vit_hidden_dim // config.vit_num_heads,
        )

        # Project ViT output to quantizer input dimensions.
        self.vit_encoder_to_quantizer = nn.Linear(
            config.vit_hidden_dim, config.quantizer_code_dim
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

        # ViT Decoder: causal transformer (upstream GAR-Font implementation).
        self.vit_decoder = UpstreamViTDecoder(
            patch_num=self.token_grid_height,
            dim=config.vit_hidden_dim,
            depth=config.vit_num_layers,
            heads=config.vit_num_heads,
            mlp_dim=config.vit_mlp_dim,
            dim_head=config.vit_hidden_dim // config.vit_num_heads,
        )
        # Reverse patch projection: ViT output → CNN feature channels.
        self.proj_unpatch = nn.Conv2d(
            config.vit_hidden_dim, config.cnn_latent_channels, kernel_size=1
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

    def encode(
        self,
        images: torch.Tensor,
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

        # Patch projection: Conv2d maps CNN channels → ViT hidden dim.
        tokens = (
            self.proj_patch(cnn_out).flatten(2).transpose(1, 2)
        )  # (B, N, vit_hidden_dim)

        # ViT encode (no class token — upstream layout).
        vit_out = self.vit_encoder(tokens)  # (B, N, vit_hidden_dim)

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
    ) -> torch.Tensor:
        """Decode quantized codes back to glyph images.

        Args:
            quantized: Quantized representations of shape (batch_size, sequence_length, code_dim)

        Returns:
            Reconstructed glyph images of shape (batch_size, 3, image_size, image_size)
        """
        batch_size = quantized.shape[0]
        # Project to ViT decoder input
        decoder_in = self.quantizer_to_vit_decoder(quantized)  # (B, N, vit_hidden_dim)

        # ViT decode (upstream causal transformer, same-dim in/out).
        vit_out = self.vit_decoder(decoder_in)  # (B, N, vit_hidden_dim)

        # Reshape to 4D, then reverse patch projection: ViT dim → CNN channels.
        vit_out_4d = vit_out.transpose(1, 2).reshape(
            batch_size,
            -1,
            self.token_grid_height,
            self.token_grid_width,
        )  # (B, vit_hidden_dim, H, W)
        cnn_in = self.proj_unpatch(vit_out_4d)  # (B, cnn_latent_channels, H, W)

        # CNN decode
        images = self.cnn_decoder(cnn_in)

        return images

    def forward(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, GtokLossInfo]:
        """Forward pass: encode, quantize, and decode.

        Args:
            images: Input glyph images of shape (batch_size, 3, image_size, image_size)

        Returns:
            Tuple of:
                - reconstructed: Reconstructed images of shape (batch_size, 3, image_size, image_size)
                - loss_info: VQ loss information tuple
        """
        quantized, loss_info = self.encode(images)
        reconstructed = self.decode(quantized)
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
    # Drop keys not recognised by the current GtokConfig (backward compat).
    valid_keys = {
        f.name for f in dataclasses.fields(GtokConfig)
    }
    filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
    config = GtokConfig(**filtered)

    model = GtokModel(config).to(device)
    model.load(str(model_path), device=device)
    model.eval()
    return model, config
