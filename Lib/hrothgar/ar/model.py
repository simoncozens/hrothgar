"""Autoregressive glyph generator for GAR-Font.

This module implements the vision-only stage of the GAR-Font AR generator:

1. A content encoder extracts structural features from a reference glyph.
2. A lightweight style encoder extracts visual style features from style references.
3. A content-style aggregator fuses both streams with stacked cross-attention,
   following the design used in FsFont and referenced by GAR-Font.
4. A causal Transformer decoder autoregressively predicts G-Tok codebook indices.
5. A soft codebook projection feeds the frozen G-Tok decoder to reconstruct images.

The training loop is intentionally left for later; this file focuses on model
definition and the tensors needed for token- and pixel-level supervision.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hrothgar.llamagen_cnn import Encoder as CNNEncoder
from hrothgar.gtok.model import (
    GtokConfig,
    GtokModel,
    create_2d_sinusoidal_position_embeddings,
)
from hrothgar.utils import SaveLoadModel
from tqdm import tqdm


@dataclass
class ARModelConfig:
    """Configuration for the visual-pretraining AR generator."""

    image_size: int = 128
    encoder_feature_dim: int = 256

    content_encoder_base_channels: int = 128
    content_encoder_channel_multipliers: tuple[int, ...] = (1, 2, 2, 4, 4)
    content_encoder_num_residual_blocks: int = 2

    style_encoder_base_channels: int = 32
    style_encoder_channel_multipliers: tuple[int, ...] = (1, 2, 2, 4, 4)
    style_encoder_num_residual_blocks: int = 2

    aggregator_num_layers: int = 3
    aggregator_num_heads: int = 8

    decoder_hidden_dim: int = 1024
    decoder_num_layers: int = 24
    decoder_num_heads: int = 16
    decoder_mlp_dim: int = 2048
    decoder_dropout: float = 0.1
    decoder_attention_dropout: float = 0.1

    freeze_gtok: bool = True

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.encoder_feature_dim % self.aggregator_num_heads != 0:
            raise ValueError(
                "encoder_feature_dim must be divisible by aggregator_num_heads "
                f"(got {self.encoder_feature_dim} and {self.aggregator_num_heads})"
            )
        if self.decoder_hidden_dim % self.decoder_num_heads != 0:
            raise ValueError(
                "decoder_hidden_dim must be divisible by decoder_num_heads "
                f"(got {self.decoder_hidden_dim} and {self.decoder_num_heads})"
            )


@dataclass
class ARModelOutput:
    """Outputs returned by ``ARModel.forward``."""

    logits: torch.Tensor
    reconstructed_images: torch.Tensor
    soft_token_embeddings: torch.Tensor
    conditioning_tokens: torch.Tensor
    target_token_indices: Optional[torch.Tensor]


@dataclass
class ARAdaptationOutput:
    """Outputs returned by multimodal adaptation methods.

    These tensors support both adaptation objectives:
    - Optional token/pixel decoding branch (same as visual pretraining path)
    - Feature-space alignment branch between visual-only and multimodal aggregation
    """

    multimodal_conditioning_tokens: torch.Tensor
    visual_conditioning_tokens: torch.Tensor
    multimodal_aggregated_style_tokens: torch.Tensor
    visual_aggregated_style_tokens: torch.Tensor
    logits: Optional[torch.Tensor]
    reconstructed_images: Optional[torch.Tensor]
    soft_token_embeddings: Optional[torch.Tensor]
    target_token_indices: Optional[torch.Tensor]


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation layers in the AR decoder.

    Attributes:
        rank: Rank of the low-rank decomposition. Smaller ranks use less memory;
            typical values are 4–64.
        alpha: Scaling factor for the LoRA output. The effective scale applied
            is ``alpha / rank``, keeping learning dynamics stable across ranks.
    """

    rank: int = 16
    alpha: float = 16.0

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.rank}")
        if self.alpha <= 0.0:
            raise ValueError(f"LoRA alpha must be positive, got {self.alpha}")


class LoRALinear(nn.Module):
    """LoRA-adapted linear layer.

    Wraps an existing ``nn.Linear`` with low-rank adaptation matrices A ∈
    R^{rank×in_features} and B ∈ R^{out_features×rank}.  The forward pass
    computes ``base(x) + (alpha/rank) * x @ A.T @ B.T``.

    The base weights are frozen; only A and B are trainable.  B is zero-
    initialised so the adapter has no effect at the start of fine-tuning.
    """

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base_linear
        for param in self.base.parameters():
            param.requires_grad = False

        base_weight = base_linear.weight
        self.lora_A = nn.Parameter(
            torch.empty(
                rank,
                base_linear.in_features,
                device=base_weight.device,
                dtype=base_weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                base_linear.out_features,
                rank,
                device=base_weight.device,
                dtype=base_weight.dtype,
            )
        )
        self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def in_features(self) -> int:
        """Input feature dimension, delegated to the base layer."""
        return self.base.in_features

    @property
    def out_features(self) -> int:
        """Output feature dimension, delegated to the base layer."""
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:
        """Merged weight matrix, required by nn.MultiheadAttention's out_proj access pattern.

        Returns ``base.weight + (lora_B @ lora_A) * scaling``.  Gradients flow
        only through the LoRA matrices because base.weight is frozen.
        """
        return self.base.weight + (self.lora_B @ self.lora_A) * self.scaling

    @property
    def bias(self) -> Optional[torch.Tensor]:
        """Bias delegated to the base layer (may be None)."""
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


class ComposedLoRALinear(nn.Module):
    """Linear layer with two stacked LoRA adapters: a frozen glyph adapter and a trainable font adapter.

    At forward time the effective weight update is the sum of both deltas::

        ΔW = scale_g * (B_g @ A_g) + scale_f * (B_f @ A_f)

    The glyph adapter (subscript ``_glyph``) is loaded from a pre-trained GA
    checkpoint and kept frozen throughout NFA.  The font adapter (subscript
    ``_font``) is zero-initialised and trained during NFA.

    Both adapters share the same rank and alpha (from *lora_config*).  The
    glyph tensors are copied from *glyph_lora_A* / *glyph_lora_B* and
    registered as non-trainable ``nn.Parameter`` so they travel with the
    module on ``.to(device)`` calls and appear in ``state_dict``.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        glyph_lora_A: torch.Tensor,
        glyph_lora_B: torch.Tensor,
        glyph_scaling: float,
        font_rank: int,
        font_alpha: float,
    ) -> None:
        super().__init__()
        self.base = base_linear
        for param in self.base.parameters():
            param.requires_grad = False

        device = base_linear.weight.device
        dtype = base_linear.weight.dtype

        # Frozen glyph adapter — non-trainable parameters so they are saved in
        # state_dict and moved with .to() without extra bookkeeping.
        self.lora_A_glyph = nn.Parameter(
            glyph_lora_A.to(device=device, dtype=dtype), requires_grad=False
        )
        self.lora_B_glyph = nn.Parameter(
            glyph_lora_B.to(device=device, dtype=dtype), requires_grad=False
        )
        self.scaling_glyph = glyph_scaling
        self.glyph_weight = 1.0

        # Trainable font adapter — zero-initialised so NFA starts from the
        # glyph prior with no additional delta.
        self.lora_A_font = nn.Parameter(
            torch.empty(font_rank, base_linear.in_features, device=device, dtype=dtype)
        )
        self.lora_B_font = nn.Parameter(
            torch.zeros(base_linear.out_features, font_rank, device=device, dtype=dtype)
        )
        self.scaling_font = font_alpha / font_rank
        self.font_weight = 1.0
        nn.init.kaiming_uniform_(self.lora_A_font, a=math.sqrt(5))

    def set_adapter_weights(self, glyph_weight: float, font_weight: float) -> None:
        """Set scalar multipliers for glyph and font adapter deltas."""
        if glyph_weight < 0.0:
            raise ValueError(f"glyph_weight must be non-negative, got {glyph_weight}")
        if font_weight < 0.0:
            raise ValueError(f"font_weight must be non-negative, got {font_weight}")
        self.glyph_weight = float(glyph_weight)
        self.font_weight = float(font_weight)

    @property
    def in_features(self) -> int:
        """Input feature dimension, delegated to the base layer."""
        return self.base.in_features

    @property
    def out_features(self) -> int:
        """Output feature dimension, delegated to the base layer."""
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:
        """Merged weight matrix including both adapter deltas.

        Required by ``nn.MultiheadAttention``'s ``out_proj`` access pattern.
        """
        return (
            self.base.weight
            + (self.lora_B_glyph @ self.lora_A_glyph)
            * self.scaling_glyph
            * self.glyph_weight
            + (self.lora_B_font @ self.lora_A_font)
            * self.scaling_font
            * self.font_weight
        )

    @property
    def bias(self) -> Optional[torch.Tensor]:
        """Bias delegated to the base layer (may be None)."""
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        glyph_delta = (
            x @ self.lora_A_glyph.T @ self.lora_B_glyph.T
        ) * self.scaling_glyph * self.glyph_weight
        font_delta = (
            x @ self.lora_A_font.T @ self.lora_B_font.T
        ) * self.scaling_font * self.font_weight
        return self.base(x) + glyph_delta + font_delta


class ContentStyleAttentionLayer(nn.Module):
    """FsFont-style cross-attention layer for content-style fusion.

    This block mirrors the reference implementation closely: a projected content
    query attends over projected style keys/values, followed by a learned output
    projection, residual connection, and layer normalization. There is no MLP in
    this block, which also matches the reported parameter count of about 0.79M
    for three 256-channel layers.
    """

    def __init__(self, feature_dim: int, num_heads: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        self.key_projection = nn.Linear(feature_dim, feature_dim, bias=False)
        self.value_projection = nn.Linear(feature_dim, feature_dim, bias=False)
        self.query_projection = nn.Linear(feature_dim, feature_dim, bias=False)
        self.output_projection = nn.Linear(feature_dim, feature_dim, bias=False)
        self.layer_norm = nn.LayerNorm(feature_dim, eps=1e-6, elementwise_affine=False)

    def forward(
        self,
        content_tokens: torch.Tensor,
        style_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, content_length, _ = content_tokens.shape
        style_length = style_tokens.shape[1]

        residual = self.query_projection(content_tokens)

        query = residual.view(batch_size, content_length, self.num_heads, self.head_dim)
        key = self.key_projection(style_tokens).view(
            batch_size, style_length, self.num_heads, self.head_dim
        )
        value = self.value_projection(style_tokens).view(
            batch_size, style_length, self.num_heads, self.head_dim
        )

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        fused = torch.matmul(attention_weights, value)
        fused = (
            fused.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, content_length, self.feature_dim)
        )
        fused = self.output_projection(fused)
        return self.layer_norm(fused + residual)


class ContentStyleAggregator(nn.Module):
    """Stacked content-style cross-attention fusion module."""

    def __init__(self, feature_dim: int, num_heads: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ContentStyleAttentionLayer(feature_dim, num_heads)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        content_tokens: torch.Tensor,
        style_tokens: torch.Tensor,
    ) -> torch.Tensor:
        fused_tokens = content_tokens
        for layer in self.layers:
            fused_tokens = layer(fused_tokens, style_tokens)
        return fused_tokens


class CausalConditionedDecoderLayer(nn.Module):
    """Single decoder block with causal self-attention and conditioning cross-attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.cross_attention_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        decoder_tokens: torch.Tensor,
        conditioning_tokens: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        self_attended, _ = self.self_attention(
            self.self_attention_norm(decoder_tokens),
            self.self_attention_norm(decoder_tokens),
            self.self_attention_norm(decoder_tokens),
            need_weights=False,
            attn_mask=causal_mask,
        )
        decoder_tokens = decoder_tokens + self.dropout(self_attended)

        cross_attended, _ = self.cross_attention(
            self.cross_attention_norm(decoder_tokens),
            conditioning_tokens,
            conditioning_tokens,
            need_weights=False,
        )
        decoder_tokens = decoder_tokens + self.dropout(cross_attended)
        decoder_tokens = decoder_tokens + self.mlp(self.mlp_norm(decoder_tokens))
        return decoder_tokens


class AutoregressiveTokenDecoder(nn.Module):
    """Causal Transformer decoder for G-Tok token prediction."""

    def __init__(
        self,
        vocab_size: int,
        sequence_length: int,
        hidden_dim: int,
        num_heads: int,
        mlp_dim: int,
        num_layers: int,
        dropout: float,
        attention_dropout: float,
        conditioning_dim: int,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.bos_token_id = vocab_size

        self.token_embedding = nn.Embedding(vocab_size + 1, hidden_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, sequence_length, hidden_dim)
        )
        self.conditioning_projection = nn.Linear(conditioning_dim, hidden_dim)
        self.conditioning_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.layers = nn.ModuleList(
            [
                CausalConditionedDecoderLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self._lora_injected: bool = False
        self._composed_lora: bool = False

    def _causal_mask(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(sequence_length, sequence_length, device=device)
        mask = mask.masked_fill(
            torch.triu(
                torch.ones(sequence_length, sequence_length, device=device), diagonal=1
            ).bool(),
            float("-inf"),
        )
        return mask

    def prepare_teacher_forcing_inputs(
        self,
        target_token_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length = target_token_indices.shape
        if sequence_length != self.sequence_length:
            raise ValueError(
                f"Expected target_token_indices length {self.sequence_length}, got {sequence_length}"
            )
        bos_column = torch.full(
            (batch_size, 1),
            fill_value=self.bos_token_id,
            device=target_token_indices.device,
            dtype=target_token_indices.dtype,
        )
        return torch.cat([bos_column, target_token_indices[:, :-1]], dim=1)

    def forward(
        self,
        input_token_indices: torch.Tensor,
        conditioning_tokens: torch.Tensor,
    ) -> torch.Tensor:
        sequence_length = input_token_indices.shape[1]
        if sequence_length > self.sequence_length:
            raise ValueError(
                f"Decoder input length {sequence_length} exceeds configured sequence length {self.sequence_length}"
            )

        decoder_tokens = self.token_embedding(input_token_indices)
        decoder_tokens = (
            decoder_tokens + self.position_embedding[:, :sequence_length, :]
        )
        decoder_tokens = self.dropout(decoder_tokens)

        projected_conditioning = self.conditioning_norm(
            self.conditioning_projection(conditioning_tokens)
        )
        causal_mask = self._causal_mask(sequence_length, input_token_indices.device)

        for layer in self.layers:
            decoder_tokens = layer(
                decoder_tokens=decoder_tokens,
                conditioning_tokens=projected_conditioning,
                causal_mask=causal_mask,
            )

        decoder_tokens = self.final_norm(decoder_tokens)
        return self.output_projection(decoder_tokens)

    def inject_lora(self, config: LoRAConfig) -> None:
        """Inject LoRA adaptation into the decoder's linear layers.

        Replaces the MLP projections and attention output projections in every
        decoder layer, plus the final output projection, with ``LoRALinear``
        wrappers.  The base weights of each replaced layer are frozen; only
        the new LoRA matrices are trainable.

        May only be called once per decoder instance.
        """
        if self._lora_injected:
            raise RuntimeError(
                "LoRA has already been injected into this decoder.  "
                "Create a fresh model before injecting again."
            )
        for layer in self.layers:
            # MLP up/down projections.
            layer.mlp[0] = LoRALinear(layer.mlp[0], config.rank, config.alpha)
            layer.mlp[3] = LoRALinear(layer.mlp[3], config.rank, config.alpha)
            # Attention output projections (nn.MultiheadAttention exposes out_proj as
            # a plain nn.Linear, which is the natural target for output-side LoRA).
            layer.self_attention.out_proj = LoRALinear(
                layer.self_attention.out_proj, config.rank, config.alpha
            )
            layer.cross_attention.out_proj = LoRALinear(
                layer.cross_attention.out_proj, config.rank, config.alpha
            )
        self.output_projection = LoRALinear(
            self.output_projection, config.rank, config.alpha
        )
        self._lora_injected = True

    def inject_composed_lora(
        self,
        glyph_state_dict: Dict[str, torch.Tensor],
        lora_config: LoRAConfig,
    ) -> None:
        """Inject two-adapter composed LoRA: frozen glyph prior + trainable font adapter.

        Each linear target in the decoder is replaced with a ``ComposedLoRALinear``
        that holds both adapters.  The glyph adapter weights are loaded from
        *glyph_state_dict* (the output of a previous GA run) and kept frozen.
        The font adapter is zero-initialised and trained during NFA.

        The glyph LoRA scaling is derived from *lora_config* (``alpha / rank``),
        so GA and NFA must use the same rank and alpha.  The glyph rank is
        cross-checked against the tensor shapes in *glyph_state_dict* and a
        ``ValueError`` is raised if they do not match.

        May only be called once per decoder instance (same as ``inject_lora``).
        """
        if self._lora_injected:
            raise RuntimeError(
                "LoRA has already been injected into this decoder.  "
                "Create a fresh model before injecting again."
            )

        def _make_composed(linear: nn.Linear, key_prefix: str) -> ComposedLoRALinear:
            a_key = f"{key_prefix}.lora_A"
            b_key = f"{key_prefix}.lora_B"
            if a_key not in glyph_state_dict or b_key not in glyph_state_dict:
                raise ValueError(
                    f"Glyph LoRA state dict is missing keys '{a_key}' and/or "
                    f"'{b_key}'.  Ensure the state dict was produced by "
                    "inject_lora() with matching layer targets."
                )
            glyph_A = glyph_state_dict[a_key]
            glyph_B = glyph_state_dict[b_key]
            inferred_rank = glyph_A.shape[0]
            if inferred_rank != lora_config.rank:
                raise ValueError(
                    f"Glyph LoRA rank ({inferred_rank}) does not match "
                    f"lora_config.rank ({lora_config.rank}).  GA and NFA must "
                    "use the same --lora-rank."
                )
            scaling = lora_config.alpha / lora_config.rank
            return ComposedLoRALinear(
                linear,
                glyph_lora_A=glyph_A,
                glyph_lora_B=glyph_B,
                glyph_scaling=scaling,
                font_rank=lora_config.rank,
                font_alpha=lora_config.alpha,
            )

        for i, layer in enumerate(self.layers):
            prefix = f"layers.{i}"
            layer.mlp[0] = _make_composed(layer.mlp[0], f"{prefix}.mlp.0")
            layer.mlp[3] = _make_composed(layer.mlp[3], f"{prefix}.mlp.3")
            layer.self_attention.out_proj = _make_composed(
                layer.self_attention.out_proj,
                f"{prefix}.self_attention.out_proj",
            )
            layer.cross_attention.out_proj = _make_composed(
                layer.cross_attention.out_proj,
                f"{prefix}.cross_attention.out_proj",
            )
        self.output_projection = _make_composed(
            self.output_projection, "output_projection"
        )
        self._lora_injected = True
        self._composed_lora = True

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return a state dict containing only the trainable LoRA parameters.

        In single-adapter mode this returns all ``lora_A`` / ``lora_B`` keys.
        In composed mode (two adapters) only the trainable *font* adapter keys
        (``lora_A_font`` / ``lora_B_font``) are returned; the frozen glyph
        adapter is omitted because it is stored in a separate GA checkpoint.
        """
        if self._composed_lora:
            return {
                k: v
                for k, v in self.state_dict().items()
                if "lora_A_font" in k or "lora_B_font" in k
            }
        return {k: v for k, v in self.state_dict().items() if "lora_" in k}

    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load LoRA parameters from a previously saved adaptation checkpoint.

        The decoder must already have LoRA injected before calling this.
        Works for both single-adapter and composed-adapter modes; in the latter
        case *state_dict* should contain only the font adapter keys as returned
        by ``get_lora_state_dict()`` in composed mode.
        """
        if not self._lora_injected:
            raise RuntimeError(
                "LoRA must be injected before loading a LoRA state dict.  "
                "Call inject_lora() or inject_composed_lora() first."
            )
        self.load_state_dict(state_dict, strict=False)

    def set_composed_lora_weights(
        self,
        glyph_weight: float,
        font_weight: float,
    ) -> None:
        """Set adapter multipliers on all composed LoRA layers.

        Raises ``RuntimeError`` if the decoder is not in composed LoRA mode.
        """
        if not self._composed_lora:
            raise RuntimeError(
                "Decoder is not in composed LoRA mode.  "
                "Call inject_composed_lora() first."
            )

        for module in self.modules():
            if isinstance(module, ComposedLoRALinear):
                module.set_adapter_weights(glyph_weight, font_weight)


class ARModel(SaveLoadModel):
    """GAR-Font autoregressive generator model definition."""

    def __init__(
        self,
        config: ARModelConfig,
        gtok_model: Optional[GtokModel] = None,
        language_adapter: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.gtok = gtok_model or GtokModel(GtokConfig(image_size=config.image_size))

        if self.gtok.config.image_size != config.image_size:
            raise ValueError(
                "ARModel and G-Tok image sizes must match "
                f"(got {config.image_size} and {self.gtok.config.image_size})"
            )

        self.sequence_length = self.gtok.sequence_length
        self.token_grid_height = self.gtok.token_grid_height
        self.token_grid_width = self.gtok.token_grid_width
        self.codebook_size = self.gtok.config.quantizer_codebook_size
        self.codebook_dim = self.gtok.config.quantizer_code_dim

        # Match AR encoder pyramid depth to the loaded G-Tok tokenizer so
        # spatial grids always align (for example 16x16 vs 8x8 token grids).
        # This is derived from the GTok config sidecar loaded with the model.
        gtok_ch_mult = self.gtok.config.cnn_channel_multipliers
        if gtok_ch_mult is None:
            raise ValueError(
                "Loaded G-Tok model has no cnn_channel_multipliers in config"
            )
        encoder_ch_mult = tuple(gtok_ch_mult)

        self.content_encoder = CNNEncoder(
            in_channels=3,
            ch=config.content_encoder_base_channels,
            ch_mult=encoder_ch_mult,
            num_res_blocks=config.content_encoder_num_residual_blocks,
            z_channels=config.encoder_feature_dim,
            dropout=0.0,
        )
        self.style_encoder = CNNEncoder(
            in_channels=3,
            ch=config.style_encoder_base_channels,
            ch_mult=encoder_ch_mult,
            num_res_blocks=config.style_encoder_num_residual_blocks,
            z_channels=config.encoder_feature_dim,
            dropout=0.0,
        )
        self.aggregator = ContentStyleAggregator(
            feature_dim=config.encoder_feature_dim,
            num_heads=config.aggregator_num_heads,
            num_layers=config.aggregator_num_layers,
        )

        conditioning_dim = config.encoder_feature_dim * 2
        self.register_buffer(
            "conditioning_position_embeddings",
            create_2d_sinusoidal_position_embeddings(
                self.sequence_length,
                self.token_grid_height,
                self.token_grid_width,
                conditioning_dim,
            ),
            persistent=False,
        )
        self.token_decoder = AutoregressiveTokenDecoder(
            vocab_size=self.codebook_size,
            sequence_length=self.sequence_length,
            hidden_dim=config.decoder_hidden_dim,
            num_heads=config.decoder_num_heads,
            mlp_dim=config.decoder_mlp_dim,
            num_layers=config.decoder_num_layers,
            dropout=config.decoder_dropout,
            attention_dropout=config.decoder_attention_dropout,
            conditioning_dim=conditioning_dim,
        )
        self.language_adapter: Optional[nn.Module] = None
        if language_adapter is not None:
            self.set_language_adapter(language_adapter)

        if config.freeze_gtok:
            self.freeze_gtok()

        self._nfa_mode: bool = False

    def freeze_gtok(self) -> None:
        """Freeze G-Tok so the AR stage trains only its own modules."""
        self.gtok.eval()
        for parameter in self.gtok.parameters():
            parameter.requires_grad = False

    @staticmethod
    def _set_module_trainable(module: nn.Module, trainable: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = trainable
        if trainable:
            module.train()
        else:
            module.eval()

    def set_language_adapter(self, adapter: nn.Module) -> None:
        """Register a language adapter module for multimodal adaptation mode."""
        self.language_adapter = adapter

    def freeze_visual_style_path(self) -> None:
        """Freeze visual style encoder and aggregator for adaptation training."""
        self._set_module_trainable(self.style_encoder, trainable=False)
        self._set_module_trainable(self.aggregator, trainable=False)

    def unfreeze_visual_style_path(self) -> None:
        """Unfreeze visual style encoder and aggregator."""
        self._set_module_trainable(self.style_encoder, trainable=True)
        self._set_module_trainable(self.aggregator, trainable=True)

    def enable_nfa_mode(self, lora_config: LoRAConfig) -> None:
        """Switch to Novel Font Adaptation mode.

        Freezes all base parameters, then injects LoRA into the token decoder.
        After this call only the LoRA parameters are trainable; the optimizer
        should be constructed from ``trainable_parameters()`` rather than
        ``parameters()``.

        May only be called once per model instance.
        """
        if self._nfa_mode:
            raise RuntimeError(
                "Model is already in NFA mode.  "
                "Create a fresh model instance to re-apply NFA."
            )
        # Freeze everything including G-Tok.
        for param in self.parameters():
            param.requires_grad = False
        # Inject LoRA — this creates new trainable parameters inside the decoder.
        self.token_decoder.inject_lora(lora_config)
        # Ensure G-Tok stays in eval mode regardless of outer train() calls.
        self.freeze_gtok()
        self._nfa_mode = True

    @property
    def is_nfa_mode(self) -> bool:
        """True once ``enable_nfa_mode`` or ``enable_composed_nfa_mode`` has been called."""
        return self._nfa_mode

    def enable_composed_nfa_mode(
        self,
        glyph_lora_state: Dict[str, torch.Tensor],
        lora_config: LoRAConfig,
    ) -> None:
        """Switch to composed NFA mode with a frozen glyph prior.

        Differs from ``enable_nfa_mode`` in that the decoder receives *two*
        stacked LoRA adapters:

        1. A **frozen glyph adapter** loaded from *glyph_lora_state* (output of
           a GA run).  This encodes structural priors for the target glyph.
        2. A **trainable font adapter** initialised to zero.  NFA fine-tuning
           updates only this adapter, so the glyph prior is preserved.

        At generation time the effective weight update is the sum of both
        adapter deltas (weighted by their shared ``alpha / rank`` scaling).

        GA and NFA must use the same ``--lora-rank`` and ``--lora-alpha``; a
        ``ValueError`` is raised if the glyph state dict's tensor shapes imply
        a different rank.

        May only be called once per model instance.
        """
        if self._nfa_mode:
            raise RuntimeError(
                "Model is already in NFA mode.  "
                "Create a fresh model instance to re-apply NFA."
            )
        # Freeze everything including G-Tok.
        for param in self.parameters():
            param.requires_grad = False
        # Inject composed LoRA — glyph adapter frozen, font adapter trainable.
        self.token_decoder.inject_composed_lora(glyph_lora_state, lora_config)
        # Ensure G-Tok stays in eval mode regardless of outer train() calls.
        self.freeze_gtok()
        self._nfa_mode = True

    def set_composed_lora_weights(
        self,
        glyph_weight: float,
        font_weight: float,
    ) -> None:
        """Set glyph/font adapter multipliers for composed LoRA mode."""
        self.token_decoder.set_composed_lora_weights(glyph_weight, font_weight)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Return a list of parameters that currently require gradients.

        In NFA mode this is just the injected LoRA matrices.  In normal
        pretraining mode it is all non-GTok parameters.
        """
        return [p for p in self.parameters() if p.requires_grad]

    def encode_content(self, content_images: torch.Tensor) -> torch.Tensor:
        """Encode content glyphs to token-aligned spatial features."""
        content_features = self.content_encoder(content_images)
        batch_size, channels, height, width = content_features.shape
        if height != self.token_grid_height or width != self.token_grid_width:
            raise ValueError(
                "Content encoder output shape does not match G-Tok token grid "
                f"(got {(height, width)}, expected {(self.token_grid_height, self.token_grid_width)})"
            )
        return content_features.permute(0, 2, 3, 1).reshape(
            batch_size, height * width, channels
        )

    def encode_style(self, style_reference_images: torch.Tensor) -> torch.Tensor:
        """Encode style references and flatten them into attention memory tokens."""
        batch_size, num_references, channels, height, width = (
            style_reference_images.shape
        )
        if channels != 3:
            raise ValueError(f"Expected RGB style references, got {channels} channels")
        flattened_images = style_reference_images.reshape(
            batch_size * num_references, channels, height, width
        )
        encoded = self.style_encoder(flattened_images)
        _, feature_channels, feature_height, feature_width = encoded.shape
        if (
            feature_height != self.token_grid_height
            or feature_width != self.token_grid_width
        ):
            raise ValueError(
                "Style encoder output shape does not match G-Tok token grid "
                f"(got {(feature_height, feature_width)}, expected {(self.token_grid_height, self.token_grid_width)})"
            )
        encoded = encoded.reshape(
            batch_size,
            num_references,
            feature_channels,
            feature_height,
            feature_width,
        )
        style_tokens = encoded.permute(0, 1, 3, 4, 2).reshape(
            batch_size,
            num_references * feature_height * feature_width,
            feature_channels,
        )
        return style_tokens

    def aggregate_conditioning(
        self,
        content_images: torch.Tensor,
        style_reference_images: torch.Tensor,
    ) -> torch.Tensor:
        """Build the conditioning sequence used by the AR decoder."""
        content_tokens = self.encode_content(content_images)
        style_tokens = self.encode_style(style_reference_images)
        aggregated_style_tokens = self.aggregator(content_tokens, style_tokens)
        conditioning_tokens = torch.cat(
            [content_tokens, aggregated_style_tokens], dim=-1
        )
        return conditioning_tokens + self.conditioning_position_embeddings.unsqueeze(0)

    def aggregate_conditioning_components(
        self,
        content_images: torch.Tensor,
        style_reference_images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return content, style-aggregated, and final visual conditioning tokens."""
        content_tokens = self.encode_content(content_images)
        style_tokens = self.encode_style(style_reference_images)
        aggregated_style_tokens = self.aggregator(content_tokens, style_tokens)
        conditioning_tokens = torch.cat(
            [content_tokens, aggregated_style_tokens], dim=-1
        )
        conditioning_tokens = (
            conditioning_tokens + self.conditioning_position_embeddings.unsqueeze(0)
        )
        return content_tokens, aggregated_style_tokens, conditioning_tokens

    def _adapt_style_tokens_with_language(
        self,
        style_tokens: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if self.language_adapter is None:
            raise RuntimeError(
                "No language adapter is registered. Call set_language_adapter(...) before adaptation mode."
            )
        adapted_style_tokens = self.language_adapter(style_tokens, text_embeddings)
        if adapted_style_tokens.shape != style_tokens.shape:
            raise ValueError(
                "Language adapter must return style tokens with the same shape as input "
                f"(got {tuple(adapted_style_tokens.shape)} vs {tuple(style_tokens.shape)})"
            )
        return adapted_style_tokens

    def _decode_with_teacher_forcing(
        self,
        conditioning_tokens: torch.Tensor,
        target_token_indices: torch.Tensor,
        descriptions: Optional[List[str]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        decoder_input_tokens = self.token_decoder.prepare_teacher_forcing_inputs(
            target_token_indices
        )
        logits = self.token_decoder(decoder_input_tokens, conditioning_tokens)
        soft_token_embeddings, reconstructed_images = self.soft_decode(
            logits,
            descriptions=descriptions,
        )
        return logits, soft_token_embeddings, reconstructed_images

    def _decode_with_scheduled_sampling(
        self,
        conditioning_tokens: torch.Tensor,
        target_token_indices: torch.Tensor,
        scheduled_sampling_probability: float,
        descriptions: Optional[List[str]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode with mixed teacher/predicted previous tokens.

        Uses a parallel scheduled-sampling approximation: first obtain token
        predictions from a teacher-forced pass, then replace each previous-token
        input with that predicted token with probability ``p``.
        """
        if scheduled_sampling_probability <= 0.0:
            return self._decode_with_teacher_forcing(
                conditioning_tokens=conditioning_tokens,
                target_token_indices=target_token_indices,
                descriptions=descriptions,
            )

        p = min(1.0, max(0.0, float(scheduled_sampling_probability)))
        teacher_inputs = self.token_decoder.prepare_teacher_forcing_inputs(
            target_token_indices
        )

        # Generate a detached prediction stream used only to construct mixed
        # decoder inputs, while gradients flow through the final decode pass.
        with torch.no_grad():
            teacher_logits = self.token_decoder(teacher_inputs, conditioning_tokens)
            teacher_predictions = torch.argmax(teacher_logits, dim=-1)

        previous_ground_truth = target_token_indices[:, :-1]
        previous_predictions = teacher_predictions[:, :-1]
        use_prediction_mask = (
            torch.rand(
                previous_ground_truth.shape,
                device=target_token_indices.device,
            )
            < p
        )
        mixed_previous_tokens = torch.where(
            use_prediction_mask,
            previous_predictions,
            previous_ground_truth,
        )

        batch_size = target_token_indices.shape[0]
        bos_column = torch.full(
            (batch_size, 1),
            fill_value=self.token_decoder.bos_token_id,
            device=target_token_indices.device,
            dtype=target_token_indices.dtype,
        )
        mixed_decoder_inputs = torch.cat([bos_column, mixed_previous_tokens], dim=1)

        logits = self.token_decoder(mixed_decoder_inputs, conditioning_tokens)
        soft_token_embeddings, reconstructed_images = self.soft_decode(
            logits,
            descriptions=descriptions,
        )
        return logits, soft_token_embeddings, reconstructed_images

    def target_token_indices_from_images(
        self,
        target_images: torch.Tensor,
        descriptions: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Encode target glyph images into G-Tok codebook indices."""
        batch_size = target_images.shape[0]

        cnn_out = self.gtok.cnn_encoder(target_images)
        _batch_size, channels, height, width = cnn_out.shape
        cnn_tokens = cnn_out.permute(0, 2, 3, 1).reshape(
            batch_size,
            height * width,
            channels,
        )
        vit_tokens = self.gtok.vit_encoder(cnn_tokens)[:, 1:, :]

        text_embeddings = self.gtok._description_embeddings(
            descriptions,
            batch_size=batch_size,
            device=target_images.device,
        )
        vit_tokens = self.gtok._apply_feature_affine(
            vit_tokens,
            text_embeddings,
            self.gtok.encoder_text_projection,
            self.gtok.encoder_text_affine,
        )

        quantizer_inputs = self.gtok.vit_encoder_to_quantizer(vit_tokens)
        quantizer_inputs = quantizer_inputs.reshape(
            batch_size,
            self.token_grid_height,
            self.token_grid_width,
            self.codebook_dim,
        ).permute(0, 3, 1, 2)

        _quantized, _loss_info, indices_info = self.gtok.quantizer(quantizer_inputs)
        token_indices = indices_info[2].reshape(batch_size, self.sequence_length)
        return token_indices

    def codebook_embeddings(self) -> torch.Tensor:
        """Return the codebook matrix used for soft and hard decoding."""
        codebook = self.gtok.quantizer.embedding.weight
        if self.gtok.quantizer.l2_norm:
            codebook = F.normalize(codebook, p=2, dim=-1)
        return codebook

    def soft_decode(
        self,
        logits: torch.Tensor,
        descriptions: Optional[List[str]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project token logits onto the G-Tok codebook and decode to images."""
        probabilities = torch.softmax(logits, dim=-1)
        soft_token_embeddings = torch.matmul(probabilities, self.codebook_embeddings())
        reconstructed_images = self.gtok.decode(
            soft_token_embeddings,
            descriptions=descriptions,
        )
        return soft_token_embeddings, reconstructed_images

    def forward(
        self,
        content_images: torch.Tensor,
        style_reference_images: torch.Tensor,
        *,
        target_token_indices: Optional[torch.Tensor] = None,
        target_images: Optional[torch.Tensor] = None,
        scheduled_sampling_probability: float = 0.0,
        descriptions: Optional[List[str]] = None,
    ) -> ARModelOutput:
        """Run teacher-forced AR decoding for visual pretraining.

        Either ``target_token_indices`` or ``target_images`` must be provided.
        The latter is convenient during training because it allows the model to
        derive token targets directly from the frozen G-Tok tokenizer.
        """
        conditioning_tokens = self.aggregate_conditioning(
            content_images=content_images,
            style_reference_images=style_reference_images,
        )

        if target_token_indices is None:
            if target_images is None:
                raise ValueError(
                    "Either target_token_indices or target_images must be provided for ARModel.forward"
                )
            with torch.no_grad():
                target_token_indices = self.target_token_indices_from_images(
                    target_images,
                    descriptions=descriptions,
                )
        logits, soft_token_embeddings, reconstructed_images = (
            self._decode_with_scheduled_sampling(
                conditioning_tokens=conditioning_tokens,
                target_token_indices=target_token_indices,
                scheduled_sampling_probability=scheduled_sampling_probability,
                descriptions=descriptions,
            )
        )

        return ARModelOutput(
            logits=logits,
            reconstructed_images=reconstructed_images,
            soft_token_embeddings=soft_token_embeddings,
            conditioning_tokens=conditioning_tokens,
            target_token_indices=target_token_indices,
        )

    def forward_adaptation(
        self,
        content_images: torch.Tensor,
        style_reference_images: torch.Tensor,
        text_embeddings: torch.Tensor,
        *,
        target_token_indices: Optional[torch.Tensor] = None,
        target_images: Optional[torch.Tensor] = None,
        run_decoder: bool = False,
        descriptions: Optional[List[str]] = None,
    ) -> ARAdaptationOutput:
        """Forward path for visual-language adaptation mode.

        This branch exposes both visual-only and multimodal aggregated features
        so later training can apply alignment losses between them. Optionally,
        it can also run the decoder branch for token/pixel supervision.
        """
        content_tokens = self.encode_content(content_images)
        style_tokens = self.encode_style(style_reference_images)

        visual_aggregated_style_tokens = self.aggregator(content_tokens, style_tokens)
        visual_conditioning_tokens = torch.cat(
            [content_tokens, visual_aggregated_style_tokens], dim=-1
        )
        visual_conditioning_tokens = (
            visual_conditioning_tokens
            + self.conditioning_position_embeddings.unsqueeze(0)
        )

        adapted_style_tokens = self._adapt_style_tokens_with_language(
            style_tokens=style_tokens,
            text_embeddings=text_embeddings,
        )
        multimodal_aggregated_style_tokens = self.aggregator(
            content_tokens, adapted_style_tokens
        )
        multimodal_conditioning_tokens = torch.cat(
            [content_tokens, multimodal_aggregated_style_tokens], dim=-1
        )
        multimodal_conditioning_tokens = (
            multimodal_conditioning_tokens
            + self.conditioning_position_embeddings.unsqueeze(0)
        )

        logits: Optional[torch.Tensor] = None
        reconstructed_images: Optional[torch.Tensor] = None
        soft_token_embeddings: Optional[torch.Tensor] = None

        if run_decoder:
            if target_token_indices is None:
                if target_images is None:
                    raise ValueError(
                        "Either target_token_indices or target_images must be provided when run_decoder=True"
                    )
                with torch.no_grad():
                    target_token_indices = self.target_token_indices_from_images(
                        target_images,
                        descriptions=descriptions,
                    )
            logits, soft_token_embeddings, reconstructed_images = (
                self._decode_with_teacher_forcing(
                    conditioning_tokens=multimodal_conditioning_tokens,
                    target_token_indices=target_token_indices,
                    descriptions=descriptions,
                )
            )

        return ARAdaptationOutput(
            multimodal_conditioning_tokens=multimodal_conditioning_tokens,
            visual_conditioning_tokens=visual_conditioning_tokens,
            multimodal_aggregated_style_tokens=multimodal_aggregated_style_tokens,
            visual_aggregated_style_tokens=visual_aggregated_style_tokens,
            logits=logits,
            reconstructed_images=reconstructed_images,
            soft_token_embeddings=soft_token_embeddings,
            target_token_indices=target_token_indices,
        )

    @torch.no_grad()
    def generate(
        self,
        content_images: torch.Tensor,
        style_reference_images: torch.Tensor,
        descriptions: Optional[List[str]] = None,
    ) -> ARModelOutput:
        """Greedily decode a full token sequence and reconstruct the glyph image."""
        conditioning_tokens = self.aggregate_conditioning(
            content_images=content_images,
            style_reference_images=style_reference_images,
        )
        batch_size = content_images.shape[0]
        generated_tokens = torch.full(
            (batch_size, 1),
            fill_value=self.token_decoder.bos_token_id,
            device=content_images.device,
            dtype=torch.long,
        )

        predicted_token_indices = []
        for _ in tqdm(range(self.sequence_length)):
            logits = self.token_decoder(generated_tokens, conditioning_tokens)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            predicted_token_indices.append(next_token)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

        predicted_token_indices_tensor = torch.cat(predicted_token_indices, dim=1)
        logits = self.token_decoder(
            self.token_decoder.prepare_teacher_forcing_inputs(
                predicted_token_indices_tensor
            ),
            conditioning_tokens,
        )
        soft_token_embeddings, reconstructed_images = self.soft_decode(
            logits,
            descriptions=descriptions,
        )
        return ARModelOutput(
            logits=logits,
            reconstructed_images=reconstructed_images,
            soft_token_embeddings=soft_token_embeddings,
            conditioning_tokens=conditioning_tokens,
            target_token_indices=predicted_token_indices_tensor,
        )

    @torch.no_grad()
    def generate_adaptation(
        self,
        content_images: torch.Tensor,
        style_reference_images: torch.Tensor,
        text_embeddings: torch.Tensor,
        descriptions: Optional[list[str]] = None,
    ) -> ARAdaptationOutput:
        """Greedy generation path that uses multimodal conditioning tokens."""
        content_tokens = self.encode_content(content_images)
        style_tokens = self.encode_style(style_reference_images)
        visual_aggregated_style_tokens = self.aggregator(content_tokens, style_tokens)
        visual_conditioning_tokens = torch.cat(
            [content_tokens, visual_aggregated_style_tokens], dim=-1
        )
        visual_conditioning_tokens = (
            visual_conditioning_tokens
            + self.conditioning_position_embeddings.unsqueeze(0)
        )

        adapted_style_tokens = self._adapt_style_tokens_with_language(
            style_tokens=style_tokens,
            text_embeddings=text_embeddings,
        )
        multimodal_aggregated_style_tokens = self.aggregator(
            content_tokens, adapted_style_tokens
        )
        multimodal_conditioning_tokens = torch.cat(
            [content_tokens, multimodal_aggregated_style_tokens], dim=-1
        )
        multimodal_conditioning_tokens = (
            multimodal_conditioning_tokens
            + self.conditioning_position_embeddings.unsqueeze(0)
        )

        batch_size = content_images.shape[0]
        generated_tokens = torch.full(
            (batch_size, 1),
            fill_value=self.token_decoder.bos_token_id,
            device=content_images.device,
            dtype=torch.long,
        )
        predicted_token_indices = []
        for _ in range(self.sequence_length):
            logits_step = self.token_decoder(
                generated_tokens, multimodal_conditioning_tokens
            )
            next_token = torch.argmax(logits_step[:, -1, :], dim=-1, keepdim=True)
            predicted_token_indices.append(next_token)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

        predicted_token_indices_tensor = torch.cat(predicted_token_indices, dim=1)
        logits, soft_token_embeddings, reconstructed_images = (
            self._decode_with_teacher_forcing(
                conditioning_tokens=multimodal_conditioning_tokens,
                target_token_indices=predicted_token_indices_tensor,
                descriptions=descriptions,
            )
        )

        return ARAdaptationOutput(
            multimodal_conditioning_tokens=multimodal_conditioning_tokens,
            visual_conditioning_tokens=visual_conditioning_tokens,
            multimodal_aggregated_style_tokens=multimodal_aggregated_style_tokens,
            visual_aggregated_style_tokens=visual_aggregated_style_tokens,
            logits=logits,
            reconstructed_images=reconstructed_images,
            soft_token_embeddings=soft_token_embeddings,
            target_token_indices=predicted_token_indices_tensor,
        )

    def load(self, path: str, device: torch.device) -> None:
        """Load AR model weights from a checkpoint, handling G-Tok and adapter gracefully.

        ``gtok.*`` keys are always stripped from the checkpoint before loading
        because G-Tok is loaded and managed independently; the embedded copy
        saved in the AR checkpoint may have a different architecture.

        ``language_adapter.*`` keys are loaded only if a language adapter has
        already been registered via ``set_language_adapter``.  If the checkpoint
        contains adapter keys but no adapter is registered, a warning is printed
        and those keys are skipped — this is the expected behaviour when a stage-1
        visual-only generate pipeline encounters a multimodal checkpoint.  If the
        adapter keys are absent but an adapter is registered, the adapter keeps its
        default (zero-init) weights and a warning is printed.

        Adapter keys are always loaded with ``strict=False`` to tolerate schema
        changes between checkpoint versions (e.g. the removal of ``output_norm``).
        Core AR keys are loaded with validated non-strict logic so we can
        intentionally ignore ``gtok.*`` keys while still failing on any missing
        or unexpected non-GTok/non-adapter parameters.
        """
        state_dict = torch.load(path, map_location=device, weights_only=True)

        # Separate the keys into three groups.
        gtok_keys = {k: v for k, v in state_dict.items() if k.startswith("gtok.")}
        adapter_keys = {
            k.removeprefix("language_adapter."): v
            for k, v in state_dict.items()
            if k.startswith("language_adapter.")
        }
        core_keys = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith("gtok.") and not k.startswith("language_adapter.")
        }

        if gtok_keys:
            print(
                f"ARModel.load: skipping {len(gtok_keys)} gtok.* keys "
                "(G-Tok is loaded separately)"
            )

        if adapter_keys and self.language_adapter is None:
            print(
                f"ARModel.load: checkpoint contains {len(adapter_keys)} language_adapter.* "
                "keys but no adapter is registered — skipping adapter weights"
            )
        elif not adapter_keys and self.language_adapter is not None:
            print(
                "ARModel.load: no language_adapter.* keys in checkpoint — "
                "adapter will keep its default initialisation"
            )
        elif adapter_keys and self.language_adapter is not None:
            missing, unexpected = self.language_adapter.load_state_dict(
                adapter_keys, strict=False
            )
            if unexpected:
                print(
                    f"ARModel.load: ignored {len(unexpected)} unexpected adapter "
                    f"key(s): {unexpected}"
                )
            if missing:
                print(
                    f"ARModel.load: {len(missing)} adapter key(s) absent in checkpoint "
                    f"(kept at default init): {missing}"
                )

        incompatible = self.load_state_dict(core_keys, strict=False)

        allowed_missing_prefixes = ["gtok."]
        if self.language_adapter is not None:
            allowed_missing_prefixes.append("language_adapter.")

        disallowed_missing = [
            key
            for key in incompatible.missing_keys
            if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
        ]

        if incompatible.unexpected_keys or disallowed_missing:
            details = []
            if incompatible.unexpected_keys:
                details.append(f"unexpected keys: {incompatible.unexpected_keys}")
            if disallowed_missing:
                details.append(f"missing keys: {disallowed_missing}")
            raise RuntimeError(
                "ARModel.load failed due to checkpoint/schema mismatch in core AR weights: "
                + "; ".join(details)
            )

    def parameter_counts(self) -> Dict[str, int]:
        """Return parameter counts for the main AR components."""
        return {
            "content_encoder": sum(
                p.numel() for p in self.content_encoder.parameters()
            ),
            "style_encoder": sum(p.numel() for p in self.style_encoder.parameters()),
            "aggregator": sum(p.numel() for p in self.aggregator.parameters()),
            "token_decoder": sum(p.numel() for p in self.token_decoder.parameters()),
            "total_trainable": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }
