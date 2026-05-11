"""Multimodal style-conditioning modules for AR adaptation.

This module provides lightweight, dependency-free building blocks for
phase-3 multimodal training:

1. ``HashedDescriptionEncoder`` maps description strings to fixed token
   embeddings (frozen text features).
2. ``TextStyleAdapter`` fuses text embeddings into visual style tokens through
   stacked cross-attention and returns style tokens with unchanged shape.

The adapter can be plugged into ``ARModel.set_language_adapter`` and then used
through ``ARModel.forward_adaptation``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Sequence

import torch
import torch.nn as nn

_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class HashedDescriptionEncoderConfig:
    """Configuration for ``HashedDescriptionEncoder``.

    Attributes:
        vocab_size: Number of hashing buckets for token lookup.
        embedding_dim: Output embedding size for each token.
        max_tokens: Maximum tokens kept from each description.
        init_std: Standard deviation used for embedding-table initialisation.
        seed: RNG seed used to create a stable embedding table.
    """

    vocab_size: int = 4096
    embedding_dim: int = 512
    max_tokens: int = 64
    init_std: float = 0.02
    seed: int = 1234

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be positive, got {self.embedding_dim}"
            )
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if self.init_std <= 0.0:
            raise ValueError(f"init_std must be positive, got {self.init_std}")


class HashedDescriptionEncoder(nn.Module):
    """Deterministic, frozen text encoder for font descriptions.

    This is intentionally lightweight: descriptions are tokenised with a simple
    regex, each token is hashed into a fixed bucket, and embeddings are looked
    up from a frozen table. The output shape is ``(B, max_tokens, embedding_dim)``.
    """

    def __init__(self, config: HashedDescriptionEncoderConfig) -> None:
        super().__init__()
        self.config = config

        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed)
        table = (
            torch.randn(
                config.vocab_size,
                config.embedding_dim,
                generator=generator,
                dtype=torch.float32,
            )
            * config.init_std
        )
        self.embedding = nn.Embedding.from_pretrained(table, freeze=True)

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        return _WORD_RE.findall(text.lower())

    def _hash_token(self, token: str) -> int:
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        return (
            int.from_bytes(digest[:8], byteorder="big", signed=False)
            % self.config.vocab_size
        )

    def _indices_for_text(self, text: str) -> list[int]:
        tokens = self._tokenise(text)
        if not tokens:
            tokens = ["<empty>"]
        hashed = [self._hash_token(token) for token in tokens]
        hashed = hashed[: self.config.max_tokens]
        if len(hashed) < self.config.max_tokens:
            hashed.extend([0] * (self.config.max_tokens - len(hashed)))
        return hashed

    def forward(self, descriptions: Sequence[str]) -> torch.Tensor:
        """Encode text descriptions into token embeddings.

        Args:
            descriptions: Sequence of description strings.

        Returns:
            Tensor with shape ``(batch_size, max_tokens, embedding_dim)``.
        """
        if len(descriptions) == 0:
            raise ValueError("descriptions must not be empty")

        indices = torch.tensor(
            [self._indices_for_text(desc) for desc in descriptions],
            dtype=torch.long,
            device=self.embedding.weight.device,
        )
        return self.embedding(indices)


@dataclass(frozen=True)
class TextStyleAdapterConfig:
    """Configuration for multimodal text-style cross-attention adapter."""

    style_token_dim: int = 256
    text_embedding_dim: int = 512
    adapter_hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.style_token_dim <= 0:
            raise ValueError(
                f"style_token_dim must be positive, got {self.style_token_dim}"
            )
        if self.text_embedding_dim <= 0:
            raise ValueError(
                f"text_embedding_dim must be positive, got {self.text_embedding_dim}"
            )
        if self.adapter_hidden_dim <= 0:
            raise ValueError(
                f"adapter_hidden_dim must be positive, got {self.adapter_hidden_dim}"
            )
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.adapter_hidden_dim % self.num_heads != 0:
            raise ValueError(
                "adapter_hidden_dim must be divisible by num_heads "
                f"(got {self.adapter_hidden_dim} and {self.num_heads})"
            )
        if self.dropout < 0.0:
            raise ValueError(f"dropout must be non-negative, got {self.dropout}")


class _AdapterBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.key_value_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, style_tokens: torch.Tensor, text_tokens: torch.Tensor
    ) -> torch.Tensor:
        attended, _ = self.cross_attention(
            query=self.query_norm(style_tokens),
            key=self.key_value_norm(text_tokens),
            value=self.key_value_norm(text_tokens),
            need_weights=False,
        )
        style_tokens = style_tokens + self.dropout(attended)
        style_tokens = style_tokens + self.ffn(self.ffn_norm(style_tokens))
        return style_tokens


class TextStyleAdapter(nn.Module):
    """Cross-attention adapter that injects text style cues into style tokens.

    Input and output style token shapes are identical so the adapter can be used
    directly with ``ARModel._adapt_style_tokens_with_language``.
    """

    def __init__(self, config: TextStyleAdapterConfig) -> None:
        super().__init__()
        self.config = config

        self.style_in = nn.Linear(config.style_token_dim, config.adapter_hidden_dim)
        self.text_in = nn.Linear(config.text_embedding_dim, config.adapter_hidden_dim)
        self.blocks = nn.ModuleList(
            [
                _AdapterBlock(
                    hidden_dim=config.adapter_hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.style_out = nn.Linear(config.adapter_hidden_dim, config.style_token_dim)

        # Initialize with small scale to enable gradient flow while minimizing
        # disruption to the frozen AR model. Zero-initialization would create a
        # gradient vanishing problem: at initialization, the adapter output is
        # identical to the visual input, resulting in zero alignment loss and
        # zero gradients everywhere. Using small-scale normal initialization
        # (std ~1/sqrt(dim)) ensures non-zero loss gradient from the first step.
        nn.init.normal_(self.style_out.weight, mean=0.0, std=1.0 / (config.adapter_hidden_dim ** 0.5))
        nn.init.normal_(self.style_out.bias, mean=0.0, std=1.0 / (config.adapter_hidden_dim ** 0.5))

    def forward(
        self, style_tokens: torch.Tensor, text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Adapt visual style tokens using text embeddings.

        Args:
            style_tokens: ``(B, S, D_style)`` visual style tokens.
            text_embeddings: ``(B, T, D_text)`` text embeddings.

        Returns:
            Adapted style tokens with shape ``(B, S, D_style)``.
        """
        if style_tokens.ndim != 3:
            raise ValueError(
                f"style_tokens must have shape (B, S, D), got {tuple(style_tokens.shape)}"
            )
        if text_embeddings.ndim != 3:
            raise ValueError(
                "text_embeddings must have shape (B, T, D), "
                f"got {tuple(text_embeddings.shape)}"
            )
        if style_tokens.shape[0] != text_embeddings.shape[0]:
            raise ValueError(
                "Batch size mismatch between style and text tensors "
                f"({style_tokens.shape[0]} vs {text_embeddings.shape[0]})"
            )

        residual = style_tokens
        style_hidden = self.style_in(style_tokens)
        text_hidden = self.text_in(text_embeddings)

        for block in self.blocks:
            style_hidden = block(style_hidden, text_hidden)

        delta = self.style_out(style_hidden)
        return residual + delta
