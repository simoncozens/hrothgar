"""Upstream GAR-Font reference implementations.

This package contains lightly-adapted copies of modules from the upstream
GAR-Font codebase (https://github.com/...), with import paths adjusted
for the hrothgar package layout.

All classes are kept as close to the upstream as possible so that
divergences from the reference are easy to identify.
"""

from hrothgar.upstream.tokenizer import (
    Tokenizer,
    TokenizerModelArgs,
    CNNEncoder,
    CNNDecoder,
    ViTEncoder,
    ViTDecoder,
    VectorQuantizer,
)
from hrothgar.upstream.blocks import (
    ConvBlock,
    ResBlock,
    AttentionBlock,
    nonlinearity,
)

__all__ = [
    "Tokenizer",
    "TokenizerModelArgs",
    "CNNEncoder",
    "CNNDecoder",
    "ViTEncoder",
    "ViTDecoder",
    "VectorQuantizer",
    "ConvBlock",
    "ResBlock",
    "AttentionBlock",
    "nonlinearity",
]
