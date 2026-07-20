"""LoRA (Low-Rank Adaptation) primitives for MaskGIT transformer fine-tuning.

These modules enable lightweight per-font (NFA) and per-glyph (GA) adaptation
of the MaskGIT token decoder without modifying the base model weights.

Design:
- ``LoRALinear`` wraps a frozen ``nn.Linear`` with trainable low-rank deltas.
- ``ComposedLoRALinear`` holds two stacked adapters: a frozen glyph prior (from
  GA training) and a trainable font adapter (for NFA).  Both share the same rank
  and scaling factor to keep the composed path a well-defined linear operation.
- ``inject_lora`` / ``inject_composed_lora`` replace targeted ``nn.Linear``
  layers in-place and are designed to be called once per transformer instance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoRAConfig:
    """Configuration for LoRA adaptation layers.

    Attributes:
        rank: Low-rank dimension *r*.  Smaller = fewer parameters but less
            capacity.  Typical values: 4–32.
        alpha: Scaling factor.  The effective scale applied to the LoRA output
            is ``alpha / rank`` so that increasing rank does not intrinsically
            amplify the adaptation signal.  Typical values: match rank or
            use a small multiple (e.g. rank=16, alpha=32 → scale=2.0).
    """

    rank: int = 16
    alpha: float = 16.0

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {self.alpha}")

    @property
    def scaling(self) -> float:
        """Effective scale factor: alpha / rank."""
        return self.alpha / self.rank


# ---------------------------------------------------------------------------
# Single-adapter LoRA (NFA or GA)
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """LoRA-adapted linear layer wrapping a frozen base ``nn.Linear``.

    The forward pass is::

        output = base(x) + (x @ lora_Aᵀ @ lora_Bᵀ) * scaling

    where ``lora_A`` is ``(r, in_features)`` and ``lora_B`` is
    ``(out_features, r)``.  ``lora_A`` is Kaiming-initialised;
    ``lora_B`` is zero-initialised so that at step 0 the adapter
    contributes nothing and the model behaves identically to the
    frozen base.

    The base layer's ``requires_grad`` is set to ``False`` on creation.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
    ) -> None:
        if rank >= base.in_features or rank >= base.out_features:
            raise ValueError(
                f"LoRA rank {rank} must be strictly less than both "
                f"in_features ({base.in_features}) and out_features "
                f"({base.out_features})"
            )

        super().__init__()
        scaling = alpha / rank

        self.base = base
        self.scaling = scaling

        # Freeze base weights — only LoRA matrices are trainable.
        for p in base.parameters():
            p.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B is zero-initialised → initial adapter output is zero.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + delta


# ---------------------------------------------------------------------------
# Composed dual-adapter LoRA (GA prior + NFA)
# ---------------------------------------------------------------------------


class ComposedLoRALinear(nn.Module):
    """Linear layer with two stacked LoRA adapters.

    - *glyph adapter*: frozen weights loaded from a GA checkpoint.
      Teaches the model the structural token patterns of a specific codepoint.
    - *font adapter*: zero-initialised, trainable.  Learns font-specific
      stylistic deviations during NFA.

    The forward pass is::

        output = base(x)
               + (x @ glyph_Aᵀ @ glyph_Bᵀ) * scaling
               + (x @ font_Aᵀ @ font_Bᵀ) * scaling

    Both adapters share the same rank and scaling.  The glyph adapter must
    be produced by a prior GA run with matching rank and layer targets.
    """

    def __init__(
        self,
        base: nn.Linear,
        glyph_lora_A: torch.Tensor,
        glyph_lora_B: torch.Tensor,
        glyph_scaling: float,
        font_rank: int,
        font_alpha: float,
    ) -> None:
        if font_rank >= base.in_features or font_rank >= base.out_features:
            raise ValueError(
                f"LoRA rank {font_rank} must be strictly less than both "
                f"in_features ({base.in_features}) and out_features "
                f"({base.out_features})"
            )

        super().__init__()
        self.scaling = font_alpha / font_rank

        self.base = base
        for p in base.parameters():
            p.requires_grad = False

        # Frozen glyph adapter.
        self.lora_A_glyph = nn.Parameter(glyph_lora_A, requires_grad=False)
        self.lora_B_glyph = nn.Parameter(glyph_lora_B, requires_grad=False)
        self.glyph_scaling = glyph_scaling

        # Trainable font adapter (zero-init).
        self.lora_A_font = nn.Parameter(torch.zeros(font_rank, base.in_features))
        self.lora_B_font = nn.Parameter(torch.zeros(base.out_features, font_rank))
        nn.init.kaiming_uniform_(self.lora_A_font, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        glyph_delta = (
            x @ self.lora_A_glyph.T @ self.lora_B_glyph.T
        ) * self.glyph_scaling
        font_delta = (
            x @ self.lora_A_font.T @ self.lora_B_font.T
        ) * self.scaling
        return base_out + glyph_delta + font_delta


# ---------------------------------------------------------------------------
# LoRA state-dict helpers
# ---------------------------------------------------------------------------


def _lora_state_dict_keys(module: nn.Module) -> Dict[str, torch.Tensor]:
    """Return a state dict containing only LoRA parameters from *module*.

    Keys containing ``lora_A`` or ``lora_B`` (single-adapter) are included.
    In composed mode, only the trainable font adapter keys
    (``lora_A_font`` / ``lora_B_font``) are returned; the frozen glyph
    adapter is omitted.
    """
    return {
        k: v
        for k, v in module.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }


def _has_lora(module: nn.Module) -> bool:
    """Return True if any submodule is a LoRA wrapper."""
    return isinstance(module, (LoRALinear, ComposedLoRALinear))
