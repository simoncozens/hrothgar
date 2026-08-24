"""MaskGIT: non-autoregressive parallel token prediction for glyph generation.

MaskGIT replaces the causal autoregressive decoder with a bidirectional
transformer trained via masked token prediction (like BERT).  During
inference, tokens are generated iteratively: the model predicts all
positions in parallel, commits the most confident predictions (or, with
the Halton scheduler, reveals positions in a fixed low-discrepancy spatial
order), and re-masks the rest for the next iteration.

This eliminates exposure bias entirely — the model is always trained
and evaluated on masked inputs, never on a distribution-shifted
autoregressive context.

Reference:
    Chang et al., "MaskGIT: Masked Generative Image Transformer", CVPR 2022.
    https://arxiv.org/abs/2202.04200
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hrothgar.upstream.gpt import (
    GPTModelArgs,
    ImgFeatureMapEmbedder,
    RMSNorm,
    TransformerBlock,
    precompute_freqs_cis_1d,
)

# ---------------------------------------------------------------------------
# Mask schedule utilities
# ---------------------------------------------------------------------------


def _cosine_mask_ratio() -> float:
    """Sample a mask ratio from the MaskGIT cosine distribution.

    Samples ``r ~ Uniform(0, 1]`` and returns ``cos(pi/2 * r)``, which
    biases toward higher mask ratios (more tokens masked during training).

    Returns:
        Float in ``(0, 1]`` — the fraction of tokens to mask.
    """
    r = 1.0 - torch.rand(1).item()  # (0, 1]
    return math.cos(math.pi / 2 * r)


def _cosine_unmask_schedule(step: int, total_steps: int, num_tokens: int) -> int:
    """Number of tokens to **keep unmasked** at inference step *step*.

    Uses the cosine schedule from MaskGIT:
        ``mask_frac(t) = cos(pi/2 * (t+1) / T)``
        ``keep(t) = N * (1 - mask_frac(t))``

    At ``step == 0`` a few tokens are immediately kept; at
    ``step == total_steps - 1`` all tokens are kept.

    Args:
        step: Current decoding step (0-indexed).
        total_steps: Total number of decoding steps.
        num_tokens: Total number of token positions.

    Returns:
        Number of tokens to keep unmasked at this step (clamped to ``[1, N]``).
    """
    if step >= total_steps - 1:
        return num_tokens
    # Use (step+1)/T so that cos(pi/2) = 0 at step = T-1 → keep = N.
    frac_masked = math.cos(math.pi / 2 * (step + 1) / total_steps)
    n_keep = round(num_tokens * (1.0 - frac_masked))
    return max(1, min(n_keep, num_tokens))


# ---------------------------------------------------------------------------
# MaskGIT Transformer
# ---------------------------------------------------------------------------


@dataclass
class MaskGITConfig:
    """Configuration for MaskGIT training and inference.

    Training:
        The mask ratio is sampled per batch from a cosine distribution
        (see ``_cosine_mask_ratio``).  All masked positions are predicted
        and the cross-entropy loss is computed only on those positions.

    Inference:
        Iterative decoding with a cosine confidence schedule.
        At each of ``num_inference_steps`` steps, the model predicts
        all positions, the most confident predictions are kept, and
        the remainder are re-masked.
    """

    # Number of iterative decoding steps during inference.
    # 8–12 is typical for MaskGIT; fewer = faster but lower quality.
    num_inference_steps: int = 8

    # Temperature for computing token confidences during inference.
    # Lower = greedier selection.  1.0 is neutral.
    temperature: float = 1.0

    # Minimum number of tokens to unmask per inference step.
    # Prevents the schedule from being too conservative early on.
    min_tokens_per_step: int = 1

    # Which inference scheduler to use: ``"confidence"`` (top-k by model
    # confidence) or ``"halton"`` (fixed low-discrepancy spatial order).
    scheduler: str = "confidence"

    # Token grid shape (H, W) used to discretize the 2D Halton sequence.
    # ``token_grid_height * token_grid_width`` must equal the transformer's
    # ``target_token_len``.
    token_grid_height: int = 16
    token_grid_width: int = 16


class MaskGITTransformer(nn.Module):
    """Bidirectional transformer for MaskGIT token prediction.

    Architecture matches the upstream GPT ``Transformer`` but:
    - Uses **full bidirectional attention** (no causal mask) among target tokens.
    - Adds a ``[MASK]`` token embedding at index ``vocab_size``.
    - Conditioning tokens (image feature map) are always fully visible.

    The conditioning map is flattened and prepended to the token sequence,
    exactly as in the PrefixLM design.  This means the transformer sees::

        [cond_0, ..., cond_C | tok_0, ..., tok_{N-1}]

    where conditioning positions attend bidirectionally to each other and
    to all token positions, and token positions attend bidirectionally.
    """

    def __init__(self, config: GPTModelArgs) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.img_feature_code_len = config.img_feature_code_len
        self.target_token_len = config.target_token_len

        # Conditioning path (same as upstream PrefixLM).
        self.img_feature_embedding = ImgFeatureMapEmbedder(
            config.img_feature_channel,
            config.dim,
            config.feature_dropout_prob,
            config.img_feature_code_len,
        )

        # Token embeddings: +1 for the [MASK] token.
        self.tok_embeddings = nn.Embedding(config.vocab_size + 1, config.dim)
        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        # Transformer blocks (reuse upstream).
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )

        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis_1d(
            config.target_token_len,
            config.dim // config.n_head,
            config.rope_base,
            config.img_feature_code_len,
        )

        self.max_batch_size = -1
        self.max_seq_length = -1
        self.initialize_weights()

        # LoRA state.
        self._lora_injected: bool = False
        self._composed_lora: bool = False

    @property
    def mask_token_id(self) -> int:
        """The token id used for ``[MASK]``."""
        return self.vocab_size

    def initialize_weights(self) -> None:
        self.apply(self._init_weights)
        nn.init.constant_(self.output.weight, 0)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def _bidirectional_mask(self, total_len: int, device: torch.device) -> torch.Tensor:
        """Return an all-zeros attention mask (bidirectional) of shape ``(total_len, total_len)``.

        In ``F.scaled_dot_product_attention``, passing a mask disables
        the built-in causal behaviour (``is_causal=False`` is used when
        ``attn_mask`` is not ``None``).  Zeros mean "allow attention."
        """
        return torch.zeros(total_len, total_len, device=device)

    def forward(
        self,
        idx: torch.Tensor,
        imgs_feature_map: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with bidirectional attention.

        Args:
            idx: Token indices of shape ``(B, N)``.  Positions to be
                predicted should be set to ``mask_token_id``.
            imgs_feature_map: Conditioning feature map of shape
                ``(B, C, H, W)``, where ``H * W == img_feature_code_len``.

        Returns:
            Logits of shape ``(B, N, vocab_size)`` — one prediction per
            token position (including positions that were not masked).
        """
        batch_size = idx.shape[0]
        device = idx.device

        # Embed conditioning.
        img_embeddings = self.img_feature_embedding(
            imgs_feature_map, train=self.training
        )  # (B, C_tokens, dim)

        # Embed target tokens (including [MASK] tokens).
        tar_embeddings = self.tok_embeddings(idx)  # (B, N, dim)

        # Concatenate: [conditioning | tokens].
        token_embeddings = torch.cat((img_embeddings, tar_embeddings), dim=1)
        h = self.tok_dropout(token_embeddings)

        total_len = token_embeddings.shape[1]
        self.freqs_cis = self.freqs_cis.to(device)
        freqs_cis = self.freqs_cis[:total_len]

        # Bidirectional mask: all positions can attend to all positions.
        attn_mask = self._bidirectional_mask(total_len, device)

        for layer in self.layers:
            h = layer(h, freqs_cis, start_pos=None, mask=attn_mask)

        h = self.norm(h)
        logits = self.output(h).float()

        # Slice to target-token positions.  In the concatenated
        # [conditioning | tokens] sequence, position ``img_feature_code_len``
        # (0-indexed) predicts the first target token.
        logits = logits[:, self.img_feature_code_len :].contiguous()
        # logits shape: (B, N, vocab_size)
        return logits

    # ------------------------------------------------------------------
    # LoRA injection
    # ------------------------------------------------------------------

    def inject_lora(self, lora_config: "LoRAConfig") -> None:
        """Inject single LoRA adapters into linear layers of the transformer.

        Targets:
        - Attention QKV and output projections in every layer.
        - Feed-forward w1, w2, w3 projections in every layer.
        - Final output projection.

        Base weights are frozen; only the new LoRA matrices are trainable.
        May only be called once per transformer instance.

        Raises:
            RuntimeError: If LoRA has already been injected.
        """
        from hrothgar.ar.lora import LoRAConfig, LoRALinear

        if self._lora_injected:
            raise RuntimeError(
                "LoRA has already been injected into this transformer.  "
                "Create a fresh model before injecting again."
            )

        for i, layer in enumerate(self.layers):
            # Attention projections.
            layer.attention.wqkv = LoRALinear(
                layer.attention.wqkv, lora_config.rank, lora_config.alpha
            )
            layer.attention.wo = LoRALinear(
                layer.attention.wo, lora_config.rank, lora_config.alpha
            )
            # Feed-forward projections.
            layer.feed_forward.w1 = LoRALinear(
                layer.feed_forward.w1, lora_config.rank, lora_config.alpha
            )
            layer.feed_forward.w2 = LoRALinear(
                layer.feed_forward.w2, lora_config.rank, lora_config.alpha
            )
            layer.feed_forward.w3 = LoRALinear(
                layer.feed_forward.w3, lora_config.rank, lora_config.alpha
            )

        # Final output projection.
        self.output = LoRALinear(self.output, lora_config.rank, lora_config.alpha)
        self._lora_injected = True

    def inject_composed_lora(
        self,
        glyph_state_dict: "Dict[str, torch.Tensor]",
        lora_config: "LoRAConfig",
    ) -> None:
        """Inject composed LoRA: frozen glyph prior + trainable font adapter.

        Each linear target is replaced with a ``ComposedLoRALinear`` that
        holds both adapters.  The glyph adapter weights are loaded from
        *glyph_state_dict* (output of a previous GA run) and frozen.
        The font adapter is zero-initialised and trained during NFA.

        The glyph LoRA rank is cross-checked against the tensor shapes in
        *glyph_state_dict* and a ``ValueError`` is raised on mismatch.

        May only be called once per transformer instance.
        """
        from hrothgar.ar.lora import ComposedLoRALinear, LoRAConfig

        if self._lora_injected:
            raise RuntimeError(
                "LoRA has already been injected into this transformer.  "
                "Create a fresh model before injecting again."
            )

        def _make_composed(
            linear: nn.Linear, key_prefix: str
        ) -> "ComposedLoRALinear":
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
            glyph_scaling = lora_config.scaling
            return ComposedLoRALinear(
                linear,
                glyph_lora_A=glyph_A,
                glyph_lora_B=glyph_B,
                glyph_scaling=glyph_scaling,
                font_rank=lora_config.rank,
                font_alpha=lora_config.alpha,
            )

        for i, layer in enumerate(self.layers):
            prefix = f"layers.{i}"
            layer.attention.wqkv = _make_composed(
                layer.attention.wqkv, f"{prefix}.attention.wqkv"
            )
            layer.attention.wo = _make_composed(
                layer.attention.wo, f"{prefix}.attention.wo"
            )
            layer.feed_forward.w1 = _make_composed(
                layer.feed_forward.w1, f"{prefix}.feed_forward.w1"
            )
            layer.feed_forward.w2 = _make_composed(
                layer.feed_forward.w2, f"{prefix}.feed_forward.w2"
            )
            layer.feed_forward.w3 = _make_composed(
                layer.feed_forward.w3, f"{prefix}.feed_forward.w3"
            )

        self.output = _make_composed(self.output, "output")
        self._lora_injected = True
        self._composed_lora = True

    def get_lora_state_dict(self) -> "Dict[str, torch.Tensor]":
        """Return a state dict containing only trainable LoRA parameters.

        In single-adapter mode this returns all ``lora_A`` / ``lora_B`` keys.
        In composed mode, only the trainable *font* adapter keys
        (``lora_A_font`` / ``lora_B_font``) are returned.

        Raises:
            RuntimeError: If LoRA has not been injected.
        """
        if not self._lora_injected:
            raise RuntimeError(
                "Cannot get LoRA state dict — inject_lora() or "
                "inject_composed_lora() must be called first."
            )
        if self._composed_lora:
            return {
                k: v
                for k, v in self.state_dict().items()
                if "lora_A_font" in k or "lora_B_font" in k
            }
        return {
            k: v
            for k, v in self.state_dict().items()
            if "lora_A" in k or "lora_B" in k
        }

    def load_lora_state_dict(self, state_dict: "Dict[str, torch.Tensor]") -> None:
        """Load a LoRA state dict (produced by ``get_lora_state_dict``).

        The keys must match the current LoRA parameter names exactly.

        Raises:
            RuntimeError: If LoRA has not been injected.
        """
        if not self._lora_injected:
            raise RuntimeError(
                "Cannot load LoRA state dict — inject_lora() or "
                "inject_composed_lora() must be called first."
            )
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(
                f"Unexpected keys in LoRA state dict: {unexpected}"
            )
        if missing:
            raise RuntimeError(
                f"Missing keys when loading LoRA state dict: {missing}"
            )


# ---------------------------------------------------------------------------
# Inference samplers
# ---------------------------------------------------------------------------


def _build_halton_order(height: int, width: int) -> torch.Tensor:
    """Return a permutation of token positions (0..H*W-1) in Halton order.

    The order follows a 2D Halton (quasi-random, low-discrepancy) sequence
    with bases 2 and 3, discretized onto the ``height x width`` token grid.
    Revealing tokens in this order spreads each step's predictions uniformly
    across the image rather than clustering around high-confidence regions
    (see ``ConfidenceSampler``).  Any grid cells the sequence fails to cover
    are appended in raster-scan order so the result is a complete permutation.
    """
    n = height * width
    nb_point = max(10_000, n * 40)

    def _halton(base: int, count: int) -> list:
        # Naive radical-inverse (van der Corput) generator, matching the
        # Halton-MaskGIT reference implementation.
        numerator, denominator = 0, 1
        out: list = []
        for _ in range(count):
            x = denominator - numerator
            if x == 1:
                numerator = 1
                denominator *= base
            else:
                y = denominator // base
                while x <= y:
                    y //= base
                numerator = (base + 1) * y - x
            out.append(numerator / denominator)
        return out

    xs = torch.as_tensor(_halton(2, nb_point))
    ys = torch.as_tensor(_halton(3, nb_point))

    # Discretize: x -> column, y -> row, flatten in raster-scan (row-major).
    cols = torch.floor(xs * width).long()
    rows = torch.floor(ys * height).long()
    flat = rows * width + cols

    order: list = []
    seen: set = set()
    for idx in flat.tolist():
        if 0 <= idx < n and idx not in seen:
            seen.add(idx)
            order.append(idx)

    # Ensure every position is covered exactly once.
    for idx in range(n):
        if idx not in seen:
            order.append(idx)

    return torch.tensor(order, dtype=torch.long)


class MaskGITSampler:
    """Base class for MaskGIT iterative inference samplers.

    Subclasses implement ``_reveal``, which decides which still-masked token
    positions to commit at each inference step.  The shared ``generate`` loop
    runs the transformer, computes softmax confidence, commits tokens with
    ``argmax`` (hard decode), and applies the per-step reveal decisions.
    """

    def __init__(self, config: MaskGITConfig) -> None:
        self.config = config

    @torch.no_grad()
    def generate(
        self,
        transformer: MaskGITTransformer,
        conditioning_map: torch.Tensor,
    ) -> torch.Tensor:
        """Iteratively decode tokens, returning predicted indices ``(B, N)``."""
        batch_size = conditioning_map.shape[0]
        device = conditioning_map.device
        N = transformer.target_token_len
        T = self.config.num_inference_steps
        temperature = self.config.temperature
        mask_token_id = transformer.mask_token_id

        predicted = torch.full(
            (batch_size, N), mask_token_id, dtype=torch.long, device=device
        )
        unmasked = torch.zeros(batch_size, N, dtype=torch.bool, device=device)

        for step in range(T):
            logits = transformer(idx=predicted, imgs_feature_map=conditioning_map)
            probs = F.softmax(logits / temperature, dim=-1)
            pred_tokens = probs.max(dim=-1).indices  # (B, N) hard commit

            target_keep = _cosine_unmask_schedule(step + 1, T, N)
            reveal = self._reveal(probs, unmasked, target_keep)

            predicted[reveal] = pred_tokens[reveal]
            unmasked[reveal] = True

        # Safety net: fill any still-masked positions (the cosine schedule
        # reaches N at the last step, but this guards against rounding).
        remaining = ~unmasked
        if remaining.any():
            logits = transformer(idx=predicted, imgs_feature_map=conditioning_map)
            predicted[remaining] = torch.argmax(logits, dim=-1)[remaining]

        return predicted

    def _reveal(
        self,
        probs: torch.Tensor,
        unmasked: torch.Tensor,
        target_keep: int,
    ) -> torch.Tensor:
        """Return a boolean ``(B, N)`` mask of positions to commit this step."""
        raise NotImplementedError


class ConfidenceSampler(MaskGITSampler):
    """Original MaskGIT confidence scheduler.

    At each step, commit the ``target_keep - already_unmasked`` most-confident
    still-masked positions (confidence = softmax max-probability).
    """

    def _reveal(self, probs, unmasked, target_keep):
        batch_size = unmasked.shape[0]
        max_probs = probs.max(dim=-1).values  # (B, N)

        # Already-unmasked positions must not be re-selected.
        max_probs = torch.where(
            unmasked, torch.full_like(max_probs, float("-inf")), max_probs
        )

        n_already = unmasked.sum(dim=1)  # (B,)
        n_to_unmask = (target_keep - n_already).clamp(
            min=self.config.min_tokens_per_step
        )

        reveal = torch.zeros_like(unmasked)
        for b in range(batch_size):
            n = min(n_to_unmask[b].item(), (~unmasked[b]).sum().item())
            if n <= 0:
                continue
            _, top = torch.topk(max_probs[b], k=n)
            reveal[b, top] = True
        return reveal


class HaltonSampler(MaskGITSampler):
    """Halton low-discrepancy scheduler.

    Commits tokens in a fixed, spatially-uniform order derived from a 2D Halton
    sequence (bases 2 and 3).  Unlike the confidence scheduler, the order does
    not depend on the model's confidence, so each step reveals a uniform sample
    of the image and never clusters around high-confidence regions.  The number
    of tokens committed per step still follows the cosine schedule.
    """

    def __init__(self, config: MaskGITConfig) -> None:
        super().__init__(config)
        self._order = _build_halton_order(
            config.token_grid_height, config.token_grid_width
        )

    def _reveal(self, probs, unmasked, target_keep):
        device = unmasked.device
        N = unmasked.shape[1]
        # Reveal is a fixed prefix of the Halton order, identical across the
        # batch (``target_keep`` is a scalar).
        already = int(unmasked.sum(dim=1)[0].item())
        n = max(self.config.min_tokens_per_step, target_keep - already)
        n = min(n, N - already)

        if n <= 0:
            return torch.zeros_like(unmasked)

        positions = self._order[already : already + n].to(device)
        reveal = torch.zeros_like(unmasked)
        reveal[:, positions] = True
        return reveal


# ---------------------------------------------------------------------------
# MaskGIT Decoder (training + inference)
# ---------------------------------------------------------------------------


class MaskGITDecoder(nn.Module):
    """MaskGIT decoder wrapping a bidirectional transformer.

    Training:
        ``forward_train`` randomly masks token positions and returns
        logits for computing cross-entropy on masked positions only.

    Inference:
        ``generate`` delegates to a configured inference sampler (confidence
        or Halton) — no autoregressive sequential dependency.
    """

    def __init__(
        self,
        transformer: MaskGITTransformer,
        config: MaskGITConfig,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.config = config
        self.sequence_length = transformer.target_token_len
        self.codebook_size = transformer.vocab_size
        self.mask_token_id = transformer.mask_token_id

        grid_size = config.token_grid_height * config.token_grid_width
        if grid_size != self.sequence_length:
            raise ValueError(
                "MaskGIT config token grid "
                f"{config.token_grid_height}x{config.token_grid_width} does not "
                f"match sequence length {self.sequence_length}"
            )

        self.sampler = self._build_sampler(config)

    def _build_sampler(self, config: MaskGITConfig) -> MaskGITSampler:
        """Construct the inference sampler selected by ``config.scheduler``."""
        if config.scheduler == "confidence":
            return ConfidenceSampler(config)
        if config.scheduler == "halton":
            return HaltonSampler(config)
        raise ValueError(f"Unknown MaskGIT scheduler: {config.scheduler!r}")

    def _sample_mask_ratio(self) -> float:
        """Sample a per-batch mask ratio from the cosine distribution."""
        return _cosine_mask_ratio()

    def forward_train(
        self,
        target_token_indices: torch.Tensor,
        conditioning_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training forward pass: randomly mask tokens, predict all.

        Args:
            target_token_indices: Ground-truth token indices, ``(B, N)``.
            conditioning_map: Conditioning feature map, ``(B, C, H, W)``.

        Returns:
            ``(logits, mask)`` where:
            - ``logits``: ``(B, N, vocab_size)`` — predictions at all positions.
            - ``mask``: ``(B, N)`` boolean — ``True`` for positions that were
              masked (and should contribute to the loss).
        """
        batch_size, seq_len = target_token_indices.shape
        device = target_token_indices.device

        # Sample mask ratio and create per-position mask.
        mask_ratio = self._sample_mask_ratio()
        num_mask = max(1, int(seq_len * mask_ratio))
        # Randomly select positions to mask (without replacement per batch item).
        rand = torch.rand(batch_size, seq_len, device=device)
        # For each row, the positions with the *smallest* random values are masked.
        rand_sorted = rand.argsort(dim=-1)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        mask.scatter_(1, rand_sorted[:, :num_mask], True)

        # Replace masked positions with [MASK] token.
        input_ids = target_token_indices.clone()
        input_ids[mask] = self.mask_token_id

        logits = self.transformer(idx=input_ids, imgs_feature_map=conditioning_map)
        return logits, mask

    def generate(self, conditioning_map: torch.Tensor) -> torch.Tensor:
        """Iteratively decode tokens via the configured inference sampler.

        Args:
            conditioning_map: Conditioning feature map, ``(B, C, H, W)``.

        Returns:
            Predicted token indices of shape ``(B, N)``.
        """
        return self.sampler.generate(self.transformer, conditioning_map)


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaskGITLossWeights:
    """Weights for the MaskGIT training objectives."""

    token_cross_entropy: float = 0.3
    pixel_l1: float = 1.0
    perceptual_lpips: float = 2.0


def compute_maskgit_loss(
    logits: torch.Tensor,
    token_mask: torch.Tensor,
    target_token_indices: torch.Tensor,
    reconstructed_images: torch.Tensor,
    target_images: torch.Tensor,
    *,
    weights: MaskGITLossWeights = MaskGITLossWeights(),
    lpips_metric: Optional[object] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute MaskGIT training loss.

    Token cross-entropy is computed **only on masked positions**, matching
    the MaskGIT training objective.  Pixel L1 and LPIPS are computed on
    the soft-decoded reconstruction as auxiliary signals.

    Args:
        logits: Model logits, shape ``(B, N, vocab_size)``.
        token_mask: Boolean mask, shape ``(B, N)`` — ``True`` for positions
            that were masked during training.
        target_token_indices: Ground-truth token indices, ``(B, N)``.
        reconstructed_images: Soft-decoded images from ``soft_decode()``.
        target_images: Ground-truth images, ``(B, 3, H, W)``.
        weights: Loss weight configuration.
        lpips_metric: LPIPS module instance (required if ``perceptual_lpips > 0``).

    Returns:
        ``(total_loss, terms)`` where terms is a dict of scalar tensors
        suitable for TensorBoard logging.
    """
    device = target_images.device

    # Token CE on masked positions only.
    n_masked = token_mask.sum()
    if n_masked > 0:
        # Gather logits and targets at masked positions.
        masked_logits = logits[token_mask]  # (n_masked, K)
        masked_targets = target_token_indices[token_mask]  # (n_masked,)
        token_cross_entropy = F.cross_entropy(masked_logits, masked_targets)

        # Token accuracy on masked positions.
        token_predictions = torch.argmax(logits, dim=-1)
        token_accuracy = (
            (token_predictions[token_mask] == masked_targets).float().mean()
        )
    else:
        token_cross_entropy = torch.tensor(0.0, device=device)
        token_accuracy = torch.tensor(0.0, device=device)

    # Pixel L1 on the full reconstruction.
    pixel_l1 = F.l1_loss(reconstructed_images, target_images)

    # Perceptual LPIPS on soft-decoded reconstruction.
    perceptual_lpips = torch.tensor(0.0, device=device)
    if weights.perceptual_lpips > 0:
        if lpips_metric is None:
            raise ValueError(
                "lpips_metric is required when perceptual_lpips > 0"
            )
        recon_clamped = torch.clamp(reconstructed_images, 0.0, 1.0)
        target_clamped = torch.clamp(target_images, 0.0, 1.0)
        perceptual_lpips = lpips_metric(recon_clamped, target_clamped).mean()

    weighted_token_ce = weights.token_cross_entropy * token_cross_entropy
    weighted_pixel_l1 = weights.pixel_l1 * pixel_l1
    weighted_perceptual_lpips = weights.perceptual_lpips * perceptual_lpips

    total_loss = weighted_token_ce + weighted_pixel_l1 + weighted_perceptual_lpips

    terms: dict[str, torch.Tensor] = {
        "total": total_loss.detach(),
        "token_cross_entropy": token_cross_entropy.detach(),
        "pixel_l1": pixel_l1.detach(),
        "perceptual_lpips": perceptual_lpips.detach(),
        "token_accuracy": token_accuracy.detach(),
        "weighted_token_cross_entropy": weighted_token_ce.detach(),
        "weighted_pixel_l1": weighted_pixel_l1.detach(),
        "weighted_perceptual_lpips": weighted_perceptual_lpips.detach(),
        "n_masked": torch.tensor(float(n_masked), device=device),
    }

    return total_loss, terms
