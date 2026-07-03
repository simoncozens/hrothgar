"""Diffusion Transformer (DiT) for glyph generation.

Adapted from the official DiT implementation (facebookresearch/dit) for our
G-Tok codebook embedding space.

Architecture:
  - Input:  noisy codebook embeddings, shape (B, 64, 16) — 8×8 grid flattened.
  - Output: predicted noise ε, same shape.
  - Conditioning injected via adaLN-Zero in each DiT block:
      · Timestep t  → sinusoidal → MLP → hidden_size
      · Glyph identity → codepoint embedding → Linear → hidden_size
      · Style         → style features      → Linear → hidden_size
      · Combined: c = t_emb + codepoint_emb + style_emb

References:
  - Peebles & Xie, "Scalable Diffusion Models with Transformers", ICCV 2023.
  - Official implementation: https://github.com/facebookresearch/dit
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# adaLN helper
# ---------------------------------------------------------------------------


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer norm modulation.

    ``x * (1 + scale) + shift``, where scale and shift are broadcast
    from ``(B, D)`` to ``(B, N, D)``.
    """
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DiTConfig:
    """Configuration for the GlyphDiT model.

    The default sizes approximate DiT-B (~130M params) but with 16 heads
    and 832 hidden dim to match our existing 141M budget.
    """

    # Transformer backbone.
    hidden_size: int = 832
    depth: int = 16
    num_heads: int = 16
    mlp_ratio: float = 4.0

    # Token sequence.
    num_tokens: int = 64  # 8×8 G-Tok grid
    token_dim: int = 16  # G-Tok codebook embedding dimension

    # Diffusion process.
    num_diffusion_steps: int = 1000
    noise_schedule: str = "squaredcos_cap_v2"  # cosine schedule

    # DDIM inference.
    ddim_steps: int = 250

    # Conditioning dimensions.
    codepoint_embedding_dim: int = 256
    style_feature_dim: int = 256

    # Conditioning dropout for classifier-free guidance on style.
    style_dropout_prob: float = 0.1

    # Attention dropout (applied inside scaled_dot_product_attention).
    attention_dropout: float = 0.0


# ---------------------------------------------------------------------------
# Embedding layers
# ---------------------------------------------------------------------------


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations.

    Projects through sinusoidal frequencies then a 2-layer MLP with SiLU.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def _timestep_embedding(
        t: torch.Tensor, dim: int, max_period: float = 10000.0
    ) -> torch.Tensor:
        """Sinusoidal timestep embeddings (as in Transformer position encoding).

        Args:
            t: (N,) tensor of integer or fractional timesteps.
            dim: Output embedding dimension.
            max_period: Minimum frequency control.

        Returns:
            (N, dim) tensor of sin/cos features.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 != 0:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self._timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class ConditioningEmbedder(nn.Module):
    """Combines codepoint identity and style features into a single
    conditioning vector used by every DiT block.

    Codepoint embedding is always provided (the model must know which
    glyph to generate).  Style features are randomly dropped during
    training with probability ``style_dropout_prob`` to support
    classifier-free guidance at inference.
    """

    def __init__(
        self,
        hidden_size: int,
        codepoint_embedding_dim: int,
        style_feature_dim: int,
        style_dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.style_dropout_prob = style_dropout_prob

        self.codepoint_proj = nn.Sequential(
            nn.Linear(codepoint_embedding_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.style_proj = nn.Sequential(
            nn.Linear(style_feature_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        # Learnable null-style embedding for dropout.
        self.null_style = nn.Parameter(torch.randn(hidden_size) / hidden_size**0.5)

    def forward(
        self,
        codepoint_emb: torch.Tensor,
        style_features: torch.Tensor,
        force_style_drop: bool = False,
    ) -> torch.Tensor:
        """Produce conditioning vector.

        Args:
            codepoint_emb: ``(B, codepoint_embedding_dim)`` — always used.
            style_features: ``(B, style_feature_dim)`` — may be dropped.
            force_style_drop: If True, always replace style with null embedding
                (used for classifier-free guidance during inference).

        Returns:
            ``(B, hidden_size)`` conditioning vector.
        """
        c_cp = self.codepoint_proj(codepoint_emb)

        drop = force_style_drop or (
            self.training
            and self.style_dropout_prob > 0
            and torch.rand(1, device=codepoint_emb.device).item()
            < self.style_dropout_prob
        )
        if drop:
            c_style = self.null_style.unsqueeze(0).expand(c_cp.shape[0], -1)
        else:
            c_style = self.style_proj(style_features)

        return c_cp + c_style


# ---------------------------------------------------------------------------
# DiT blocks
# ---------------------------------------------------------------------------


class DiTBlock(nn.Module):
    """A DiT block with adaLN-Zero conditioning.

    Each block has self-attention + MLP, both modulated by the
    conditioning vector ``c``.  The modulation weights are
    zero-initialised so the model starts as an identity function.
    """

    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = (
            x
            + gate_msa.unsqueeze(1)
            * self.attn(
                modulate(self.norm1(x), shift_msa, scale_msa),
                modulate(self.norm1(x), shift_msa, scale_msa),
                modulate(self.norm1(x), shift_msa, scale_msa),
                need_weights=False,
            )[0]
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """Final DiT layer: norm → adaLN → linear projection to token_dim."""

    def __init__(self, hidden_size: int, token_dim: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, token_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ---------------------------------------------------------------------------
# 2D sinusoidal position embeddings
# ---------------------------------------------------------------------------


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Return 2D sinusoidal position embeddings for a square grid.

    Args:
        embed_dim: Output dimension (must be divisible by 2).
        grid_size: Grid height/width (e.g. 8 for an 8×8 token grid).

    Returns:
        ``(grid_size * grid_size, embed_dim)`` position embedding tensor.
    """
    assert embed_dim % 2 == 0

    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    grid = torch.stack(grid, dim=0)  # (2, grid_size, grid_size)

    # Use half the dims for height, half for width.
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (N, D/2)
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (N, D/2)
    return torch.cat([emb_h, emb_w], dim=1)  # (N, D)


def _get_1d_sincos_pos_embed_from_grid(
    embed_dim: int, pos: torch.Tensor
) -> torch.Tensor:
    """1D sinusoidal position embedding for a grid of positions.

    Args:
        embed_dim: Output dimension (must be even).
        pos: ``(H, W)`` grid of positions.

    Returns:
        ``(H * W, embed_dim)`` embedding tensor.
    """
    assert embed_dim % 2 == 0

    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = omega / (embed_dim / 2.0)
    omega = 1.0 / (10000.0**omega)  # (D/2,)

    pos = pos.reshape(-1).float()  # (N,)
    out = torch.outer(pos, omega)  # (N, D/2)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    return torch.cat([emb_sin, emb_cos], dim=1)


# ---------------------------------------------------------------------------
# Main DiT model
# ---------------------------------------------------------------------------


class GlyphDiT(nn.Module):
    """Diffusion Transformer for glyph token generation.

    Operates in the continuous G-Tok codebook embedding space (64 × 16).
    The transformer backbone uses adaLN-Zero conditioning for timestep,
    codepoint identity, and style features.
    """

    def __init__(self, config: DiTConfig) -> None:
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size

        # Input projection: 16-dim codebook embedding → hidden_size.
        self.x_embedder = nn.Linear(config.token_dim, hidden_size, bias=True)

        # 2D sinusoidal position embeddings (frozen).
        pos_embed = _get_2d_sincos_pos_embed(
            hidden_size, int(config.num_tokens**0.5)
        )  # (N, hidden_size)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)

        # Timestep embedder.
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Conditioning embedder (codepoint + style).
        self.c_embedder = ConditioningEmbedder(
            hidden_size=hidden_size,
            codepoint_embedding_dim=config.codepoint_embedding_dim,
            style_feature_dim=config.style_feature_dim,
            style_dropout_prob=config.style_dropout_prob,
        )

        # Transformer blocks.
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, config.num_heads, config.mlp_ratio)
                for _ in range(config.depth)
            ]
        )

        # Final output layer.
        self.final_layer = FinalLayer(hidden_size, config.token_dim)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize weights following the DiT paper recipe."""

        # Basic init for all linear layers.
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Input embedder.
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.constant_(self.x_embedder.bias, 0)

        # Timestep embedder MLP.
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Conditioning embedder projections.
        for proj in [self.c_embedder.codepoint_proj, self.c_embedder.style_proj]:
            nn.init.normal_(proj[0].weight, std=0.02)
            nn.init.normal_(proj[2].weight, std=0.02)

        # Zero-out adaLN modulation layers (adaLN-Zero init).
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out final layer.
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        codepoint_emb: torch.Tensor,
        style_features: torch.Tensor,
        force_style_drop: bool = False,
    ) -> torch.Tensor:
        """Predict the noise ε added to x_0 to produce x_t.

        Args:
            x_t: Noisy embeddings, shape ``(B, N, token_dim)``.
            t: Diffusion timesteps, shape ``(B,)``.
            codepoint_emb: Codepoint identity, ``(B, D_cp)``.
            style_features: Style features, ``(B, D_style)``.
            force_style_drop: If True, replace style with null embedding
                (for classifier-free guidance).

        Returns:
            Predicted noise ``ε_θ``, shape ``(B, N, token_dim)``.
        """
        # Input embedding + position.
        x = self.x_embedder(x_t) + self.pos_embed  # (B, N, hidden_size)

        # Conditioning.
        t_emb = self.t_embedder(t)  # (B, hidden_size)
        c_emb = self.c_embedder(
            codepoint_emb, style_features, force_style_drop=force_style_drop
        )  # (B, hidden_size)
        c = t_emb + c_emb  # (B, hidden_size)

        # Transformer blocks.
        for block in self.blocks:
            x = block(x, c)

        # Final projection → predicted noise.
        x = self.final_layer(x, c)  # (B, N, token_dim)
        return x


# ---------------------------------------------------------------------------
# Noise schedule helpers
# ---------------------------------------------------------------------------


def _cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule (``squaredcos_cap_v2`` from improved DDPM).

    Returns betas of shape ``(num_steps,)``.

    Args:
        num_steps: Number of diffusion timesteps.
        s: Small offset to prevent singularities at t=0.
    """
    steps = num_steps + 1
    t = torch.linspace(0, num_steps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos((t / num_steps + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clamp(betas, max=0.999).float()


def _linear_beta_schedule(
    num_steps: int, beta_start: float = 1e-4, beta_end: float = 0.02
) -> torch.Tensor:
    """Linear noise schedule (from DDPM).

    Returns betas of shape ``(num_steps,)``.
    """
    scale = 1000.0 / num_steps
    return torch.linspace(
        scale * beta_start, scale * beta_end, num_steps, dtype=torch.float32
    )


def get_beta_schedule(name: str, num_steps: int) -> torch.Tensor:
    """Return beta schedule tensor for the given name."""
    if name == "squaredcos_cap_v2":
        return _cosine_beta_schedule(num_steps)
    elif name == "linear":
        return _linear_beta_schedule(num_steps)
    else:
        raise ValueError(f"Unknown noise schedule: {name}")


# ---------------------------------------------------------------------------
# Diffusion process helpers
# ---------------------------------------------------------------------------


class NoiseScheduler(nn.Module):
    """Pre-computed noise schedule values for training and DDIM sampling.

    Subclasses :class:`nn.Module` so that its tensors (registered as buffers)
    automatically move to the correct device when the parent model calls
    ``.to(device)``.
    """

    def __init__(self, betas: torch.Tensor) -> None:
        super().__init__()

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # (T,)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), alphas_cumprod[:-1]]
        )  # (T,)

        # Register as buffers so they move with the model.
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt()
        )

        self.num_steps = len(betas)

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward diffusion: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε.

        Args:
            x_start: Clean embeddings, shape ``(B, N, D)``.
            t: Timesteps, shape ``(B,)`` with values in [0, T-1].
            noise: Optional pre-sampled noise.

        Returns:
            Noisy embeddings, shape ``(B, N, D)``.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def predict_x0_from_eps(
        self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        """Recover x_0 from noisy x_t and predicted noise ε."""
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return (x_t - sqrt_one_minus * eps) / sqrt_alpha


# ---------------------------------------------------------------------------
# DDIM sampler
# ---------------------------------------------------------------------------


@torch.no_grad()
def ddim_sample(
    model: GlyphDiT,
    scheduler: NoiseScheduler,
    shape: tuple[int, int, int],
    codepoint_emb: torch.Tensor,
    style_features: torch.Tensor,
    ddim_steps: int,
    cfg_scale: float = 1.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """DDIM sampling loop.

    Args:
        model: Trained GlyphDiT model (in eval mode).
        scheduler: Pre-computed noise schedule.
        shape: ``(B, N, D)`` output shape.
        codepoint_emb: Codepoint identity, ``(B, D_cp)``.
        style_features: Style features, ``(B, D_style)``.
        ddim_steps: Number of DDIM steps.
        cfg_scale: Classifier-free guidance scale (> 1.0 amplifies style
            conditioning relative to unconditional).
        device: Target device.

    Returns:
        Denoised embeddings ``x_0``, shape ``(B, N, D)``.
    """
    if device is None:
        device = next(model.parameters()).device

    B, N, D = shape
    T = scheduler.num_steps

    # Ensure scheduler buffers are on the target device.
    scheduler.to(device)

    # Sample initial noise.
    x_t = torch.randn(B, N, D, device=device)

    # Uniformly spaced timesteps from T-1 down to 0.
    times = torch.linspace(T - 1, 0, ddim_steps, dtype=torch.long, device=device)

    for i in range(ddim_steps - 1):
        t = times[i]  # current timestep
        t_next = times[i + 1]  # next timestep (closer to 0)
        t_batch = torch.full((B,), t, dtype=torch.long, device=device)
        t_next_batch = torch.full((B,), t_next, dtype=torch.long, device=device)

        # Classifier-free guidance.
        if cfg_scale > 1.0:
            eps_cond = model(x_t, t_batch, codepoint_emb, style_features)
            eps_uncond = model(
                x_t, t_batch, codepoint_emb, style_features, force_style_drop=True
            )
            eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        else:
            eps = model(x_t, t_batch, codepoint_emb, style_features)

        # Predict x_0 from eps.
        x0_pred = scheduler.predict_x0_from_eps(x_t, t_batch, eps)

        # DDIM update: x_{t_next} = √ᾱ_{t_next} · x̂_0  + √(1-ᾱ_{t_next}) · ε
        alpha_next = scheduler.alphas_cumprod[t_next].view(-1, 1, 1)
        sqrt_one_minus_alpha_next = (1.0 - alpha_next).sqrt()
        x_t = alpha_next.sqrt() * x0_pred + sqrt_one_minus_alpha_next * eps
    # Final step: predict x_0 from noise at t=0.
    t_batch = torch.full((B,), times[-1], dtype=torch.long, device=device)
    if cfg_scale > 1.0:
        eps_cond = model(x_t, t_batch, codepoint_emb, style_features)
        eps_uncond = model(
            x_t, t_batch, codepoint_emb, style_features, force_style_drop=True
        )
        eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
    else:
        eps = model(x_t, t_batch, codepoint_emb, style_features)

    x0 = scheduler.predict_x0_from_eps(x_t, t_batch, eps)
    return x0
