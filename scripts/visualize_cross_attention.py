#!/usr/bin/env python3
"""Visualize cross-attention weights in the FeatureFusionModule.

Hooks into every ``StyleAttentionBlock`` inside the ``FeatureFusionModule``
and captures the multi-head cross-attention matrices produced when content
features (target glyph) attend to style features (reference glyphs).

Generates a multi-panel figure:
  1. **Token grid overlay** — the glyph image with 16×16 token cells coloured
     by per-token attention entropy (how focused or spread-out each content
     token's attention is).
  2. **Per-content-token style heatmaps** — for a user-selected content token,
     a heatmap over each reference glyph showing which spatial regions that
     token attends to most.
  3. **Content-token × content-token correlation matrix** — cosine similarity
     between the attention vectors of every pair of content tokens.  This is
     the primary diagnostic: if symmetric-pair tokens (e.g. both ends of a
     crossbar) show low correlation, that's the "smoking gun" for fragmented
     style conditioning.
  4. **Per-head, per-block attention heatmaps** — raw attention matrices
     aggregated across reference glyphs.

Usage::

    python scripts/visualize_cross_attention.py \\
        --font-path Font.ttf \\
        --target-char A \\
        --gtok-model-path models/gtok.pth \\
        --ar-model-path models/ar.pth \\
        --dataset-path ~/google-fonts
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize

# Project imports.
from hrothgar.ar.config import ARModelConfig
from hrothgar.ar.model import ARModel
from hrothgar.googlefonts import GoogleFont, find_google_font_by_basename
from hrothgar.gtok.model import load_model as load_gtok_model
from hrothgar.ar.style_sampling import (
    _font_has_codepoint,
    _has_non_empty_glyph,
    _is_blank_rendering,
)
from hrothgar.dataset_constants import LATIN_KERNEL
from hrothgar.utils import pick_device

matplotlib.use("Agg")  # non-interactive backend


# ---------------------------------------------------------------------------
# Attention capture
# ---------------------------------------------------------------------------

class AttentionCapture:
    """Registers forward hooks to capture cross-attention weights.

    Hooks into every ``StyleAttentionBlock.forward`` in the FeatureFusionModule
    and, if style pooling is active, also hooks ``StyleAttentionPool.forward``.
    """

    def __init__(self, aggregator: nn.Module):
        self.aggregator = aggregator
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        # captured_blocks[block_idx] = (B, n_heads, Q, K) — content→style attention
        self.captured_blocks: dict[int, torch.Tensor] = {}
        # captured_pool = (B, pool_n_heads, n_tokens, N_style) — learned queries→style positions
        self.captured_pool: Optional[torch.Tensor] = None
        self._uses_pooling: bool = aggregator.style_pool is not None

    # ---- StyleAttentionBlock hooks ----------------------------------------

    def _make_block_hook(self, block_idx: int):
        block = self.aggregator.style_blocks[block_idx]

        def hook_fn(module, args, kwargs, output):
            if "query_feat" in kwargs:
                query_feat = kwargs["query_feat"]
                key_feat = kwargs["key_feat"]
                value_feat = kwargs["value_feat"]
            else:
                query_feat, key_feat, value_feat = args
            Q = block.q_proj(query_feat)
            K = block.k_proj(key_feat)
            V = block.v_proj(value_feat)  # noqa: F841

            n_heads = block.n_heads
            head_dim = block.head_dim
            len_q = Q.shape[1]
            len_k = K.shape[1]

            Q_ = Q.view(-1, len_q, n_heads, head_dim).transpose(1, 2)
            K_ = K.view(-1, len_k, n_heads, head_dim).transpose(1, 2)
            attn_scores = torch.matmul(Q_, K_.transpose(-2, -1)) / (head_dim**0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)  # (B, n_heads, Q, K)
            self.captured_blocks[block_idx] = attn_weights.detach().cpu()
        return hook_fn

    # ---- StyleAttentionPool hook ------------------------------------------

    def _make_pool_hook(self):
        pool = self.aggregator.style_pool

        def hook_fn(module, args, kwargs, output):
            # args[0] is style_seq: (B, N_style, C)
            style_seq = args[0] if args else kwargs["style_seq"]
            B, N_style, C = style_seq.shape

            queries = pool.query_tokens.expand(B, -1, -1)
            Q = pool.q_proj(queries)
            K = pool.k_proj(style_seq)

            n_heads = pool.n_heads
            head_dim = pool.head_dim
            n_tokens = pool.n_tokens

            Q_ = Q.view(B, n_tokens, n_heads, head_dim).transpose(1, 2)
            K_ = K.view(B, N_style, n_heads, head_dim).transpose(1, 2)
            attn_scores = torch.matmul(Q_, K_.transpose(-2, -1)) / (head_dim**0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)  # (B, n_heads, n_tokens, N_style)
            self.captured_pool = attn_weights.detach().cpu()
        return hook_fn

    # ---- Registration -----------------------------------------------------

    def register(self) -> None:
        for i, block in enumerate(self.aggregator.style_blocks):
            handle = block.register_forward_hook(
                self._make_block_hook(i), with_kwargs=True
            )
            self.handles.append(handle)

        if self._uses_pooling:
            handle = self.aggregator.style_pool.register_forward_hook(
                self._make_pool_hook(), with_kwargs=True
            )
            self.handles.append(handle)

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    @property
    def uses_pooling(self) -> bool:
        return self._uses_pooling


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_glyph_codepoint(
    font: GoogleFont,
    codepoint: int,
    size: int,
) -> np.ndarray:
    """Render a codepoint from a GoogleFont as an RGB float32 [0,1] array."""
    try:
        return font.render(codepoint, size=size)
    except Exception:
        return np.ones((3, size, size), dtype=np.float32)


def _attention_entropy(attn: torch.Tensor) -> np.ndarray:
    """Per-query entropy of attention distribution.  Higher = less focused."""
    # attn: (B, n_heads, Q, K)  or averaged across heads → (B, Q, K)
    eps = 1e-8
    entropy = -(attn * (attn + eps).log()).sum(dim=-1)  # (..., Q)
    return entropy.cpu().numpy()


def _cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity of rows of *vectors*."""
    v = vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-8)
    return v @ v.T


# ---------------------------------------------------------------------------
# Main visualisation
# ---------------------------------------------------------------------------

def build_figure(
    *,
    glyph_image: np.ndarray,
    style_images: np.ndarray,  # (n_ref, 3, H, W)
    style_codepoints: list[int],
    captured_blocks: dict[int, torch.Tensor],  # block_idx → (1, n_heads, Q, K)
    captured_pool: Optional[torch.Tensor],  # (1, pool_heads, n_tokens, N_style) or None
    grid_size: int,
    n_ref: int,
    content_token_idx: Optional[int] = None,
    zero_aggregator: bool = False,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Build the multi-panel visualisation figure.

    When style pooling is active (``captured_pool`` is not None), panels D–F
    show per-global-token heatmaps over reference glyphs using the pool's
    attention weights, and panel C shows content→token attention.
    """
    Q = grid_size * grid_size
    K_total = n_ref * Q

    # ── Aggregate attention across blocks and heads ───────────────────
    n_blocks = len(captured_blocks)
    all_attn = torch.stack([captured_blocks[i] for i in range(n_blocks)], dim=0)
    attn_mean = all_attn.mean(dim=(0, 2))  # (1, Q, K_total_or_n_tokens)
    attn_np = attn_mean.squeeze(0).numpy()

    uses_pooling = captured_pool is not None
    n_tokens = captured_pool.shape[2] if uses_pooling else 0

    # ── Pick default content token ────────────────────────────────────
    if content_token_idx is None:
        entropies = _attention_entropy(attn_mean.squeeze(0))
        content_token_idx = int(np.argmax(entropies))

    # ── Figure setup ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2], hspace=0.35, wspace=0.35)

    # ----- Panel A: Glyph with token grid overlay + entropy colours ----
    ax_a = fig.add_subplot(gs[0, 0])
    _draw_token_grid_overlay(
        ax=ax_a,
        glyph_image=glyph_image,
        grid_size=grid_size,
        attn_np=attn_np,
        highlight_idx=content_token_idx,
    )

    # ----- Panel B: Content × Content attention correlation matrix -----
    ax_b = fig.add_subplot(gs[0, 1])
    _draw_content_correlation_matrix(ax=ax_b, attn_np=attn_np, grid_size=grid_size)

    # ----- Panel C: Attention summary ----------------------------------
    ax_c = fig.add_subplot(gs[0, 2])
    if uses_pooling:
        # Content token → global tokens.
        token_attn = attn_np[content_token_idx]  # (n_tokens,)
        _draw_token_attention_bars(
            ax=ax_c,
            values=token_attn,
            labels=[f"Tok {i}" for i in range(n_tokens)],
            title=f"Token {content_token_idx} → global style tokens",
        )
    else:
        # Content token → reference glyphs.
        K_per_ref = Q
        _draw_style_attention_summary(
            ax=ax_c,
            attn_np=attn_np,
            content_token_idx=content_token_idx,
            n_ref=n_ref,
            style_codepoints=style_codepoints,
        )

    # ----- Panels D–F: Per-global-token or per-reference heatmaps -------
    if uses_pooling:
        # Show pool attention: each global token → style reference glyphs.
        pool_attn = captured_pool.squeeze(0).mean(dim=0)  # (n_tokens, N_style)
        pool_attn_np = pool_attn.numpy()  # (n_tokens, n_ref*Q)
        _draw_pool_heatmaps(
            fig=fig,
            gs=gs,
            pool_attn_np=pool_attn_np,
            style_images=style_images,
            style_codepoints=style_codepoints,
            grid_size=grid_size,
            n_ref=n_ref,
            n_tokens=n_tokens,
        )
    else:
        # Per-reference heatmaps for the selected content token.
        K_per_ref = Q
        n_ref_cols = min(n_ref, 3)
        n_ref_rows = max(1, math.ceil(n_ref / 3))
        gs_ref = fig.add_gridspec(
            n_ref_rows, n_ref_cols,
            left=gs[1, :].get_position(fig).x0,
            right=gs[1, :].get_position(fig).x1,
            bottom=gs[1, :].get_position(fig).y0,
            top=gs[1, :].get_position(fig).y1,
            hspace=0.4, wspace=0.25,
        )
        for ref_i in range(n_ref):
            row, col = divmod(ref_i, 3)
            ax_ref = fig.add_subplot(gs_ref[row, col])
            _draw_per_reference_heatmap(
                ax=ax_ref,
                style_image=style_images[ref_i],
                grid_size=grid_size,
                token_attn=attn_np[content_token_idx, ref_i * K_per_ref : (ref_i + 1) * K_per_ref],
                ref_label=f"U+{style_codepoints[ref_i]:04X}",
            )
        for ref_i in range(n_ref, n_ref_rows * n_ref_cols):
            row, col = divmod(ref_i, 3)
            ax_ref = fig.add_subplot(gs_ref[row, col])
            ax_ref.axis("off")

    # ----- Title -------------------------------------------------------
    agg_note = " [AGGREGATOR ZEROED]" if zero_aggregator else ""
    pool_note = f" | {n_tokens} pooled style tokens" if uses_pooling else ""
    fig.suptitle(
        f"FeatureFusionModule Cross-Attention Diagnostics{agg_note}{pool_note}\n"
        f"Highlighted content token: {content_token_idx} "
        f"(row={content_token_idx // grid_size}, col={content_token_idx % grid_size})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved visualisation to {output_path}")
    return fig


# ---------------------------------------------------------------------------
# Panel drawing helpers
# ---------------------------------------------------------------------------

def _draw_token_grid_overlay(
    ax: plt.Axes,
    glyph_image: np.ndarray,
    grid_size: int,
    attn_np: np.ndarray,
    highlight_idx: int,
) -> None:
    """Draw glyph image with token grid coloured by attention entropy."""
    H, W = glyph_image.shape
    ax.imshow(glyph_image, cmap="gray", extent=[0, W, H, 0], interpolation="bilinear")

    # Compute per-token entropy for colouring.
    entropies = _attention_entropy(torch.from_numpy(attn_np))
    entropies_2d = entropies.reshape(grid_size, grid_size)
    norm = Normalize(vmin=entropies_2d.min(), vmax=entropies_2d.max())

    cell = H / grid_size
    for r in range(grid_size):
        for c in range(grid_size):
            ent = entropies_2d[r, c]
            color = plt.cm.plasma(norm(ent))
            rect = Rectangle(
                (c * cell, r * cell), cell, cell,
                linewidth=1.0, edgecolor=color, facecolor=color, alpha=0.35,
            )
            ax.add_patch(rect)

    # Highlight selected token.
    hr, hc = divmod(highlight_idx, grid_size)
    rect = Rectangle(
        (hc * cell, hr * cell), cell, cell,
        linewidth=2.5, edgecolor="cyan", facecolor="none",
    )
    ax.add_patch(rect)

    # Token index labels (small font).
    for r in range(grid_size):
        for c in range(grid_size):
            idx = r * grid_size + c
            ax.text(
                c * cell + cell / 2, r * cell + cell / 2,
                str(idx), ha="center", va="center",
                fontsize=5, color="white", weight="bold",
            )

    ax.set_title("Token grid + attention entropy\n(plasma: high entropy = unfocused)", fontsize=9)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    # Colour bar.
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="Entropy")


def _draw_content_correlation_matrix(
    ax: plt.Axes,
    attn_np: np.ndarray,
    grid_size: int,
) -> None:
    """Content-token × content-token cosine similarity of attention vectors."""
    corr = _cosine_similarity_matrix(attn_np)  # (Q, Q)
    im = ax.imshow(corr, cmap="RdYlBu_r", aspect="equal", vmin=0.5, vmax=1.0,
                   interpolation="nearest")
    ax.set_title("Content token × content token\nattention correlation", fontsize=9)
    ax.set_xlabel("Content token index")
    ax.set_ylabel("Content token index")

    # Grid lines every `grid_size` tokens (row boundary).
    for i in range(1, grid_size):
        ax.axhline(i * grid_size - 0.5, color="white", linewidth=0.5)
        ax.axvline(i * grid_size - 0.5, color="white", linewidth=0.5)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cosine similarity")


def _draw_style_attention_summary(
    ax: plt.Axes,
    attn_np: np.ndarray,
    content_token_idx: int,
    n_ref: int,
    style_codepoints: list[int],
) -> None:
    """Bar chart: total attention mass on each style reference for the selected token."""
    Q_per_ref = attn_np.shape[1] // n_ref
    masses = []
    for i in range(n_ref):
        mass = float(attn_np[content_token_idx, i * Q_per_ref : (i + 1) * Q_per_ref].sum())
        masses.append(mass)

    labels = [f"U+{cp:04X}" for cp in style_codepoints]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_ref))
    bars = ax.bar(range(n_ref), masses, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(n_ref))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Attention mass")
    ax.set_title(
        f"Token {content_token_idx} — attention to each reference",
        fontsize=9,
    )

    # Annotate with percentages.
    total = sum(masses)
    for bar, mass in zip(bars, masses):
        pct = mass / total * 100 if total > 0 else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=7,
        )


def _draw_per_reference_heatmap(
    ax: plt.Axes,
    style_image: np.ndarray,
    grid_size: int,
    token_attn: np.ndarray,  # (K_per_ref,)
    ref_label: str,
) -> None:
    """Overlay attention heatmap on a single style reference glyph.

    *style_image* is ``(3, H, W)`` or ``(H, W)`` in [0, 1].
    """
    if style_image.ndim == 3:
        img = style_image[0]  # grayscale channel
    else:
        img = style_image
    H, W = img.shape

    ax.imshow(img, cmap="gray", extent=[0, W, H, 0], interpolation="bilinear")

    # Reshape attention to 2D grid.
    attn_2d = token_attn.reshape(grid_size, grid_size)
    cell = H / grid_size

    # Normalise for colour mapping (within this reference's attention).
    vmin = token_attn.min()
    vmax = token_attn.max()
    if vmax - vmin < 1e-8:
        vmin, vmax = 0, 1
    norm = Normalize(vmin=vmin, vmax=vmax)

    for r in range(grid_size):
        for c in range(grid_size):
            val = attn_2d[r, c]
            alpha = 0.1 + 0.7 * (val - vmin) / (vmax - vmin + 1e-8)
            color = plt.cm.hot(norm(val))
            rect = Rectangle(
                (c * cell, r * cell), cell, cell,
                linewidth=0.5, edgecolor=color, facecolor=color, alpha=alpha,
            )
            ax.add_patch(rect)

    ax.set_title(ref_label, fontsize=9)
    ax.axis("off")


def _draw_token_attention_bars(
    ax: plt.Axes,
    values: np.ndarray,
    labels: list[str],
    title: str,
) -> None:
    """Bar chart of attention values for a single content token."""
    n = len(values)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n))
    bars = ax.bar(range(n), values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Attention mass")
    ax.set_title(title, fontsize=9)

    total = float(values.sum())
    for bar, val in zip(bars, values):
        pct = float(val) / total * 100 if total > 0 else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=7,
        )


def _draw_pool_heatmaps(
    fig: plt.Figure,
    gs,
    pool_attn_np: np.ndarray,  # (n_tokens, n_ref*Q)
    style_images: np.ndarray,   # (n_ref, 3, H, W)
    style_codepoints: list[int],
    grid_size: int,
    n_ref: int,
    n_tokens: int,
) -> None:
    """Draw per-global-token heatmaps over reference glyphs.

    Each global token gets one row showing its attention on every style
    reference glyph.  This reveals *what* each token is summarizing.
    """
    Q = grid_size * grid_size
    # Show up to 6 tokens and up to 4 refs per row.
    max_tokens = min(n_tokens, 6)
    n_cols = n_ref
    n_rows = max_tokens

    gs_pool = fig.add_gridspec(
        n_rows, n_cols,
        left=gs[1, :].get_position(fig).x0,
        right=gs[1, :].get_position(fig).x1,
        bottom=gs[1, :].get_position(fig).y0,
        top=gs[1, :].get_position(fig).y1,
        hspace=0.45, wspace=0.25,
    )

    for token_i in range(max_tokens):
        for ref_i in range(n_ref):
            ax = fig.add_subplot(gs_pool[token_i, ref_i])
            if style_images[ref_i].ndim == 3:
                img = style_images[ref_i][0]
            else:
                img = style_images[ref_i]
            H, W_img = img.shape
            ax.imshow(img, cmap="gray", extent=[0, W_img, H, 0], interpolation="bilinear")

            token_attn = pool_attn_np[token_i, ref_i * Q : (ref_i + 1) * Q]
            attn_2d = token_attn.reshape(grid_size, grid_size)
            cell = H / grid_size

            vmin = token_attn.min()
            vmax = token_attn.max()
            if vmax - vmin < 1e-8:
                vmin, vmax = 0, 1
            norm = Normalize(vmin=vmin, vmax=vmax)
            for r in range(grid_size):
                for c in range(grid_size):
                    val = attn_2d[r, c]
                    alpha = 0.1 + 0.7 * (val - vmin) / (vmax - vmin + 1e-8)
                    color = plt.cm.hot(norm(val))
                    rect = Rectangle(
                        (c * cell, r * cell), cell, cell,
                        linewidth=0.5, edgecolor=color, facecolor=color, alpha=alpha,
                    )
                    ax.add_patch(rect)

            if token_i == 0:
                ax.set_title(f"U+{style_codepoints[ref_i]:04X}", fontsize=8)
            if ref_i == 0:
                ax.set_ylabel(f"Tok {token_i}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])


# ---------------------------------------------------------------------------
# Per-block, per-head decomposition plot
# ---------------------------------------------------------------------------

def build_decomposition_figure(
    captured: dict[int, torch.Tensor],
    grid_size: int,
    n_ref: int,
    content_token_idx: int,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """Build a figure showing per-block, per-head attention.

    When K equals ``n_ref * grid_size * grid_size`` the subplots show spatial
    attention over style glyphs (legacy per-position mode).  When K is small
    (e.g. 16), the subplots show attention to global style tokens as a bar
    chart per head.
    """
    n_blocks = len(captured)
    n_heads = captured[0].shape[1]  # (B, n_heads, Q, K)
    K = captured[0].shape[3]
    Q_per_ref = grid_size * grid_size
    is_pooled = K < n_ref * Q_per_ref

    if is_pooled:
        # Pooled tokens: show as a compact bar-chart grid.
        n_tokens = K
        fig, axes = plt.subplots(
            n_heads, n_blocks,
            figsize=(3 * n_blocks, 2 * n_heads),
            squeeze=False,
        )
        fig.suptitle(
            f"Per-head, per-block attention to {n_tokens} global style tokens "
            f"for content token {content_token_idx}",
            fontsize=12, fontweight="bold",
        )
        for block_i in range(n_blocks):
            attn_block = captured[block_i].squeeze(0)  # (n_heads, Q, K)
            for head_i in range(n_heads):
                ax = axes[head_i, block_i]
                values = attn_block[head_i, content_token_idx].numpy()  # (K,)
                colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_tokens))
                ax.bar(range(n_tokens), values, color=colors, width=0.8)
                ax.set_xticks(range(n_tokens))
                ax.set_xticklabels([str(i) for i in range(n_tokens)], fontsize=5)
                ax.tick_params(axis="y", labelsize=6)
                if head_i == 0:
                    ax.set_title(f"Block {block_i}", fontsize=9)
                if block_i == 0:
                    ax.set_ylabel(f"Head {head_i}", fontsize=8)
    else:
        fig, axes = plt.subplots(
            n_heads, n_blocks,
            figsize=(3 * n_blocks, 3 * n_heads),
            squeeze=False,
        )
        fig.suptitle(
            f"Per-head, per-block attention maps for content token {content_token_idx}\n"
            f"(averaged across {n_ref} style references)",
            fontsize=12, fontweight="bold",
        )
        for block_i in range(n_blocks):
            attn_block = captured[block_i].squeeze(0)  # (n_heads, Q, K_total)
            for head_i in range(n_heads):
                ax = axes[head_i, block_i]
                attn_head = attn_block[head_i, content_token_idx]  # (K_total,)
                attn_2d = np.zeros((grid_size, grid_size))
                for ref_i in range(n_ref):
                    ref_attn = attn_head[ref_i * Q_per_ref : (ref_i + 1) * Q_per_ref].numpy()
                    attn_2d += ref_attn.reshape(grid_size, grid_size)
                attn_2d /= n_ref

                im = ax.imshow(attn_2d, cmap="hot", aspect="equal", interpolation="nearest")
                ax.set_title(
                    f"Block {block_i}, Head {head_i}" if block_i == 0 and head_i == 0
                    else f"Head {head_i}" if block_i == 0
                    else f"Block {block_i}" if head_i == 0
                    else "",
                    fontsize=8,
                )
                ax.set_xticks([])
                ax.set_yticks([])

    plt.tight_layout()
    if output_path:
        stem = str(output_path)
        decomp_path = stem.replace(".png", "_decomp.png")
        fig.savefig(decomp_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved decomposition to {decomp_path}")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize FeatureFusionModule cross-attention weights"
    )
    p.add_argument("--font-path", type=Path, required=True, help="Path to .ttf/.otf font")
    p.add_argument("--target-char", type=str, required=True,
                   help="Single character to generate, e.g. 'A' or 'U+0041'")
    p.add_argument("--gtok-model-path", type=Path, required=True)
    p.add_argument("--ar-model-path", type=Path, required=True)
    p.add_argument("--dataset-path", type=Path, required=True,
                   help="Path to google-fonts checkout")
    p.add_argument("--style-glyph-count", type=int, default=8,
                   help="Number of style reference glyphs (ignored when --style-chars is provided)")
    p.add_argument("--style-chars", type=str, default=None,
                   help="Comma-separated style reference codepoints, e.g. 'U+0041,U+0042,A,B'. "
                        "Overrides the model config's style_codepoints and --style-glyph-count.")
    p.add_argument("--content-token", type=int, default=None,
                   help="Highlight a specific content token index (default: auto)")
    p.add_argument("--output", type=Path, default=None,
                   help="Output path for the PNG (default: auto-generated name)")
    p.add_argument("--show", action="store_true",
                   help="Show figure interactively instead of saving")
    p.add_argument("--save-decomp", action="store_true",
                   help="Also save per-head, per-block decomposition plot")
    p.add_argument("--zero-aggregator", action="store_true",
                   help="Zero out the FeatureFusionModule cross-attention output, "
                        "isolating the global style vector + codepoint embedding")
    return p.parse_args()


def _parse_target_char(s: str) -> int:
    if s.startswith("U+") or s.startswith("u+"):
        return int(s[2:], 16)
    return ord(s)


def _parse_chars(s: str) -> list[int]:
    """Parse a comma-separated string of codepoints like 'U+0041,A,U+00E9'."""
    result: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if part.upper().startswith("U+"):
            result.append(int(part[2:], 16))
        elif len(part) == 1:
            result.append(ord(part))
        else:
            raise ValueError(f"Cannot parse codepoint: {part!r}")
    return result


def _resolve_style_codepoints(
    *,
    font,
    target_char: int,
    style_glyph_count: int,
    cli_chars: Optional[list[int]],
    config_chars: Optional[list[int]],
) -> list[int]:
    """Resolve style reference codepoints from CLI, config, or font sampling.

    Precedence:
      1. ``cli_chars`` (--style-chars) — used verbatim, filtered to font availability.
      2. ``config_chars`` (ARModelConfig.style_codepoints) — model's training style set.
      3. Per-font random sampling from Latin Kernel (legacy fallback).
    """
    # ── Helper: filter a candidate list to what the font actually has ──
    def _filter(candidates: list[int]) -> list[int]:
        result: list[int] = []
        for cp in candidates:
            if cp == target_char:
                print(f"  Skipping target char U+{cp:04X} from style candidates")
                continue
            if not _font_has_codepoint(font, cp):
                print(f"  Font missing U+{cp:04X}, skipping")
                continue
            if not _has_non_empty_glyph(font, cp):
                print(f"  U+{cp:04X} has empty glyph, skipping")
                continue
            result.append(cp)
        return result

    # ── 1. CLI override ─────────────────────────────────────────────
    if cli_chars is not None:
        filtered = _filter(cli_chars)
        if not filtered:
            raise RuntimeError(
                "None of the --style-chars codepoints are available in the font"
            )
        print(f"Style references (from --style-chars, {len(filtered)}/{len(cli_chars)} available): "
              f"{[f'U+{cp:04X}' for cp in filtered]}")
        return filtered

    # ── 2. Model config's training style set ─────────────────────────
    if config_chars is not None:
        filtered = _filter(config_chars)
        if filtered:
            # Take up to style_glyph_count, preserving order.
            selected = filtered[:style_glyph_count]
            print(f"Style references (from model config style_codepoints, "
                  f"{len(selected)}/{len(config_chars)}): "
                  f"{[f'U+{cp:04X}' for cp in selected]}")
            return selected
        print("Model config style_codepoints are all missing from this font; "
              "falling back to Latin Kernel sampling.")

    # ── 3. Legacy fallback: sample from Latin Kernel ─────────────────
    from hrothgar.ar.style_sampling import _sample_style_codepoints

    common = sorted(set(LATIN_KERNEL) - {target_char})[:32]
    result = _sample_style_codepoints(
        font=font,
        target_char=target_char,
        style_glyph_count=style_glyph_count,
        common_style_codepoints=common,
    )
    print(f"Style references (sampled from Latin Kernel): "
          f"{[f'U+{cp:04X}' for cp in result]}")
    return result


def main() -> None:
    args = _parse_args()
    target_cp = _parse_target_char(args.target_char)

    # ── Validate paths ─────────────────────────────────────────────────
    for p, label in [
        (args.font_path, "font"),
        (args.gtok_model_path, "gtok model"),
        (args.ar_model_path, "ar model"),
        (args.dataset_path, "dataset"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    device = pick_device()
    print(f"Using device: {device}")

    # ── Load models ────────────────────────────────────────────────────
    ar_config = ARModelConfig.from_sidecar(args.ar_model_path)
    image_size = ar_config.image_size

    gtok, gtok_config = load_gtok_model(Path(args.gtok_model_path), device)

    ar_model = ARModel(ar_config, gtok_model=gtok).to(device)
    ar_model.load(str(args.ar_model_path), device=device)
    ar_model.gtok.load_state_dict(gtok.state_dict(), strict=False)
    ar_model.freeze_gtok()
    ar_model.eval()

    grid_size = ar_model.token_grid_height

    # ── Find font ──────────────────────────────────────────────────────
    font = find_google_font_by_basename(args.dataset_path, args.font_path)
    print(f"Matched font: {font.family}")

    # ── Render glyphs ──────────────────────────────────────────────────
    # Target (content) glyph.
    content_img = np.asarray(font.render(target_cp, size=image_size))
    if _is_blank_rendering(content_img):
        raise RuntimeError(f"Blank rendering for U+{target_cp:04X} in {font.family}")

    # ── Style reference glyphs ───────────────────────────────────────────
    cli_chars = _parse_chars(args.style_chars) if args.style_chars else None
    style_cps = _resolve_style_codepoints(
        font=font,
        target_char=target_cp,
        style_glyph_count=args.style_glyph_count,
        cli_chars=cli_chars,
        config_chars=ar_config.style_codepoints,
    )

    style_imgs = np.stack([_render_glyph_codepoint(font, cp, image_size) for cp in style_cps])
    # style_imgs: (n_ref, 3, H, W)

    # ── Prepare tensors ────────────────────────────────────────────────
    content_t = torch.from_numpy(content_img).unsqueeze(0).to(device)      # (1, 3, H, W)
    style_t = torch.from_numpy(style_imgs).unsqueeze(0).to(device)          # (1, n_ref, 3, H, W)
    latincore_idx = ar_model._unicode_to_latincore(
        torch.tensor([target_cp], device=device)
    )

    # ── Register attention hooks ───────────────────────────────────────
    capture = AttentionCapture(ar_model.aggregator)
    capture.register()

    try:
        # ── Run forward pass (conditioning only — no generation needed) ─
        with torch.no_grad():
            _ = ar_model.build_conditioning_map(
                content_images=content_t,
                style_reference_images=style_t,
                latincore_idx=latincore_idx,
                zero_aggregator=args.zero_aggregator,
            )
    finally:
        capture.remove()

    if not capture.captured_blocks:
        raise RuntimeError(
            "No attention weights captured. The forward pass may not have "
            "triggered the StyleAttentionBlock hooks."
        )

    print(
        f"Captured cross-attention from {len(capture.captured_blocks)} blocks × "
        f"{capture.captured_blocks[0].shape[1]} heads.\n"
        f"Attention matrix shape: {capture.captured_blocks[0].shape}  "
        f"(B, heads, Q={grid_size*grid_size}, K={capture.captured_blocks[0].shape[3]})"
    )
    if capture.uses_pooling:
        if capture.captured_pool is not None:
            print(
                f"Style pool attention: {capture.captured_pool.shape[1]} heads × "
                f"{capture.captured_pool.shape[2]} tokens → "
                f"{capture.captured_pool.shape[3]} style positions"
            )
        else:
            print("Style pool active but no pool attention captured.")

    # ── Determine output path ──────────────────────────────────────────
    if args.output:
        out_path = args.output
    else:
        out_stem = f"{args.font_path.stem}_U+{target_cp:04X}_crossattn"
        out_path = Path(f"{out_stem}.png")

    # ── Build figure ───────────────────────────────────────────────────
    glyph_gray = content_img[0]  # single-channel grayscale

    n_ref = len(style_cps)

    fig = build_figure(
        glyph_image=glyph_gray,
        style_images=style_imgs,
        style_codepoints=style_cps,
        captured_blocks=capture.captured_blocks,
        captured_pool=capture.captured_pool,
        grid_size=grid_size,
        n_ref=n_ref,
        content_token_idx=args.content_token,
        zero_aggregator=args.zero_aggregator,
        output_path=None if args.show else out_path,
    )

    if args.save_decomp:
        build_decomposition_figure(
            captured=capture.captured_blocks,
            grid_size=grid_size,
            n_ref=n_ref,
            content_token_idx=args.content_token or 0,
            output_path=out_path,
        )

    if args.show:
        plt.show()
    else:
        plt.close(fig)
        print("Done.")


if __name__ == "__main__":
    main()
