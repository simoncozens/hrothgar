"""Diffusion-based glyph generator for GAR-Font.

Architecture:
  1. Frozen G-Tok CNN encoder → structural content features.
  2. Upstream GAR-Font StyleEncoder → visual style features.
  3. FeatureFusionModule → cross-attention fusion of content + style.
  4. GlyphDiT (Diffusion Transformer) → denoises G-Tok codebook embeddings
     conditioned on timestep, codepoint identity, and fused style.
  5. Gumbel-softmax quantisation → frozen G-Tok decoder → images.

Training:
  1. Get clean embeddings x₀ from G-Tok encoder + quantiser.
  2. Sample t ~ Uniform(0, T-1), add noise: x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε.
  3. GlyphDiT predicts ε_θ(x_t, t, codepoint, style).
  4. Primary loss: MSE(ε_θ, ε).
  5. Auxiliary: predict x̂₀, Gumbel-softmax quantise, decode → image,
     compute L1 + LPIPS against ground truth.

Inference (DDIM):
  1. Sample x_T ~ N(0, I).
  2. DDIM loop (250 steps) → x₀.
  3. Nearest-codebook quantise → G-Tok decode → image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hrothgar.ar.dit import (
    DiTConfig,
    GlyphDiT,
    NoiseScheduler,
    ddim_sample,
    get_beta_schedule,
)
from hrothgar.dataset import LATIN_CORE
from hrothgar.gtok.model import (
    GtokConfig,
    GtokModel,
)
from hrothgar.upstream.feature_fusion_module import FeatureFusionModule
from hrothgar.upstream.style_encoder import StyleEncoder
from hrothgar.utils import SaveLoadModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class GlyphGenConfig:
    """Configuration for the DiT-based glyph generator."""

    image_size: int = 128
    encoder_feature_dim: int = 256

    # Style encoder.
    style_encoder_base_channels: int = 32

    # FeatureFusionModule.
    aggregator_num_layers: int = 3
    aggregator_num_heads: int = 8

    # DiT backbone (~141M params, close to DiT-B).
    dit_hidden_size: int = 832
    dit_depth: int = 16
    dit_num_heads: int = 16
    dit_mlp_ratio: float = 4.0

    # Diffusion schedule.
    diffusion_steps: int = 1000
    noise_schedule: str = "squaredcos_cap_v2"
    ddim_steps: int = 250

    # Classifier-free guidance scale (> 1.0 amplifies style conditioning).
    cfg_scale: float = 1.0

    # Conditioning dropout for CFG.
    style_dropout_prob: float = 0.1

    # Gumbel-softmax temperature for auxiliary image loss.
    gumbel_temperature: float = 0.5

    # Regularisation on the style encoding path.
    style_dropout: float = 0.3

    freeze_gtok: bool = True

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.encoder_feature_dim % self.aggregator_num_heads != 0:
            raise ValueError(
                "encoder_feature_dim must be divisible by aggregator_num_heads "
                f"(got {self.encoder_feature_dim} and {self.aggregator_num_heads})"
            )
        if self.dit_hidden_size % self.dit_num_heads != 0:
            raise ValueError(
                "dit_hidden_size must be divisible by dit_num_heads "
                f"(got {self.dit_hidden_size} and {self.dit_num_heads})"
            )

    def dit_config(self) -> DiTConfig:
        """Build DiTConfig from this config + derived sizes."""
        return DiTConfig(
            hidden_size=self.dit_hidden_size,
            depth=self.dit_depth,
            num_heads=self.dit_num_heads,
            mlp_ratio=self.dit_mlp_ratio,
            num_diffusion_steps=self.diffusion_steps,
            noise_schedule=self.noise_schedule,
            ddim_steps=self.ddim_steps,
            codepoint_embedding_dim=self.encoder_feature_dim,
            style_feature_dim=self.encoder_feature_dim,
            style_dropout_prob=self.style_dropout_prob,
        )


# ---------------------------------------------------------------------------
# Model output
# ---------------------------------------------------------------------------


@dataclass
class GlyphGenOutput:
    """Outputs returned by ``GlyphGenerator.forward`` and ``generate``."""

    # Predicted noise ε_θ (for MSE loss during training).
    noise_pred: Optional[torch.Tensor] = None
    # Added noise (for MSE target during training).
    noise_target: Optional[torch.Tensor] = None
    # Denoised embeddings x̂₀ (for auxiliary loss).
    denoised_embeddings: Optional[torch.Tensor] = None
    # Soft-decoded reconstruction from x̂₀.
    reconstructed_images: Optional[torch.Tensor] = None
    # Gumbel-softmax hard reconstruction (for LPIPS).
    perceptual_recon: Optional[torch.Tensor] = None
    # Final quantised token indices (generation only).
    token_indices: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class GlyphGenerator(SaveLoadModel):
    """Diffusion Transformer glyph generator.

    Conditioning pipeline:
      content image → G-Tok CNN encoder → content features  (frozen)
      style images → StyleEncoder → style features
      content + style → FeatureFusionModule → fused style features
      codepoint → LATIN_CORE index → embedding → codepoint features

    Generation:
      GlyphDiT denoises random Gaussian noise into G-Tok codebook
      embeddings, which are quantised and decoded to images.
    """

    def __init__(
        self,
        config: GlyphGenConfig,
        gtok_model: Optional[GtokModel] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.gtok = gtok_model or GtokModel(GtokConfig(image_size=config.image_size))

        if self.gtok.config.image_size != config.image_size:
            raise ValueError(
                f"GlyphGenerator and G-Tok image sizes must match "
                f"(got {config.image_size} and {self.gtok.config.image_size})"
            )

        # Derived dimensions from G-Tok.
        self.sequence_length = self.gtok.sequence_length  # 64
        self.token_grid_height = self.gtok.token_grid_height  # 8
        self.token_grid_width = self.gtok.token_grid_width  # 8
        self.codebook_size = self.gtok.config.quantizer_codebook_size  # 8192
        self.codebook_dim = self.gtok.config.quantizer_code_dim  # 16

        # Content encoder (reuses frozen G-Tok CNN).
        self.content_encoder = self.gtok.cnn_encoder

        # Style encoder.
        style_downsample_ratio = config.image_size // self.token_grid_height
        if style_downsample_ratio not in (8, 16):
            raise ValueError(
                f"StyleEncoder downsample ratio must be 8 or 16, got "
                f"{style_downsample_ratio}"
            )
        self.style_encoder = StyleEncoder(
            C_in=3,
            C=config.style_encoder_base_channels,
            C_out=config.encoder_feature_dim,
            norm="in",
            activ="relu",
            pad_type="reflect",
            sigmoid=False,
            scale_var=True,
            downsample_ratio=style_downsample_ratio,
        )
        self.style_dropout_2d = nn.Dropout2d(config.style_dropout)

        # Codepoint embedding: Unicode → LATIN_CORE index → dense vector.
        self._register_latincore_mapping()
        self.codepoint_embedding = nn.Embedding(
            len(LATIN_CORE), config.encoder_feature_dim
        )

        # Content-style aggregator.
        self.aggregator = FeatureFusionModule(
            z_channel=config.encoder_feature_dim,
            n_heads=config.aggregator_num_heads,
            n_style_blocks=config.aggregator_num_layers,
        )

        # DiT backbone.
        dit_config = config.dit_config()
        self.dit = GlyphDiT(dit_config)

        # Noise scheduler.
        betas = get_beta_schedule(config.noise_schedule, config.diffusion_steps)
        self.scheduler = NoiseScheduler(betas)

        self._gtok_frozen: bool = False
        if config.freeze_gtok:
            self.freeze_gtok()

    # ------------------------------------------------------------------
    # G-Tok management
    # ------------------------------------------------------------------

    def freeze_gtok(self) -> None:
        """Freeze G-Tok so only the DiT and conditioning modules train."""
        self.gtok.eval()
        for parameter in self.gtok.parameters():
            parameter.requires_grad = False
        self._gtok_frozen = True

    def train(self, mode: bool = True) -> "GlyphGenerator":
        """Set training mode while keeping frozen G-Tok in eval mode."""
        super().train(mode)
        if self._gtok_frozen:
            self.gtok.eval()
        return self

    # ------------------------------------------------------------------
    # LATIN_CORE mapping
    # ------------------------------------------------------------------

    def _register_latincore_mapping(self) -> None:
        max_cp = max(LATIN_CORE)
        mapping = torch.full((max_cp + 1,), -1, dtype=torch.long)
        for idx, cp in enumerate(LATIN_CORE):
            mapping[cp] = idx
        self.register_buffer("_latincore_map", mapping, persistent=False)

    def _unicode_to_latincore(self, codepoints: torch.Tensor) -> torch.Tensor:
        max_cp = self._latincore_map.shape[0] - 1  # type: ignore[attr-defined]
        clamped = torch.clamp(codepoints, max=max_cp)
        indices = self._latincore_map[clamped]  # type: ignore[attr-defined]
        oob = indices < 0
        if oob.any():
            print(
                f"Warning: {oob.sum().item()} codepoint(s) not in LATIN_CORE; "
                f"{codepoints[oob].tolist()} mapping to index 0"
            )
            indices = torch.where(oob, torch.zeros_like(indices), indices)
        return indices

    # ------------------------------------------------------------------
    # Encoding and conditioning
    # ------------------------------------------------------------------

    def encode_content(self, content_images: torch.Tensor) -> torch.Tensor:
        """Encode content glyphs → (B, C, H, W) feature map."""
        features = self.content_encoder(content_images)
        _batch, _channels, height, width = features.shape
        if height != self.token_grid_height or width != self.token_grid_width:
            raise ValueError(
                f"Content encoder output shape mismatch: "
                f"got {(height, width)}, expected "
                f"{(self.token_grid_height, self.token_grid_width)}"
            )
        return features

    def encode_style(self, style_reference_images: torch.Tensor) -> torch.Tensor:
        """Encode style references → (B, n_ref, C, H, W)."""
        batch_size, num_refs, channels, height, width = style_reference_images.shape
        if channels != 3:
            raise ValueError(f"Expected RGB style references, got {channels} channels")
        flattened = style_reference_images.reshape(
            batch_size * num_refs, channels, height, width
        )
        # Normalise [0, 1] → [-1, 1] for style encoder.
        flattened = (flattened - 0.5) / 0.5
        encoded = self.style_encoder(flattened)
        encoded = self.style_dropout_2d(encoded)
        _, fc, fh, fw = encoded.shape
        if fh != self.token_grid_height or fw != self.token_grid_width:
            raise ValueError(
                f"Style encoder output shape mismatch: "
                f"got {(fh, fw)}, expected "
                f"{(self.token_grid_height, self.token_grid_width)}"
            )
        return encoded.reshape(batch_size, num_refs, fc, fh, fw)  # (B, n_ref, C, H, W)

    def build_style_features(
        self,
        content_images: torch.Tensor,
        style_reference_images: torch.Tensor,
    ) -> torch.Tensor:
        """Build a single style feature vector per batch item.

        Content and style features are fused via the FeatureFusionModule
        cross-attention, then spatially averaged into a single
        ``(B, encoder_feature_dim)`` vector suitable for DiT conditioning.

        Both content and style encoder outputs share the same spatial
        resolution (``token_grid_height × token_grid_width``), matching
        the G-Tok token grid.

        Args:
            content_images: ``(B, 3, H, W)`` content glyphs.
            style_reference_images: ``(B, n_ref, 3, H, W)`` style refs.

        Returns:
            ``(B, encoder_feature_dim)`` style feature vector.
        """
        content_features = self.encode_content(content_images)  # (B, C, H, W)
        style_features = self.encode_style(
            style_reference_images
        )  # (B, n_ref, C, H, W)

        # Fuse via cross-attention (preserves 2D spatial structure).
        fused = self.aggregator(content_features, style_features)  # (B, C, H, W)

        # Pool spatially to a single vector per batch item.
        return fused.mean(dim=[2, 3])  # (B, C)

    def get_codepoint_embedding(self, codepoints: torch.Tensor) -> torch.Tensor:
        """Get codepoint embedding for target glyphs.

        Args:
            codepoints: ``(B,)`` integer tensor of Unicode codepoints.

        Returns:
            ``(B, encoder_feature_dim)`` codepoint embedding.
        """
        latincore_idx = self._unicode_to_latincore(codepoints)
        return self.codepoint_embedding(latincore_idx)

    # ------------------------------------------------------------------
    # G-Tok utilities
    # ------------------------------------------------------------------

    def codebook_embeddings(self) -> torch.Tensor:
        """Return the G-Tok codebook matrix ``(K, D)``."""
        codebook = self.gtok.quantizer.embedding.weight
        if self.gtok.quantizer.l2_norm:
            codebook = F.normalize(codebook, p=2, dim=-1)
        return codebook

    def target_embeddings_from_images(
        self, target_images: torch.Tensor
    ) -> torch.Tensor:
        """Get clean codebook embeddings for target images via G-Tok.

        Returns:
            ``(B, N, D)`` quantised embeddings.
        """
        with torch.no_grad():
            quantized, _loss_info = self.gtok.encode(target_images)
        return quantized

    def target_token_indices_from_images(
        self, target_images: torch.Tensor
    ) -> torch.Tensor:
        """Get quantised token indices for target images via G-Tok.

        Returns:
            ``(B, N)`` long tensor of codebook indices.
        """
        with torch.no_grad():
            quantized, _loss_info = self.gtok.encode(target_images)
        # Nearest-codebook lookup.
        codebook = self.codebook_embeddings()  # (K, D)
        dists = (
            (quantized.unsqueeze(-2) - codebook.unsqueeze(0).unsqueeze(0))
            .pow(2)
            .sum(dim=-1)
        )  # (B, N, K)
        return dists.argmin(dim=-1)  # (B, N)

    def soft_decode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Decode continuous embeddings through G-Tok decoder.

        The embeddings are quantised to the nearest codebook entry
        via straight-through, then decoded to images.

        Args:
            embeddings: ``(B, N, D)`` continuous embeddings.

        Returns:
            ``(B, 3, H, W)`` reconstructed images.
        """
        codebook = self.codebook_embeddings()  # (K, D)
        # Nearest-neighbour lookup.
        dists = (
            (embeddings.unsqueeze(-2) - codebook.unsqueeze(0).unsqueeze(0))
            .pow(2)
            .sum(dim=-1)
        )  # (B, N, K)
        indices = dists.argmin(dim=-1)  # (B, N)
        quantised = codebook[indices]  # (B, N, D)
        return self.gtok.decode(quantised)

    def gumbel_softmax_decode(
        self,
        embeddings: torch.Tensor,
        temperature: float = 0.5,
    ) -> torch.Tensor:
        """Decode embeddings via Gumbel-softmax straight-through quantisation.

        The forward pass uses hard one-hot codes (sharp images), while the
        backward pass propagates gradients through the softmax, teaching
        the DiT to produce embeddings close to valid codebook entries.

        Args:
            embeddings: ``(B, N, D)`` continuous embeddings.
            temperature: Gumbel-softmax temperature (lower = harder).

        Returns:
            ``(B, 3, H, W)`` reconstructed images with gradient path.
        """
        codebook = self.codebook_embeddings()  # (K, D)
        # Negative squared distance as logits.
        logits = -(
            (embeddings.unsqueeze(-2) - codebook.unsqueeze(0).unsqueeze(0))
            .pow(2)
            .sum(dim=-1)
        )  # (B, N, K)
        hard_one_hot = F.gumbel_softmax(
            logits, tau=temperature, hard=True, dim=-1
        )  # (B, N, K)
        hard_embeddings = torch.matmul(hard_one_hot, codebook)  # (B, N, D)
        return self.gtok.decode(hard_embeddings)

    # ------------------------------------------------------------------
    # Training and generation
    # ------------------------------------------------------------------

    def forward(
        self,
        content_images: torch.Tensor,
        style_reference_images: torch.Tensor,
        *,
        target_images: torch.Tensor,
        target_codepoints: torch.Tensor,
    ) -> GlyphGenOutput:
        """Training forward pass: add noise, predict ε, decode for aux loss.

        Args:
            content_images: ``(B, 3, H, W)`` content glyphs.
            style_reference_images: ``(B, n_ref, 3, H, W)`` style refs.
            target_images: ``(B, 3, H, W)`` ground-truth glyphs.
            target_codepoints: ``(B,)`` integer Unicode codepoints.

        Returns:
            ``GlyphGenOutput`` with noise_pred, noise_target,
            denoised_embeddings, and reconstructed images.
        """
        B = target_images.shape[0]
        device = target_images.device
        N, D = self.sequence_length, self.codebook_dim

        # Conditioning.
        style_features = self.build_style_features(
            content_images, style_reference_images
        )  # (B, C)
        codepoint_emb = self.get_codepoint_embedding(target_codepoints)  # (B, C)

        # Clean embeddings from G-Tok.
        x_0 = self.target_embeddings_from_images(target_images)  # (B, N, D)

        # Sample timesteps and noise.
        t = torch.randint(0, self.scheduler.num_steps, (B,), device=device)
        noise = torch.randn_like(x_0)

        # Forward diffusion: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε.
        x_t = self.scheduler.q_sample(x_0, t, noise=noise)

        # Predict noise.
        noise_pred = self.dit(
            x_t=x_t,
            t=t,
            codepoint_emb=codepoint_emb,
            style_features=style_features,
        )

        # Predict x_0 for auxiliary image loss.
        x0_pred = self.scheduler.predict_x0_from_eps(x_t, t, noise_pred)

        # Soft decode for L1.
        reconstructed = self.soft_decode(x0_pred)

        # Gumbel-softmax decode for LPIPS.
        perceptual_recon = self.gumbel_softmax_decode(
            x0_pred, temperature=self.config.gumbel_temperature
        )

        return GlyphGenOutput(
            noise_pred=noise_pred,
            noise_target=noise,
            denoised_embeddings=x0_pred,
            reconstructed_images=reconstructed,
            perceptual_recon=perceptual_recon,
        )

    @torch.no_grad()
    def generate(
        self,
        content_images: torch.Tensor,
        style_reference_images: torch.Tensor,
        target_codepoints: torch.Tensor,
    ) -> GlyphGenOutput:
        """DDIM sampling → quantise → decode → glyph image.

        Args:
            content_images: ``(B, 3, H, W)`` content glyphs.
            style_reference_images: ``(B, n_ref, 3, H, W)`` style refs.
            target_codepoints: ``(B,)`` integer Unicode codepoints.

        Returns:
            ``GlyphGenOutput`` with reconstructed_images and token_indices.
        """
        B = content_images.shape[0]
        device = content_images.device
        N, D = self.sequence_length, self.codebook_dim

        # Conditioning.
        style_features = self.build_style_features(
            content_images, style_reference_images
        )
        codepoint_emb = self.get_codepoint_embedding(target_codepoints)

        # DDIM sampling.
        x0 = ddim_sample(
            model=self.dit,
            scheduler=self.scheduler,
            shape=(B, N, D),
            codepoint_emb=codepoint_emb,
            style_features=style_features,
            ddim_steps=self.config.ddim_steps,
            cfg_scale=self.config.cfg_scale,
            device=device,
        )

        # Quantise to nearest codebook entry.
        codebook = self.codebook_embeddings()
        dists = (
            (x0.unsqueeze(-2) - codebook.unsqueeze(0).unsqueeze(0)).pow(2).sum(dim=-1)
        )
        token_indices = dists.argmin(dim=-1)  # (B, N)

        # Decode.
        quantised = codebook[token_indices]  # (B, N, D)
        reconstructed = self.gtok.decode(quantised)

        return GlyphGenOutput(
            reconstructed_images=reconstructed,
            token_indices=token_indices,
        )

    # ------------------------------------------------------------------
    # Checkpoint / utility
    # ------------------------------------------------------------------

    def parameter_counts(self) -> Dict[str, int]:
        """Return parameter counts for key components."""
        return {
            "content_encoder": sum(
                p.numel() for p in self.content_encoder.parameters()
            ),
            "style_encoder": sum(p.numel() for p in self.style_encoder.parameters()),
            "aggregator": sum(p.numel() for p in self.aggregator.parameters()),
            "codepoint_embedding": sum(
                p.numel() for p in self.codepoint_embedding.parameters()
            ),
            "dit": sum(p.numel() for p in self.dit.parameters()),
            "total_trainable": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Return parameters that currently require gradients."""
        return [p for p in self.parameters() if p.requires_grad]
