"""MaskGIT-based glyph generator for GAR-Font.

This module implements the vision-only stage of the GAR-Font generator:

1. A frozen G-Tok CNN encoder extracts structural content features.
2. The upstream GAR-Font StyleEncoder extracts visual style features.
3. A content-style aggregator (FeatureFusionModule) fuses both streams.
4. A bidirectional MaskGIT transformer predicts G-Tok codebook indices
   via masked token prediction (like BERT).
5. A soft codebook projection feeds the frozen G-Tok decoder to reconstruct
   images.

Metric conditioning (concern 3):
- ``MetricEmbedder`` injects font vertical metrics + per-glyph advance width
  into the conditioning map to improve baseline/x-height/width alignment.
- ``GlyphWidthHead`` predicts advance width from token embeddings as an
  auxiliary training signal.

NFA/GA adaptation:
- ``enable_nfa_mode()`` freezes the model and injects LoRA adapters into the
  MaskGIT transformer decoder, enabling lightweight per-font fine-tuning.
- ``enable_composed_nfa_mode()`` stacks a frozen GA glyph prior with a
  trainable NFA font adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hrothgar.ar.maskgit import (
    MaskGITConfig,
    MaskGITDecoder,
    MaskGITTransformer,
)
from hrothgar.ar.config import ARModelConfig
from hrothgar.dataset import LATIN_CORE
from hrothgar.gtok.model import (
    GtokConfig,
    GtokModel,
)
from hrothgar.upstream.feature_fusion_module import FeatureFusionModule
from hrothgar.upstream.gpt import GPTModelArgs
from hrothgar.upstream.style_encoder import StyleEncoder
from hrothgar.utils import SaveLoadModel

# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class ARModelOutput:
    """Outputs returned by ``ARModel.forward``."""

    logits: torch.Tensor
    reconstructed_images: torch.Tensor
    soft_token_embeddings: torch.Tensor
    target_token_indices: Optional[torch.Tensor]
    token_mask: Optional[torch.Tensor] = None
    predicted_width: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Metric conditioning modules
# ---------------------------------------------------------------------------


class MetricEmbedder(nn.Module):
    """Embeds font metrics into conditioning dimension.

    Input: ``(B, 6)`` tensor of normalised metrics:
    ``[ascender, descender, x_height, cap_height, baseline, advance_width]``

    Output: ``(B, encoder_feature_dim)`` vector added to the codepoint
    embedding before spatial expansion.
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # Zero-init the final projection so the metric signal ramps in gently.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, metrics: torch.Tensor) -> torch.Tensor:
        return self.net(metrics)


class GlyphWidthHead(nn.Module):
    """Predicts glyph advance width from pooled token embeddings.

    Input: ``(B, codebook_dim)`` mean-pooled soft token embeddings.
    Output: ``(B,)`` scalar width prediction (normalised, in [0, 1]).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pooled_tokens: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_tokens).squeeze(-1)


# ---------------------------------------------------------------------------
# ARModel
# ---------------------------------------------------------------------------


class ARModel(SaveLoadModel):
    """MaskGIT glyph generator with metric conditioning and LoRA adaptation."""

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

        style_downsample_ratio = config.image_size // self.token_grid_height
        if style_downsample_ratio not in (8, 16):
            raise ValueError(
                f"StyleEncoder downsample ratio must be 8 or 16, but "
                f"image_size={config.image_size} / token_grid_height="
                f"{self.token_grid_height} = {style_downsample_ratio}"
            )

        self.content_encoder = self.gtok.cnn_encoder
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
        self.style_dropout = nn.Dropout2d(config.style_dropout)

        self._register_latincore_mapping()
        latincore_size = len(LATIN_CORE)
        self.codepoint_embedding = nn.Embedding(
            latincore_size, config.encoder_feature_dim
        )

        self.aggregator = FeatureFusionModule(
            z_channel=config.encoder_feature_dim,
            n_heads=config.aggregator_num_heads,
            n_style_blocks=config.aggregator_num_layers,
            n_style_tokens=config.style_pool_tokens,
        )

        conditioning_dim = config.encoder_feature_dim * 2
        gpt_config = GPTModelArgs(
            vocab_size=self.codebook_size,
            dim=config.decoder_hidden_dim,
            n_layer=config.decoder_num_layers,
            n_head=config.decoder_num_heads,
            img_feature_channel=conditioning_dim,
            img_feature_code_len=self.sequence_length,
            target_token_len=self.sequence_length,
            token_dropout_p=config.decoder_dropout,
            attn_dropout_p=config.decoder_attention_dropout,
            resid_dropout_p=config.decoder_dropout,
            ffn_dropout_p=config.decoder_dropout,
        )

        maskgit_transformer = MaskGITTransformer(gpt_config)
        maskgit_config = MaskGITConfig(
            num_inference_steps=config.maskgit_num_inference_steps,
            temperature=config.maskgit_temperature,
        )
        self.maskgit_decoder = MaskGITDecoder(maskgit_transformer, maskgit_config)

        self.language_adapter: Optional[nn.Module] = None
        if language_adapter is not None:
            self.set_language_adapter(language_adapter)

        # Global style vector: pools frozen G-Tok ViT features into a
        # single vector injected uniformly across all conditioning positions.
        self.global_style_projection: Optional[nn.Module] = None
        if config.use_global_style:
            self.global_style_projection = nn.Sequential(
                nn.Linear(
                    self.gtok.config.vit_hidden_dim, config.encoder_feature_dim
                ),
                nn.SiLU(),
                nn.Linear(
                    config.encoder_feature_dim, config.encoder_feature_dim
                ),
            )
            # Zero-init the final projection so the global signal ramps in gently.
            nn.init.zeros_(self.global_style_projection[-1].weight)
            nn.init.zeros_(self.global_style_projection[-1].bias)

        # Metric conditioning (concern 3).
        self.metric_embedder: Optional[MetricEmbedder] = None
        self.width_head: Optional[GlyphWidthHead] = None
        if config.use_metrics:
            self.metric_embedder = MetricEmbedder(
                input_dim=6,
                hidden_dim=config.metric_embedding_hidden_dim,
                output_dim=config.encoder_feature_dim,
            )
            self.width_head = GlyphWidthHead(
                input_dim=self.codebook_dim,
                hidden_dim=config.width_head_hidden_dim,
            )

        self._gtok_frozen: bool = False
        if config.freeze_gtok:
            self.freeze_gtok()

        self._global_step: int = 0
        self._content_only_step: bool = False
        self._style_only_step: bool = False
        self._nfa_mode: bool = False

    def _register_latincore_mapping(self) -> None:
        """Build a buffer that maps Unicode codepoints -> LATIN_CORE indices.

        Creates a tensor of shape ``(max_unicode + 1,)`` where each entry is
        either the LATIN_CORE index or -1 for codepoints not in LATIN_CORE.
        """
        max_cp = max(LATIN_CORE)
        mapping = torch.full((max_cp + 1,), -1, dtype=torch.long)
        for idx, cp in enumerate(LATIN_CORE):
            mapping[cp] = idx
        self.register_buffer("_latincore_map", mapping, persistent=False)

    def _unicode_to_latincore(self, codepoints: torch.Tensor) -> torch.Tensor:
        """Convert Unicode codepoint tensor to LATIN_CORE indices.

        Args:
            codepoints: ``(B,)`` LongTensor of Unicode codepoints.

        Returns:
            ``(B,)`` LongTensor of LATIN_CORE indices.  Codepoints not in
            LATIN_CORE are mapped to 0 (first entry) with a warning.
        """
        max_cp = self._latincore_map.shape[0] - 1  # type: ignore[attr-defined]
        clamped = torch.clamp(codepoints, max=max_cp)
        indices = self._latincore_map[clamped]  # type: ignore[attr-defined]
        oob = indices < 0
        if oob.any():
            print(
                f"Warning: {oob.sum().item()} codepoint(s) not in LATIN_CORE; "
                f"{codepoints[oob].tolist()}"
                f"mapping to index 0"
            )
            indices = torch.where(oob, torch.zeros_like(indices), indices)
        return indices

    def freeze_gtok(self) -> None:
        """Freeze G-Tok so the AR stage trains only its own modules."""
        self.gtok.eval()
        for parameter in self.gtok.parameters():
            parameter.requires_grad = False
        self._gtok_frozen = True

    def train(self, mode: bool = True) -> "ARModel":
        """Set training mode while keeping frozen G-Tok in eval mode.

        PyTorch recursively toggles all submodules when ``train()`` is called.
        AR training loops call ``model.train()`` every epoch, so without this
        guard a frozen tokenizer can be switched back to train mode, causing
        stochastic token targets. When G-Tok is frozen, force it back to eval.
        """
        super().train(mode)
        if self._gtok_frozen:
            self.gtok.eval()
        return self

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

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Return a list of parameters that currently require gradients.

        In NFA mode this is just the injected LoRA matrices.  In normal
        pretraining mode it is all non-GTok parameters.
        """
        return [p for p in self.parameters() if p.requires_grad]

    # ------------------------------------------------------------------
    # LoRA adaptation (NFA / GA)
    # ------------------------------------------------------------------

    def enable_nfa_mode(self, lora_config) -> None:
        """Switch to Novel Font Adaptation mode.

        Freezes all base parameters, then injects LoRA into the MaskGIT
        transformer.  After this call only the LoRA parameters are trainable;
        the optimizer should be constructed from ``trainable_parameters()``.

        May only be called once per model instance.

        Raises:
            RuntimeError: If NFA mode has already been enabled.
        """
        if self._nfa_mode:
            raise RuntimeError(
                "Model is already in NFA mode.  "
                "Create a fresh model instance to re-apply NFA."
            )
        # Freeze everything including G-Tok, encoders, aggregator, metric modules.
        for param in self.parameters():
            param.requires_grad = False
        # Inject LoRA — this creates new trainable parameters.
        self.maskgit_decoder.transformer.inject_lora(lora_config)
        # Ensure G-Tok stays in eval mode.
        self.freeze_gtok()
        self._nfa_mode = True

    def enable_composed_nfa_mode(
        self,
        glyph_lora_state: Dict[str, torch.Tensor],
        lora_config,
    ) -> None:
        """Switch to composed NFA mode with a frozen glyph prior.

        Combines a frozen glyph-specialist LoRA state dict (from GA training)
        with a trainable font-specific LoRA adapter.

        Args:
            glyph_lora_state: LoRA state dict from a prior GA run (produced
                by ``MaskGITTransformer.get_lora_state_dict()``).
            lora_config: LoRAConfig with matching rank and alpha.

        Raises:
            RuntimeError: If NFA mode has already been enabled.
        """
        if self._nfa_mode:
            raise RuntimeError(
                "Model is already in NFA mode.  "
                "Create a fresh model instance to re-apply NFA."
            )
        for param in self.parameters():
            param.requires_grad = False
        self.maskgit_decoder.transformer.inject_composed_lora(
            glyph_lora_state, lora_config
        )
        self.freeze_gtok()
        self._nfa_mode = True

    @property
    def is_nfa_mode(self) -> bool:
        """True once ``enable_nfa_mode`` or ``enable_composed_nfa_mode``
        has been called."""
        return self._nfa_mode

    # ------------------------------------------------------------------
    # Encoding and conditioning (PrefixLM: 2D feature maps preserved)
    # ------------------------------------------------------------------

    def encode_content(self, content_images: torch.Tensor) -> torch.Tensor:
        """Encode content glyphs to a 2D feature map ``(B, C, H, W)``."""
        content_features = self.content_encoder(content_images)
        _batch, _channels, height, width = content_features.shape
        if height != self.token_grid_height or width != self.token_grid_width:
            raise ValueError(
                "Content encoder output shape does not match G-Tok token grid "
                f"(got {(height, width)}, expected "
                f"{(self.token_grid_height, self.token_grid_width)})"
            )
        return content_features  # (B, C, H, W) -- 2D preserved

    def encode_style(self, style_reference_images: torch.Tensor) -> torch.Tensor:
        """Encode style references to a 2D feature map ``(B, n_ref, C, H, W)``.

        Style images are normalised from ``[0, 1]`` to ``[-1, 1]`` before the
        style encoder because the upstream GPT architecture (RoPE, RMSNorm,
        SwiGLU) was designed and tuned for zero-centred image features.
        The G-Tok content path stays at ``[0, 1]`` -- its encoder was
        pretrained on that range and is frozen during AR training.
        """
        batch_size, num_references, channels, height, width = (
            style_reference_images.shape
        )
        if channels != 3:
            raise ValueError(f"Expected RGB style references, got {channels} channels")
        flattened = style_reference_images.reshape(
            batch_size * num_references, channels, height, width
        )
        # Normalise from [0, 1] to [-1, 1].
        flattened = (flattened - 0.5) / 0.5
        encoded = self.style_encoder(flattened)
        encoded = self.style_dropout(encoded)
        _, feature_channels, feature_height, feature_width = encoded.shape
        if (
            feature_height != self.token_grid_height
            or feature_width != self.token_grid_width
        ):
            raise ValueError(
                "Style encoder output shape does not match G-Tok token grid "
                f"(got {(feature_height, feature_width)}, "
                f"expected {(self.token_grid_height, self.token_grid_width)})"
            )
        return encoded.reshape(
            batch_size, num_references, feature_channels, feature_height, feature_width
        )  # (B, n_ref, C, H, W) -- 2D preserved

    def encode_style_global(self, style_reference_images: torch.Tensor) -> torch.Tensor:
        """Extract a single global style vector from frozen G-Tok ViT features.

        Runs style references through G-Tok's CNN + ViT encoder, mean-pools
        across all spatial positions and reference images, and projects to
        ``encoder_feature_dim``.  Because G-Tok's ViT uses full self-attention,
        every position already carries global context; mean-pooling distills
        this into a compact style summary.

        Args:
            style_reference_images: ``(B, n_ref, 3, H, W)`` style glyphs in
                [0, 1] (same range G-Tok expects).

        Returns:
            ``(B, encoder_feature_dim)`` global style vector.
        """
        assert self.global_style_projection is not None, (
            "encode_style_global requires use_global_style=True in config"
        )
        batch_size, num_references, channels, height, width = (
            style_reference_images.shape
        )
        # Flatten references into batch dimension.
        flat = style_reference_images.reshape(
            batch_size * num_references, channels, height, width
        )
        # G-Tok expects [0, 1] input — same range as the content path.
        with torch.no_grad():
            cnn_out = self.gtok.cnn_encoder(flat)                     # (B*n_ref, cnn_latent, H', W')
            tokens = (
                self.gtok.proj_patch(cnn_out)                         # (B*n_ref, vit_hidden, H', W')
                .flatten(2)
                .transpose(1, 2)
            )                                                          # (B*n_ref, N, vit_hidden)
            vit_out = self.gtok.vit_encoder(tokens)                    # (B*n_ref, N, vit_hidden)
            vit_features = vit_out.mean(dim=1)                         # (B*n_ref, vit_hidden)

        # Mean-pool across reference images for each batch item.
        vit_features = vit_features.reshape(
            batch_size, num_references, self.gtok.config.vit_hidden_dim
        ).mean(dim=1)                                                   # (B, vit_hidden)

        global_style = self.global_style_projection(vit_features)      # (B, encoder_feature_dim)
        return global_style

    def build_conditioning_map(
        self,
        content_images: torch.Tensor,
        style_reference_images: torch.Tensor,
        latincore_idx: torch.Tensor,
        *,
        metrics: Optional[torch.Tensor] = None,
        zero_aggregator: bool = False,
    ) -> torch.Tensor:
        """Build the 2D conditioning feature map for PrefixLM.

        Returns ``(B, 2*encoder_feature_dim, H, W)`` -- codepoint identity
        embedding (optionally enriched with font metrics) projected to the
        feature dimension, and style-fused features concatenated along the
        channel axis.

        The codepoint embedding provides explicit, font-agnostic character
        identity.  When *metrics* is provided and ``use_metrics`` is enabled,
        the metric embedding is added to the codepoint embedding before
        spatial expansion, giving the model explicit geometric anchors.

        Args:
            content_images: ``(B, 3, H, W)`` content glyph renderings.
            style_reference_images: ``(B, n_ref, 3, H, W)`` style references.
            latincore_idx: ``(B,)`` LATIN_CORE indices.
            metrics: Optional ``(B, 6)`` tensor of normalised metrics
                ``[ascender, descender, x_height, cap_height, baseline,
                advance_width]``.
            zero_aggregator: If True, zero out the per-position cross-attention
                output from the FeatureFusionModule, leaving only the global
                style vector (if enabled) and codepoint embedding.
        """
        content_features = self.encode_content(content_images)  # (B, C, H, W)
        style_features = self.encode_style(
            style_reference_images
        )  # (B, n_ref, C, H, W)
        fused = self.aggregator(content_features, style_features)  # (B, C, H, W)

        if zero_aggregator:
            fused = torch.zeros_like(fused)

        # Inject global style vector: identical signal broadcast to every
        # spatial position, giving the MaskGIT transformer a position-agnostic
        # style summary alongside the position-specific fused features.
        if self.global_style_projection is not None:
            style_global = self.encode_style_global(style_reference_images)
            # Apply per-batch dropout to the global style vector.
            if self.training and self.config.global_style_dropout > 0:
                style_global = F.dropout(
                    style_global, p=self.config.global_style_dropout
                )
            fused = fused + style_global[:, :, None, None]  # (B, C, H, W)

        # Codepoint embedding from LATIN_CORE index.
        codepoint_emb = self.codepoint_embedding(latincore_idx)  # (B, C)

        # Add metric embedding if available.
        if (
            metrics is not None
            and self.metric_embedder is not None
        ):
            metric_emb = self.metric_embedder(metrics)  # (B, C)
            codepoint_emb = codepoint_emb + metric_emb

        codepoint_map = codepoint_emb[:, :, None, None].expand(
            -1, -1, fused.shape[2], fused.shape[3]
        )  # (B, C, H, W)

        # Content-only dropout: zero out style features.
        self._content_only_step = False
        if (
            self.training
            and self.config.content_only_prob > 0
            and torch.rand(1).item() < self.config.content_only_prob
        ):
            fused = torch.zeros_like(fused)
            self._content_only_step = True

        # Style-only dropout: zero out codepoint embedding.
        self._style_only_step = False
        if (
            self.training
            and not self._content_only_step
            and self.config.style_only_prob > 0
            and torch.rand(1).item() < self.config.style_only_prob
        ):
            codepoint_map = torch.zeros_like(codepoint_map)
            self._style_only_step = True

        return torch.cat([codepoint_map, fused], dim=1)  # (B, 2C, H, W)

    # ------------------------------------------------------------------
    # Token decoding (PrefixLM: image features prepended to code tokens)
    # ------------------------------------------------------------------

    def soft_decode(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project token logits onto the G-Tok codebook and decode to images."""
        logits = logits / temperature
        probabilities = torch.softmax(logits, dim=-1)
        soft_token_embeddings = torch.matmul(probabilities, self.codebook_embeddings())
        reconstructed_images = self.gtok.decode(soft_token_embeddings)
        return soft_token_embeddings, reconstructed_images

    def target_token_indices_from_images(
        self,
        target_images: torch.Tensor,
    ) -> torch.Tensor:
        """Encode target glyph images into G-Tok codebook indices."""
        batch_size = target_images.shape[0]
        cnn_out = self.gtok.cnn_encoder(target_images)
        tokens = self.gtok.proj_patch(cnn_out).flatten(2).transpose(1, 2)
        vit_tokens = self.gtok.vit_encoder(tokens)
        quantizer_inputs = self.gtok.vit_encoder_to_quantizer(vit_tokens)
        quantizer_inputs = quantizer_inputs.reshape(
            batch_size,
            self.token_grid_height,
            self.token_grid_width,
            self.codebook_dim,
        ).permute(0, 3, 1, 2)
        _quantized, _loss_info, indices_info = self.gtok.quantizer(quantizer_inputs)
        return indices_info[2].reshape(batch_size, self.sequence_length)

    def codebook_embeddings(self) -> torch.Tensor:
        """Return the codebook matrix used for soft and hard decoding."""
        codebook = self.gtok.quantizer.embedding.weight
        if self.gtok.quantizer.l2_norm:
            codebook = F.normalize(codebook, p=2, dim=-1)
        return codebook

    # ------------------------------------------------------------------
    # Training and generation entry points
    # ------------------------------------------------------------------

    def forward(
        self,
        content_images: torch.Tensor,
        style_reference_images: torch.Tensor,
        *,
        target_token_indices: Optional[torch.Tensor] = None,
        target_images: Optional[torch.Tensor] = None,
        target_codepoints: Optional[torch.Tensor] = None,
        metrics: Optional[torch.Tensor] = None,
        global_step: int = 0,
    ) -> ARModelOutput:
        """MaskGIT training / teacher-forced forward pass.

        Args:
            content_images: ``(B, 3, H, W)`` content glyphs.
            style_reference_images: ``(B, n_ref, 3, H, W)`` style references.
            target_token_indices: Optional ``(B, N)`` ground-truth token indices.
            target_images: Optional ``(B, 3, H, W)`` ground-truth images (used
                to derive token indices if not provided).
            target_codepoints: ``(B,)`` Unicode codepoint tensor (required).
            metrics: Optional ``(B, 6)`` normalised metric tensor.
            global_step: Current training step (unused, accepted for
                compatibility).
        """
        if target_codepoints is None:
            raise ValueError("target_codepoints is required")
        target_latincore_idx = self._unicode_to_latincore(target_codepoints)

        conditioning_map = self.build_conditioning_map(
            content_images=content_images,
            style_reference_images=style_reference_images,
            latincore_idx=target_latincore_idx,
            metrics=metrics,
        )

        if target_token_indices is None:
            if target_images is None:
                raise ValueError(
                    "Either target_token_indices or target_images must be provided"
                )
            with torch.no_grad():
                target_token_indices = self.target_token_indices_from_images(
                    target_images,
                )

        if self.training:
            logits, token_mask = self.maskgit_decoder.forward_train(
                target_token_indices=target_token_indices,
                conditioning_map=conditioning_map,
            )
        else:
            logits = self.maskgit_decoder.transformer(
                idx=target_token_indices,
                imgs_feature_map=conditioning_map,
            )
            token_mask = None

        soft_token_embeddings, reconstructed_images = self.soft_decode(
            logits, temperature=1.0
        )

        # Width prediction from pooled token embeddings.
        predicted_width: Optional[torch.Tensor] = None
        if self.width_head is not None:
            pooled = soft_token_embeddings.mean(dim=1)
            predicted_width = self.width_head(pooled)

        return ARModelOutput(
            logits=logits,
            reconstructed_images=reconstructed_images,
            soft_token_embeddings=soft_token_embeddings,
            target_token_indices=target_token_indices,
            token_mask=token_mask,
            predicted_width=predicted_width,
        )

    @torch.no_grad()
    def generate(
        self,
        content_images: torch.Tensor,
        style_reference_images: torch.Tensor,
        target_codepoints: torch.Tensor,
        *,
        metrics: Optional[torch.Tensor] = None,
        zero_aggregator: bool = False,
    ) -> ARModelOutput:
        """Generate target glyphs via MaskGIT iterative decoding.

        Args:
            content_images: ``(B, 3, H, W)`` content glyphs.
            style_reference_images: ``(B, n_ref, 3, H, W)`` style references.
            target_codepoints: ``(B,)`` Unicode codepoint tensor.
            metrics: Optional ``(B, 6)`` normalised metric tensor.
            zero_aggregator: If True, zero out the per-position cross-attention
                output, isolating the global style vector + codepoint embedding.
        """
        latincore_idx = self._unicode_to_latincore(target_codepoints)
        conditioning_map = self.build_conditioning_map(
            content_images=content_images,
            style_reference_images=style_reference_images,
            latincore_idx=latincore_idx,
            metrics=metrics,
            zero_aggregator=zero_aggregator,
        )

        predicted = self.maskgit_decoder.generate(
            conditioning_map=conditioning_map,
        )

        logits = self.maskgit_decoder.transformer(
            idx=predicted,
            imgs_feature_map=conditioning_map,
        )

        soft_token_embeddings, reconstructed_images = self.soft_decode(
            logits, temperature=1.0
        )

        predicted_width: Optional[torch.Tensor] = None
        if self.width_head is not None:
            pooled = soft_token_embeddings.mean(dim=1)
            predicted_width = self.width_head(pooled)

        return ARModelOutput(
            logits=logits,
            reconstructed_images=reconstructed_images,
            soft_token_embeddings=soft_token_embeddings,
            target_token_indices=predicted,
            predicted_width=predicted_width,
        )

    def load(self, path: str, device: torch.device) -> None:
        """Load AR model weights from a checkpoint.

        ``gtok.*``, ``token_decoder.*``, and ``lookahead_decoders.*`` keys
        are stripped before loading.  The ``gtok.*`` keys are loaded
        separately; the other prefixes come from the removed AR decoder
        and are harmlessly skipped.
        """
        state_dict = torch.load(path, map_location=device, weights_only=True)

        gtok_keys = {k: v for k, v in state_dict.items() if k.startswith("gtok.")}
        adapter_keys = {
            k.removeprefix("language_adapter."): v
            for k, v in state_dict.items()
            if k.startswith("language_adapter.")
        }
        core_keys = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith("gtok.")
            and not k.startswith("language_adapter.")
            and not k.startswith("token_decoder.")
            and not k.startswith("lookahead_decoders.")
        }

        if gtok_keys:
            print(
                f"ARModel.load: skipping {len(gtok_keys)} gtok.* keys "
                "(G-Tok is loaded separately)"
            )

        if adapter_keys and self.language_adapter is None:
            print(
                f"ARModel.load: checkpoint contains {len(adapter_keys)} "
                "language_adapter.* keys but no adapter is registered -- skipping"
            )
        elif not adapter_keys and self.language_adapter is not None:
            print(
                "ARModel.load: no language_adapter.* keys in checkpoint -- "
                "adapter will keep its default initialisation"
            )
        elif adapter_keys and self.language_adapter is not None:
            missing, unexpected = self.language_adapter.load_state_dict(
                adapter_keys, strict=False
            )
            if unexpected:
                print(
                    f"ARModel.load: ignored {len(unexpected)} unexpected "
                    f"adapter key(s): {unexpected}"
                )
            if missing:
                print(
                    f"ARModel.load: {len(missing)} adapter key(s) absent "
                    f"in checkpoint (kept at default init): {missing}"
                )

        incompatible = self.load_state_dict(core_keys, strict=False)

        allowed_missing_prefixes = [
            "gtok.", "token_decoder.", "lookahead_decoders.",
            "metric_embedder.", "width_head.", "aggregator.style_pool."
        ]
        if self.language_adapter is not None:
            allowed_missing_prefixes.append("language_adapter.")
        if self.global_style_projection is not None:
            allowed_missing_prefixes.append("global_style_projection.")

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
                "ARModel.load failed due to checkpoint/schema mismatch: "
                + "; ".join(details)
            )

    def parameter_counts(self) -> Dict[str, int]:
        """Return parameter counts for the main AR components."""
        counts: Dict[str, int] = {
            "content_encoder": sum(
                p.numel() for p in self.content_encoder.parameters()
            ),
            "style_encoder": sum(p.numel() for p in self.style_encoder.parameters()),
            "aggregator": sum(p.numel() for p in self.aggregator.parameters()),
            "total_trainable": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }
        counts["maskgit_decoder"] = sum(
            p.numel() for p in self.maskgit_decoder.parameters()
        )
        if self.metric_embedder is not None:
            counts["metric_embedder"] = sum(
                p.numel() for p in self.metric_embedder.parameters()
            )
        if self.width_head is not None:
            counts["width_head"] = sum(
                p.numel() for p in self.width_head.parameters()
            )
        if self.global_style_projection is not None:
            counts["global_style_projection"] = sum(
                p.numel() for p in self.global_style_projection.parameters()
            )
        return counts
