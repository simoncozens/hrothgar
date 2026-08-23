"""MaskGIT-based glyph generator with font-level style conditioning.

This module implements the vision-only stage of the glyph generator:

1. A frozen G-Tok CNN encoder extracts structural content features.
2. Per-glyph style reference images are encoded by the upstream GAR-Font
   ``StyleEncoder`` and fused with content via ``FeatureFusionModule``
   cross-attention, preserving spatial style detail (terminals, serifs,
   corners, stroke modulation).
3. A pre-computed font-level style embedding (from FontStyleEmbedder)
   provides a globally-consistent style signal, broadcast to all spatial
   positions and added to the fused features.
4. A bidirectional MaskGIT transformer predicts G-Tok codebook indices
   via masked token prediction (like BERT).
5. A hard codebook projection feeds the frozen G-Tok decoder to
   reconstruct images.

Metric conditioning:
- ``MetricEmbedder`` injects font vertical metrics + per-glyph advance
  width into the conditioning map for baseline/x-height/width alignment.
- ``GlyphBBoxHead`` predicts the ink bounding box (width, height) from token
  embeddings as an auxiliary signal for denormalization.

NFA/GA adaptation:
- ``enable_nfa_mode()`` freezes the model and injects LoRA adapters into
  the MaskGIT transformer for per-font fine-tuning.
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
    token_embeddings: torch.Tensor
    target_token_indices: Optional[torch.Tensor]
    token_mask: Optional[torch.Tensor] = None
    predicted_bbox: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Metric conditioning modules
# ---------------------------------------------------------------------------


class MetricEmbedder(nn.Module):
    """Project vertical metrics into the feature dimension."""

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        output_dim: int = 256,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, metrics: torch.Tensor) -> torch.Tensor:
        """Project ``(B, 6)`` metrics to ``(B, output_dim)``."""
        return self.net(metrics)


class GlyphBBoxHead(nn.Module):
    """Predict the ink bbox ``(width, height)`` from token embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, pooled_embeddings: torch.Tensor) -> torch.Tensor:
        """Predict normalized ``(width, height)`` ``(B, 2)``."""
        return self.net(pooled_embeddings)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class ARModel(SaveLoadModel):
    """MaskGIT glyph generator with font-level style conditioning."""

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

        self.content_encoder = self.gtok.cnn_encoder

        style_downsample_ratio = config.image_size // self.token_grid_height
        if style_downsample_ratio not in (8, 16):
            raise ValueError(
                f"StyleEncoder downsample ratio must be 8 or 16, but "
                f"image_size={config.image_size} / token_grid_height="
                f"{self.token_grid_height} = {style_downsample_ratio}"
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

        # Metric conditioning.
        self.metric_embedder: Optional[MetricEmbedder] = None
        self.bbox_head: Optional[GlyphBBoxHead] = None
        if config.use_metrics:
            self.metric_embedder = MetricEmbedder(
                input_dim=6,
                hidden_dim=config.metric_embedding_hidden_dim,
                output_dim=config.encoder_feature_dim,
            )
            self.bbox_head = GlyphBBoxHead(
                input_dim=self.codebook_dim,
                hidden_dim=config.bbox_head_hidden_dim,
            )

        self._gtok_frozen: bool = False
        if config.freeze_gtok:
            self.freeze_gtok()

        self._nfa_mode: bool = False

    # ------------------------------------------------------------------
    # Codepoint helpers
    # ------------------------------------------------------------------

    def _register_latincore_mapping(self) -> None:
        """Build a buffer mapping Unicode codepoints -> LATIN_CORE indices."""
        max_cp = max(LATIN_CORE)
        mapping = torch.full((max_cp + 1,), -1, dtype=torch.long)
        for idx, cp in enumerate(LATIN_CORE):
            mapping[cp] = idx
        self.register_buffer("_latincore_map", mapping, persistent=False)

    def _unicode_to_latincore(self, codepoints: torch.Tensor) -> torch.Tensor:
        """Convert Unicode codepoint tensor to LATIN_CORE indices."""
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

    # ------------------------------------------------------------------
    # Freezing and training mode
    # ------------------------------------------------------------------

    def freeze_gtok(self) -> None:
        """Freeze G-Tok so the AR stage trains only its own modules."""
        self.gtok.eval()
        for parameter in self.gtok.parameters():
            parameter.requires_grad = False
        self._gtok_frozen = True

    def train(self, mode: bool = True) -> "ARModel":
        """Set training mode while keeping frozen G-Tok in eval mode."""
        super().train(mode)
        if self._gtok_frozen:
            self.gtok.eval()
        return self

    def set_language_adapter(self, adapter: nn.Module) -> None:
        """Register a language adapter module for multimodal adaptation."""
        self.language_adapter = adapter

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Return parameters that currently require gradients."""
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
        """
        if self._nfa_mode:
            raise RuntimeError(
                "Model is already in NFA mode.  "
                "Create a fresh model instance to re-apply NFA."
            )
        for param in self.parameters():
            param.requires_grad = False
        self.maskgit_decoder.transformer.inject_lora(lora_config)
        self.freeze_gtok()
        self._nfa_mode = True

    def enable_composed_nfa_mode(
        self,
        glyph_lora_state: Dict[str, torch.Tensor],
        lora_config,
    ) -> None:
        """Switch to composed NFA mode with a frozen glyph prior."""
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
        """True once NFA or composed NFA mode has been enabled."""
        return self._nfa_mode

    # ------------------------------------------------------------------
    # Encoding
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
        SwiGLU) was designed and tuned for zero-centred image features.  The
        G-Tok content path stays at ``[0, 1]`` -- its encoder was pretrained on
        that range and is frozen during AR training.
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

    # ------------------------------------------------------------------
    # Conditioning map
    # ------------------------------------------------------------------

    def build_conditioning_map(
        self,
        content_images: torch.Tensor,
        style_reference_images: torch.Tensor,
        latincore_idx: torch.Tensor,
        *,
        font_style_embedding: torch.Tensor,
        metrics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build the 2D conditioning feature map for PrefixLM.

        Returns ``(B, 2*encoder_feature_dim, H, W)`` — codepoint identity
        concatenated with fused content+style features along the channel axis.

        Content features are fused with per-glyph style reference features via
        the ``FeatureFusionModule`` cross-attention, preserving spatial style
        detail (terminals, serifs, corners).  The font style embedding is then
        broadcast to all spatial positions and added, giving every generation
        token an identical, globally-consistent style signal on top of the
        spatially-resolved style features.

        Args:
            content_images: ``(B, 3, H, W)`` content glyph renderings.
            style_reference_images: ``(B, n_ref, 3, H, W)`` style references.
            latincore_idx: ``(B,)`` LATIN_CORE indices.
            font_style_embedding: ``(B, encoder_feature_dim)`` pre-computed
                font-level style vector from ``FontStyleEmbedder``.
            metrics: Optional ``(B, 6)`` normalised metric tensor.
        """
        content_features = self.encode_content(content_images)  # (B, C, H, W)
        style_features = self.encode_style(
            style_reference_images
        )  # (B, n_ref, C, H, W)
        fused = self.aggregator(content_features, style_features)  # (B, C, H, W)

        # Broadcast font style to all spatial positions and add to fused.
        if self.training and self.config.font_style_dropout > 0:
            font_style_embedding = F.dropout(
                font_style_embedding, p=self.config.font_style_dropout
            )
        fused = fused + font_style_embedding[:, :, None, None]  # (B, C, H, W)

        # Codepoint embedding.
        codepoint_emb = self.codepoint_embedding(latincore_idx)  # (B, C)
        if metrics is not None and self.metric_embedder is not None:
            metric_emb = self.metric_embedder(metrics)  # (B, C)
            codepoint_emb = codepoint_emb + metric_emb

        codepoint_map = codepoint_emb[:, :, None, None].expand(
            -1, -1, fused.shape[2], fused.shape[3]
        )  # (B, C, H, W)

        return torch.cat([codepoint_map, fused], dim=1)  # (B, 2C, H, W)

    # ------------------------------------------------------------------
    # Token decoding
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

    def hard_decode(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode logits into committed (hard) token embeddings + image.

        Uses a straight-through estimator: the forward pass decodes the argmax
        token, while gradients flow through the soft embedding. This rewards
        committing to the correct token instead of hedging over a spread
        distribution.
        """
        logits = logits / temperature
        indices = logits.argmax(dim=-1)  # (B, N)
        codebook = self.codebook_embeddings()  # (V, D)
        hard = codebook[indices]  # (B, N, D)

        # Straight-through: forward = hard (committed), backward = soft.
        soft = torch.softmax(logits, dim=-1) @ codebook  # (B, N, D)
        token_embeddings = soft + (hard - soft).detach()

        reconstructed_images = self.gtok.decode(token_embeddings)
        return token_embeddings, reconstructed_images

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
        font_style_embedding: torch.Tensor,
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
            font_style_embedding: ``(B, encoder_feature_dim)`` pre-computed
                font-level style vector (required).
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
            font_style_embedding=font_style_embedding,
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

        token_embeddings, reconstructed_images = self.hard_decode(
            logits, temperature=1.0
        )

        predicted_bbox: Optional[torch.Tensor] = None
        if self.bbox_head is not None:
            pooled = token_embeddings.mean(dim=1)
            predicted_bbox = self.bbox_head(pooled)

        return ARModelOutput(
            logits=logits,
            reconstructed_images=reconstructed_images,
            token_embeddings=token_embeddings,
            target_token_indices=target_token_indices,
            token_mask=token_mask,
            predicted_bbox=predicted_bbox,
        )

    @torch.no_grad()
    def generate(
        self,
        content_images: torch.Tensor,
        style_reference_images: torch.Tensor,
        target_codepoints: torch.Tensor,
        *,
        font_style_embedding: torch.Tensor,
        metrics: Optional[torch.Tensor] = None,
    ) -> ARModelOutput:
        """Generate target glyphs via MaskGIT iterative decoding.

        Args:
            content_images: ``(B, 3, H, W)`` content glyphs.
            style_reference_images: ``(B, n_ref, 3, H, W)`` style references.
            target_codepoints: ``(B,)`` Unicode codepoint tensor.
            font_style_embedding: ``(B, encoder_feature_dim)`` pre-computed
                font-level style vector (required).
            metrics: Optional ``(B, 6)`` normalised metric tensor.
        """
        latincore_idx = self._unicode_to_latincore(target_codepoints)
        conditioning_map = self.build_conditioning_map(
            content_images=content_images,
            style_reference_images=style_reference_images,
            latincore_idx=latincore_idx,
            font_style_embedding=font_style_embedding,
            metrics=metrics,
        )

        predicted = self.maskgit_decoder.generate(
            conditioning_map=conditioning_map,
        )

        logits = self.maskgit_decoder.transformer(
            idx=predicted,
            imgs_feature_map=conditioning_map,
        )

        token_embeddings, reconstructed_images = self.hard_decode(
            logits, temperature=1.0
        )

        predicted_bbox: Optional[torch.Tensor] = None
        if self.bbox_head is not None:
            pooled = token_embeddings.mean(dim=1)
            predicted_bbox = self.bbox_head(pooled)

        return ARModelOutput(
            logits=logits,
            reconstructed_images=reconstructed_images,
            token_embeddings=token_embeddings,
            target_token_indices=predicted,
            predicted_bbox=predicted_bbox,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def load(self, path: str, device: torch.device, strict: bool = False) -> None:
        """Load AR model weights from a checkpoint.

        ``gtok.*``, ``token_decoder.*``, and ``lookahead_decoders.*`` keys
        are stripped before loading (G-Tok is loaded separately; the other two
        come from the removed AR decoder and are harmlessly skipped).
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
            "metric_embedder.", "bbox_head.",
        ]
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
            "maskgit_decoder": sum(
                p.numel() for p in self.maskgit_decoder.parameters()
            ),
            "total_trainable": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }
        if self.metric_embedder is not None:
            counts["metric_embedder"] = sum(
                p.numel() for p in self.metric_embedder.parameters()
            )
        if self.bbox_head is not None:
            counts["bbox_head"] = sum(
                p.numel() for p in self.bbox_head.parameters()
            )
        return counts
