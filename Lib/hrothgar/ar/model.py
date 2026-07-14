"""MaskGIT-based glyph generator for GAR-Font.

This module implements the vision-only stage of the GAR-Font generator:

1. A frozen G-Tok CNN encoder extracts structural content features.
2. The upstream GAR-Font StyleEncoder extracts visual style features.
3. A content-style aggregator (FeatureFusionModule) fuses both streams.
4. A bidirectional MaskGIT transformer predicts G-Tok codebook indices
   via masked token prediction (like BERT).
5. A soft codebook projection feeds the frozen G-Tok decoder to reconstruct
   images.
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
from hrotgar.ar.config import ARModelConfig
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


# ---------------------------------------------------------------------------
# ARModel
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ARModel
# ---------------------------------------------------------------------------

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

        self._gtok_frozen: bool = False
        if config.freeze_gtok:
            self.freeze_gtok()

        self._global_step: int = 0
        self._content_only_step: bool = False
        self._style_only_step: bool = False


    def _register_latincore_mapping(self) -> None:
        """Build a buffer that maps Unicode codepoints → LATIN_CORE indices.

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
    # Encoding and conditioning (PrefixLM: 2D feature maps preserved)
    # ------------------------------------------------------------------


    def encode_content(self, content_images: torch.Tensor) -> torch.Tensor:
        """Encode content glyphs to a 2D feature map ``(B, C, H, W)``."""
        content_features = self.content_encoder(content_images)
        _batch, _channels, height, width = content_features.shape
        if height != self.token_grid_height or width != self.token_grid_width:
            raise ValueError(
                "Content encoder output shape does not match G-Tok token grid "
                f"(got {(height, width)}, expected {(self.token_grid_height, self.token_grid_width)})"
            )
        return content_features  # (B, C, H, W) — 2D preserved


    def encode_style(self, style_reference_images: torch.Tensor) -> torch.Tensor:
        """Encode style references to a 2D feature map ``(B, n_ref, C, H, W)``.

        Style images are normalised from ``[0, 1]`` to ``[-1, 1]`` before the
        style encoder because the upstream GPT architecture (RoPE, RMSNorm,
        SwiGLU) was designed and tuned for zero-centred image features.
        The G-Tok content path stays at ``[0, 1]`` — its encoder was
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
        # Normalise from [0, 1] to [-1, 1] — upstream GPT expects zero-centred features.
        flattened = (flattened - 0.5) / 0.5
        encoded = self.style_encoder(flattened)
        # Dropout on style features to reduce font-specific memorisation.
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
        )  # (B, n_ref, C, H, W) — 2D preserved


    def build_conditioning_map(
        self,
        content_images: torch.Tensor,
        style_reference_images: torch.Tensor,
        latincore_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Build the 2D conditioning feature map for PrefixLM.

        Returns ``(B, 2*encoder_feature_dim, H, W)`` — codepoint identity
        embedding projected to the feature dimension, and style-fused features
        concatenated along the channel axis.

        The codepoint embedding provides explicit, font-agnostic character
        identity.  This replaces structural content images as the primary
        signal, because Latin glyph structure varies too much across fonts for
        a reference image to be reliable.
        """
        content_features = self.encode_content(content_images)  # (B, C, H, W)
        style_features = self.encode_style(
            style_reference_images
        )  # (B, n_ref, C, H, W)
        fused = self.aggregator(content_features, style_features)  # (B, C, H, W)

        # Codepoint embedding from LATIN_CORE index.
        codepoint_emb = self.codepoint_embedding(latincore_idx)  # (B, C)
        codepoint_map = codepoint_emb[:, :, None, None].expand(
            -1, -1, fused.shape[2], fused.shape[3]
        )  # (B, C, H, W)

        # Content-only dropout: zero out style features with configured
        # probability during training.
        self._content_only_step = False
        if (
            self.training
            and self.config.content_only_prob > 0
            and torch.rand(1).item() < self.config.content_only_prob
        ):
            fused = torch.zeros_like(fused)
            self._content_only_step = True

        # Style-only dropout: zero out codepoint embedding with configured
        # probability.  Forces the model to extract character identity from
        # the fused features, which carries gradient to the aggregator and
        # style encoder.  Mutually exclusive with content-only — if both
        # trigger, content-only wins (zeroing both paths would be unlearnable).
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


    # ------------------------------------------------------------------
    # Training and generation
    # ------------------------------------------------------------------

    def forward(
        self,
        content_images: torch.Tensor,
        style_reference_images: torch.Tensor,
        *,
        target_token_indices: Optional[torch.Tensor] = None,
        target_images: Optional[torch.Tensor] = None,
        target_codepoints: Optional[torch.Tensor] = None,
        global_step: int = 0,
    ) -> ARModelOutput:
        """MaskGIT training / teacher-forced forward pass."""
        if target_codepoints is None:
            raise ValueError("target_codepoints is required")
        target_latincore_idx = self._unicode_to_latincore(target_codepoints)

        conditioning_map = self.build_conditioning_map(
            content_images=content_images,
            style_reference_images=style_reference_images,
            latincore_idx=target_latincore_idx,
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

        return ARModelOutput(
            logits=logits,
            reconstructed_images=reconstructed_images,
            soft_token_embeddings=soft_token_embeddings,
            target_token_indices=target_token_indices,
            token_mask=token_mask,
        )

    @torch.no_grad()
    def generate(
        self,
        content_images: torch.Tensor,
        style_reference_images: torch.Tensor,
        target_codepoints: torch.Tensor,
    ) -> ARModelOutput:
        """Generate target glyphs via MaskGIT iterative decoding."""
        latincore_idx = self._unicode_to_latincore(target_codepoints)
        conditioning_map = self.build_conditioning_map(
            content_images=content_images,
            style_reference_images=style_reference_images,
            latincore_idx=latincore_idx,
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

        return ARModelOutput(
            logits=logits,
            reconstructed_images=reconstructed_images,
            soft_token_embeddings=soft_token_embeddings,
            target_token_indices=predicted,
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
                "language_adapter.* keys but no adapter is registered — skipping"
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
            "gtok.", "token_decoder.", "lookahead_decoders."
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
            "total_trainable": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }
        if self.config.use_maskgit and self.maskgit_decoder is not None:
            counts["maskgit_decoder"] = sum(
                p.numel() for p in self.maskgit_decoder.parameters()
            )
        else:
            counts["token_decoder"] = sum(
                p.numel()
                for p in self.token_decoder.parameters()  # type: ignore[union-attr]
            )
            counts["lookahead_decoders"] = sum(
                p.numel() for p in self.lookahead_decoders.parameters()
            )
        return counts

