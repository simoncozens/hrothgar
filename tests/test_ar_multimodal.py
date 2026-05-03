"""Tests for multimodal AR adaptation components."""

import torch

from hrothgar.ar.losses import ARAdaptationLossWeights, compute_ar_adaptation_loss
from hrothgar.ar.model import ARAdaptationOutput
from hrothgar.ar.multimodal import (
    HashedDescriptionEncoder,
    HashedDescriptionEncoderConfig,
    TextStyleAdapter,
    TextStyleAdapterConfig,
)


def test_hashed_description_encoder_shape_and_repeatability() -> None:
    encoder = HashedDescriptionEncoder(
        HashedDescriptionEncoderConfig(vocab_size=128, embedding_dim=32, max_tokens=8)
    )
    descriptions = ["script lively formal", "display retro rounded"]

    out1 = encoder(descriptions)
    out2 = encoder(descriptions)

    assert out1.shape == (2, 8, 32)
    assert torch.allclose(out1, out2)


def test_text_style_adapter_preserves_shape() -> None:
    adapter = TextStyleAdapter(
        TextStyleAdapterConfig(
            style_token_dim=64,
            text_embedding_dim=32,
            adapter_hidden_dim=64,
            num_layers=2,
            num_heads=4,
        )
    )

    style_tokens = torch.randn(3, 10, 64)
    text_tokens = torch.randn(3, 6, 32)

    output = adapter(style_tokens, text_tokens)

    assert output.shape == style_tokens.shape


def test_compute_ar_adaptation_loss_alignment_only() -> None:
    output = ARAdaptationOutput(
        multimodal_conditioning_tokens=torch.randn(2, 4, 8),
        visual_conditioning_tokens=torch.randn(2, 4, 8),
        multimodal_aggregated_style_tokens=torch.ones(2, 4, 4),
        visual_aggregated_style_tokens=torch.zeros(2, 4, 4),
        logits=None,
        reconstructed_images=None,
        soft_token_embeddings=None,
        target_token_indices=None,
    )

    total, terms = compute_ar_adaptation_loss(output)

    assert total.ndim == 0
    assert "alignment_l2" in terms
    assert "total" in terms
    assert terms["alignment_l2"] > 0


def test_compute_ar_adaptation_loss_with_decoder_terms() -> None:
    token_targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    output = ARAdaptationOutput(
        multimodal_conditioning_tokens=torch.randn(2, 2, 6),
        visual_conditioning_tokens=torch.randn(2, 2, 6),
        multimodal_aggregated_style_tokens=torch.randn(2, 2, 3),
        visual_aggregated_style_tokens=torch.randn(2, 2, 3),
        logits=torch.randn(2, 2, 4),
        reconstructed_images=torch.zeros(2, 3, 4, 4),
        soft_token_embeddings=torch.randn(2, 2, 8),
        target_token_indices=token_targets,
    )
    target_images = torch.ones(2, 3, 4, 4)

    total, terms = compute_ar_adaptation_loss(
        output,
        target_images=target_images,
        weights=ARAdaptationLossWeights(
            alignment_l2=1.0,
            token_cross_entropy=1.0,
            pixel_l1=1.0,
        ),
    )

    assert total.ndim == 0
    assert "token_cross_entropy" in terms
    assert "pixel_l1" in terms
    assert "token_accuracy" in terms
