"""Tests for the GAR-Font autoregressive generator definition."""

import torch
import torch.nn as nn
import pytest

from hrothgar.ar.model import ARModel, ARModelConfig, ContentStyleAggregator
from hrothgar.gtok.model import GtokConfig, GtokModel


def make_test_models() -> ARModel:
    gtok = GtokModel(
        GtokConfig(
            image_size=64,
            cnn_base_channels=32,
            cnn_latent_channels=64,
            vit_hidden_dim=128,
            vit_num_layers=2,
            vit_num_heads=4,
            vit_mlp_dim=256,
        )
    )
    config = ARModelConfig(
        image_size=64,
        encoder_feature_dim=64,
        content_encoder_base_channels=32,
        style_encoder_base_channels=32,
        aggregator_num_heads=4,
        decoder_hidden_dim=128,
        decoder_num_layers=2,
        decoder_num_heads=4,
        decoder_mlp_dim=256,
    )
    return ARModel(config=config, gtok_model=gtok)


def test_content_style_aggregator_shape() -> None:
    aggregator = ContentStyleAggregator(feature_dim=256, num_heads=8, num_layers=3)
    content_tokens = torch.randn(2, 256, 256)
    style_tokens = torch.randn(2, 8 * 256, 256)

    output = aggregator(content_tokens, style_tokens)

    assert output.shape == content_tokens.shape
    assert sum(p.numel() for p in aggregator.parameters()) == 786432


def test_ar_model_forward_teacher_forcing() -> None:
    model = make_test_models()
    model.eval()

    content_images = torch.randn(2, 3, 64, 64)
    style_reference_images = torch.randn(2, 4, 3, 64, 64)
    target_images = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        output = model(
            content_images,
            style_reference_images,
            target_images=target_images,
        )

    assert output.logits.shape == (2, model.sequence_length, model.codebook_size)
    assert output.reconstructed_images.shape == target_images.shape
    assert output.soft_token_embeddings.shape == (
        2,
        model.sequence_length,
        model.codebook_dim,
    )
    assert output.conditioning_tokens.shape == (
        2,
        model.sequence_length,
        model.config.encoder_feature_dim * 2,
    )
    assert output.target_token_indices is not None
    assert output.target_token_indices.shape == (2, model.sequence_length)


def test_ar_model_generate_shape() -> None:
    model = make_test_models()
    model.eval()

    content_images = torch.randn(1, 3, 64, 64)
    style_reference_images = torch.randn(1, 2, 3, 64, 64)

    with torch.no_grad():
        output = model.generate(content_images, style_reference_images)

    assert output.reconstructed_images.shape == content_images.shape
    assert output.target_token_indices is not None
    assert output.target_token_indices.shape == (1, model.sequence_length)


class DummyLanguageAdapter(nn.Module):
    def __init__(self, token_dim: int, text_dim: int) -> None:
        super().__init__()
        self.text_projection = nn.Linear(text_dim, token_dim)

    def forward(
        self, style_tokens: torch.Tensor, text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        # text_embeddings: (batch, text_dim) -> (batch, 1, token_dim)
        text_bias = self.text_projection(text_embeddings).unsqueeze(1)
        return style_tokens + text_bias


def test_adaptation_requires_registered_adapter() -> None:
    model = make_test_models()
    model.eval()

    content_images = torch.randn(1, 3, 64, 64)
    style_reference_images = torch.randn(1, 2, 3, 64, 64)
    text_embeddings = torch.randn(1, 16)

    with pytest.raises(RuntimeError):
        _ = model.forward_adaptation(
            content_images,
            style_reference_images,
            text_embeddings,
            run_decoder=False,
        )


def test_forward_adaptation_shapes() -> None:
    model = make_test_models()
    model.set_language_adapter(DummyLanguageAdapter(token_dim=64, text_dim=16))
    model.eval()

    content_images = torch.randn(2, 3, 64, 64)
    style_reference_images = torch.randn(2, 3, 3, 64, 64)
    text_embeddings = torch.randn(2, 16)
    target_images = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        output = model.forward_adaptation(
            content_images,
            style_reference_images,
            text_embeddings,
            target_images=target_images,
            run_decoder=True,
        )

    assert output.multimodal_conditioning_tokens.shape == (
        2,
        model.sequence_length,
        model.config.encoder_feature_dim * 2,
    )
    assert (
        output.visual_conditioning_tokens.shape
        == output.multimodal_conditioning_tokens.shape
    )
    assert output.multimodal_aggregated_style_tokens.shape == (
        2,
        model.sequence_length,
        model.config.encoder_feature_dim,
    )
    assert (
        output.visual_aggregated_style_tokens.shape
        == output.multimodal_aggregated_style_tokens.shape
    )
    assert output.logits is not None
    assert output.reconstructed_images is not None
    assert output.target_token_indices is not None


def test_generate_adaptation_shapes() -> None:
    model = make_test_models()
    model.set_language_adapter(DummyLanguageAdapter(token_dim=64, text_dim=16))
    model.eval()

    content_images = torch.randn(1, 3, 64, 64)
    style_reference_images = torch.randn(1, 2, 3, 64, 64)
    text_embeddings = torch.randn(1, 16)

    with torch.no_grad():
        output = model.generate_adaptation(
            content_images,
            style_reference_images,
            text_embeddings,
        )

    assert output.reconstructed_images is not None
    assert output.reconstructed_images.shape == content_images.shape
    assert output.target_token_indices is not None
    assert output.target_token_indices.shape == (1, model.sequence_length)


def test_freeze_unfreeze_visual_style_path() -> None:
    model = make_test_models()

    model.freeze_visual_style_path()
    assert all(not p.requires_grad for p in model.style_encoder.parameters())
    assert all(not p.requires_grad for p in model.aggregator.parameters())

    model.unfreeze_visual_style_path()
    assert all(p.requires_grad for p in model.style_encoder.parameters())
    assert all(p.requires_grad for p in model.aggregator.parameters())
