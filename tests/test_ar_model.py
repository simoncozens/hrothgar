"""Tests for the GAR-Font autoregressive generator definition."""

import torch
import torch.nn as nn
import pytest

from hrothgar.ar.model import (
    ARModel,
    ARModelConfig,
    ContentStyleAggregator,
    LoRAConfig,
    LoRALinear,
)
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


# ---------------------------------------------------------------------------
# LoRA tests
# ---------------------------------------------------------------------------


def test_lora_linear_zero_init_unchanged_output() -> None:
    """A freshly constructed LoRALinear should produce the same output as its base."""
    base = nn.Linear(32, 64)
    lora = LoRALinear(base, rank=4, alpha=4.0)
    x = torch.randn(2, 32)
    with torch.no_grad():
        assert torch.allclose(lora(x), base(x))


def test_lora_linear_base_frozen() -> None:
    """Base weights inside LoRALinear must not be trainable."""
    base = nn.Linear(16, 32)
    lora = LoRALinear(base, rank=4, alpha=4.0)
    assert all(not p.requires_grad for p in lora.base.parameters())
    assert lora.lora_A.requires_grad
    assert lora.lora_B.requires_grad
    assert lora.lora_A.device == base.weight.device
    assert lora.lora_B.device == base.weight.device
    assert lora.lora_A.dtype == base.weight.dtype
    assert lora.lora_B.dtype == base.weight.dtype


def test_lora_linear_modifies_output_after_b_init() -> None:
    """After manually setting lora_B to non-zero, output should differ from base."""
    base = nn.Linear(16, 32, bias=False)
    lora = LoRALinear(base, rank=4, alpha=4.0)
    with torch.no_grad():
        lora.lora_B.fill_(1.0)
    x = torch.randn(2, 16)
    with torch.no_grad():
        assert not torch.allclose(lora(x), base(x))


def test_inject_lora_only_lora_params_trainable() -> None:
    """After enable_nfa_mode, only LoRA params in the decoder should be trainable."""
    model = make_test_models()
    lora_config = LoRAConfig(rank=4, alpha=4.0)
    model.enable_nfa_mode(lora_config)

    for name, param in model.named_parameters():
        if "lora_" in name:
            assert param.requires_grad, f"LoRA param {name} should be trainable"
        else:
            assert not param.requires_grad, f"Non-LoRA param {name} should be frozen"


def test_inject_lora_trainable_param_count_is_small() -> None:
    """LoRA parameter count should be a small fraction of total parameters."""
    model = make_test_models()
    total_before = sum(p.numel() for p in model.parameters())
    model.enable_nfa_mode(LoRAConfig(rank=4, alpha=4.0))
    trainable = sum(p.numel() for p in model.trainable_parameters())
    # With rank=4, LoRA should be < 5% of total parameters.
    assert (
        trainable < 0.05 * total_before
    ), f"LoRA params ({trainable:,}) unexpectedly large vs total ({total_before:,})"


def test_enable_nfa_mode_idempotent_raises() -> None:
    """Calling enable_nfa_mode twice should raise RuntimeError."""
    model = make_test_models()
    model.enable_nfa_mode(LoRAConfig(rank=4, alpha=4.0))
    with pytest.raises(RuntimeError):
        model.enable_nfa_mode(LoRAConfig(rank=4, alpha=4.0))


def test_inject_lora_idempotent_raises() -> None:
    """Calling inject_lora twice on the decoder should raise RuntimeError."""
    model = make_test_models()
    config = LoRAConfig(rank=4, alpha=4.0)
    model.token_decoder.inject_lora(config)
    with pytest.raises(RuntimeError):
        model.token_decoder.inject_lora(config)


def test_nfa_forward_produces_valid_output() -> None:
    """Model in NFA mode should produce the same output shapes as normal mode."""
    model = make_test_models()
    model.enable_nfa_mode(LoRAConfig(rank=4, alpha=4.0))
    model.eval()

    content_images = torch.randn(2, 3, 64, 64)
    style_images = torch.randn(2, 4, 3, 64, 64)
    target_images = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        output = model(content_images, style_images, target_images=target_images)

    assert output.logits.shape == (2, model.sequence_length, model.codebook_size)
    assert output.reconstructed_images.shape == target_images.shape


def test_lora_state_dict_roundtrip() -> None:
    """get_lora_state_dict / load_lora_state_dict should preserve LoRA values."""
    model = make_test_models()
    model.enable_nfa_mode(LoRAConfig(rank=4, alpha=4.0))

    # Modify LoRA B matrices so they are non-zero.
    with torch.no_grad():
        for name, param in model.token_decoder.named_parameters():
            if "lora_B" in name:
                param.fill_(0.5)

    lora_state = model.token_decoder.get_lora_state_dict()
    assert all("lora_" in k for k in lora_state)

    # Create a second model, inject LoRA, load the state.
    model2 = make_test_models()
    model2.enable_nfa_mode(LoRAConfig(rank=4, alpha=4.0))
    model2.token_decoder.load_lora_state_dict(lora_state)

    for key in lora_state:
        v1 = model.token_decoder.state_dict()[key]
        v2 = model2.token_decoder.state_dict()[key]
        assert torch.allclose(v1, v2), f"Mismatch in {key} after roundtrip"
