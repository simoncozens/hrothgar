"""Tests for light G-Tok fine-tuning helpers."""

import torch

from hrothgar.gtok.finetune import configure_decoder_only_finetuning
from hrothgar.gtok.model import GtokConfig, GtokModel


def test_configure_decoder_only_finetuning_freezes_encoder_and_quantizer() -> None:
    config = GtokConfig(
        image_size=64,
        cnn_base_channels=32,
        cnn_channel_multipliers=[1, 2, 2, 4],
        cnn_latent_channels=64,
        vit_hidden_dim=96,
        vit_num_layers=2,
        vit_num_heads=4,
        vit_mlp_dim=192,
        quantizer_codebook_size=64,
        quantizer_code_dim=8,
    )
    model = GtokModel(config)

    trainable_names = configure_decoder_only_finetuning(model)

    assert trainable_names
    assert all(
        name.startswith(("quantizer_to_vit_decoder", "vit_decoder", "cnn_decoder"))
        for name in trainable_names
    )

    for name, parameter in model.named_parameters():
        expected = name.startswith(
            ("quantizer_to_vit_decoder", "vit_decoder", "cnn_decoder")
        )
        assert parameter.requires_grad == expected, name


def test_configure_decoder_only_finetuning_leaves_some_parameters_trainable() -> None:
    model = GtokModel(GtokConfig())

    configure_decoder_only_finetuning(model)

    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    frozen_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if not parameter.requires_grad
    )

    assert trainable_parameter_count > 0
    assert frozen_parameter_count > 0
