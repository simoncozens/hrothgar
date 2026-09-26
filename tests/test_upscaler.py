"""Tests for the simplified (no style conditioning) glyph upscaler."""

import pytest
import torch

from hrothgar.upscaler.model import UpscalerConfig, UpscalerModel
from hrothgar.upscaler.train import compute_upscaler_loss


def test_config_rejects_invalid_sizes() -> None:
    with pytest.raises(ValueError):
        UpscalerConfig(low_res_size=128, high_res_size=64)  # high <= low
    with pytest.raises(ValueError):
        UpscalerConfig(low_res_size=100, high_res_size=512)  # not divisible


def test_model_forward_shape_and_range() -> None:
    config = UpscalerConfig(
        low_res_size=32, high_res_size=128, base_channels=16, num_residual_blocks=2
    )
    model = UpscalerModel(config)
    x = torch.rand(2, 1, 32, 32)
    out = model(x)
    assert out.shape == (2, 1, 128, 128)
    assert torch.all((out >= 0.0) & (out <= 1.0))


def test_compute_upscaler_loss_handles_invalid_values() -> None:
    predictions = torch.full((1, 1, 8, 8), 0.5)
    predictions[0, 0, 0, 0] = float("nan")
    predictions[0, 0, 0, 1] = 1.5
    predictions[0, 0, 0, 2] = -0.5
    targets = torch.zeros_like(predictions)

    loss, terms = compute_upscaler_loss(predictions, targets)

    assert torch.isfinite(loss)
    assert torch.isfinite(terms["bce"])
    assert torch.isfinite(terms["glyphloss"])
    assert "loss" in terms
