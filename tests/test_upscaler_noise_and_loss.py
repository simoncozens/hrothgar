"""Regression tests for upscaler low-res corruption and BCE stability."""

import torch

from hrothgar.upscaler.dataset import UpscalerDatasetMaker
from hrothgar.upscaler.train import compute_upscaler_loss


class _DummyEagleLoss:
    def __call__(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros((), dtype=predictions.dtype, device=predictions.device)


def test_low_res_noise_injection_is_monochrome() -> None:
    maker = UpscalerDatasetMaker.__new__(UpscalerDatasetMaker)
    maker.low_res_noise_std = 1.0

    low_res = torch.zeros((2, 3, 8, 8), dtype=torch.float32)
    noised = maker._inject_monochrome_low_res_noise(low_res)

    assert torch.allclose(noised[:, 0, :, :], noised[:, 1, :, :])
    assert torch.allclose(noised[:, 1, :, :], noised[:, 2, :, :])
    assert torch.all((noised >= 0.0) & (noised <= 1.0))


def test_compute_upscaler_loss_handles_invalid_prediction_values() -> None:
    predictions = torch.full((1, 3, 8, 8), 0.5, dtype=torch.float32)
    predictions[0, 0, 0, 0] = float("nan")
    predictions[0, 1, 0, 1] = 1.5
    predictions[0, 2, 0, 2] = -0.5

    targets = torch.zeros_like(predictions)

    loss, terms = compute_upscaler_loss(
        predictions=predictions,
        targets=targets,
        eagle_loss=_DummyEagleLoss(),
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(terms["bce"])
    assert torch.isfinite(terms["edge_l1"])
    assert torch.isfinite(terms["eagle"])
