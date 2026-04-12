"""Tests for G-Tok loss aggregation."""

import torch

from hrothgar.gtok.losses import GtokLossWeights, compute_gtok_loss


def _dummy_perceptual_loss(
    reconstructed_images: torch.Tensor,
    target_images: torch.Tensor,
) -> torch.Tensor:
    """Simple deterministic perceptual proxy for tests."""
    return torch.mean((reconstructed_images - target_images) ** 2)


def test_compute_gtok_loss_with_all_terms() -> None:
    reconstructed = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    target = torch.ones((1, 3, 2, 2), dtype=torch.float32)

    vq_loss_info = (
        torch.tensor(2.0),
        torch.tensor(3.0),
        torch.tensor(4.0),
        0.5,
    )

    weights = GtokLossWeights(
        l1=1.0,
        perceptual=0.5,
        vq=2.0,
        commit=3.0,
        entropy=4.0,
    )

    total_loss, terms = compute_gtok_loss(
        reconstructed,
        target,
        vq_loss_info,
        perceptual_loss_fn=_dummy_perceptual_loss,
        weights=weights,
    )

    # reconstructed=zeros, target=ones => L1=1, MSE=1
    expected = 1.0 + 0.5 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0 + 4.0 * 4.0
    assert torch.isclose(total_loss, torch.tensor(expected, dtype=torch.float32))

    assert torch.isclose(terms["l1"], torch.tensor(1.0))
    assert torch.isclose(terms["perceptual"], torch.tensor(1.0))
    assert torch.isclose(terms["vq"], torch.tensor(2.0))
    assert torch.isclose(terms["commit"], torch.tensor(3.0))
    assert torch.isclose(terms["entropy"], torch.tensor(4.0))
    assert torch.isclose(terms["codebook_usage"], torch.tensor(0.5))


def test_compute_gtok_loss_handles_missing_vq_terms() -> None:
    reconstructed = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    target = torch.ones((1, 3, 2, 2), dtype=torch.float32)

    vq_loss_info = (None, None, None, 0.0)

    total_loss, terms = compute_gtok_loss(
        reconstructed,
        target,
        vq_loss_info,
        perceptual_loss_fn=None,
        weights=GtokLossWeights(),
    )

    # Only L1 contributes when perceptual and VQ terms are absent.
    assert torch.isclose(total_loss, torch.tensor(1.0))
    assert torch.isclose(terms["vq"], torch.tensor(0.0))
    assert torch.isclose(terms["commit"], torch.tensor(0.0))
    assert torch.isclose(terms["entropy"], torch.tensor(0.0))
    assert torch.isclose(terms["perceptual"], torch.tensor(0.0))
