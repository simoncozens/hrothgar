"""Tests for G-Tok loss aggregation."""

import torch

from hrothgar.gtok.losses import (
    GtokLossInfo,
    GtokLossWeights,
    _sobel_gradient_magnitude,
    compute_gtok_loss,
)


def _dummy_perceptual_loss(
    reconstructed_images: torch.Tensor,
    target_images: torch.Tensor,
) -> torch.Tensor:
    """Simple deterministic perceptual proxy for tests."""
    return torch.mean((reconstructed_images - target_images) ** 2)


def test_compute_gtok_loss_with_all_terms() -> None:
    reconstructed = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    target = torch.ones((1, 3, 2, 2), dtype=torch.float32)

    vq_loss_info = GtokLossInfo(
        vq_loss=torch.tensor(2.0),
        commit_loss=torch.tensor(3.0),
        entropy_loss=torch.tensor(4.0),
        codebook_usage=0.5,
    )

    weights = GtokLossWeights(
        l1=1.0,
        perceptual=0.5,
        edge=0.0,
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

    vq_loss_info = GtokLossInfo(
        vq_loss=None, commit_loss=None, entropy_loss=None, codebook_usage=0.0
    )

    total_loss, terms = compute_gtok_loss(
        reconstructed,
        target,
        vq_loss_info,
        perceptual_loss_fn=None,
        weights=GtokLossWeights(edge=0.0),
    )

    # Only L1 contributes when perceptual and VQ terms are absent.
    assert torch.isclose(total_loss, torch.tensor(1.0))
    assert torch.isclose(terms["vq"], torch.tensor(0.0))
    assert torch.isclose(terms["commit"], torch.tensor(0.0))
    assert torch.isclose(terms["entropy"], torch.tensor(0.0))
    assert torch.isclose(terms["perceptual"], torch.tensor(0.0))


def test_sobel_gradient_magnitude_shape() -> None:
    """Output shape must match input shape."""
    images = torch.rand(2, 3, 8, 8)
    magnitude = _sobel_gradient_magnitude(images)
    assert magnitude.shape == images.shape


def test_sobel_gradient_magnitude_uniform_image_has_low_gradient() -> None:
    """A uniform (flat) image has near-zero gradient everywhere."""
    uniform = torch.ones(1, 1, 16, 16)
    magnitude = _sobel_gradient_magnitude(uniform)
    # Interior pixels should be essentially zero; padded border may be nonzero.
    assert magnitude[:, :, 1:-1, 1:-1].max().item() < 1e-4


def test_sobel_gradient_magnitude_edge_image_has_high_gradient() -> None:
    """An image with a hard vertical edge should produce large gradient values."""
    edge_image = torch.zeros(1, 1, 16, 16)
    edge_image[:, :, :, 8:] = 1.0  # Right half is white
    magnitude = _sobel_gradient_magnitude(edge_image)
    # Maximum gradient should be well above zero along the edge column.
    assert magnitude.max().item() > 0.5


def test_compute_gtok_loss_edge_term_logged() -> None:
    """Edge loss term must be present in returned terms dict."""
    reconstructed = torch.zeros(1, 1, 8, 8)
    target = torch.ones(1, 1, 8, 8)
    vq_loss_info = (None, None, None, 0.0)

    _, terms = compute_gtok_loss(
        reconstructed, target, vq_loss_info, weights=GtokLossWeights(edge=1.0)
    )

    assert "edge" in terms
    assert "weighted_edge" in terms
    # Non-identical images must produce a positive edge loss.
    assert terms["edge"].item() > 0.0
    assert terms["weighted_edge"].item() > 0.0


def test_compute_gtok_loss_edge_weight_zero_does_not_affect_total() -> None:
    """With edge weight 0.0 the edge term must not contribute to total loss."""
    reconstructed = torch.zeros(1, 1, 8, 8)
    target = torch.ones(1, 1, 8, 8)
    vq_loss_info = (None, None, None, 0.0)

    total_no_edge, _ = compute_gtok_loss(
        reconstructed, target, vq_loss_info, weights=GtokLossWeights(edge=0.0)
    )
    total_with_edge, _ = compute_gtok_loss(
        reconstructed, target, vq_loss_info, weights=GtokLossWeights(edge=1.0)
    )

    assert total_with_edge.item() > total_no_edge.item()
