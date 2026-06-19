"""Tests for G-Tok loss aggregation."""

import torch

from hrothgar.gtok.losses import (
    GtokLossInfo,
    GtokLossWeights,
    compute_gtok_loss,
)


def _dummy_glyphloss(
    reconstructed_images: torch.Tensor,
    target_images: torch.Tensor,
) -> torch.Tensor:
    """Simple deterministic glyph-reconstruction proxy for tests."""
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
        glyphloss=0.5,
        vq=2.0,
        commit=3.0,
        entropy=4.0,
    )

    total_loss, terms = compute_gtok_loss(
        reconstructed,
        target,
        vq_loss_info,
        glyphloss_fn=_dummy_glyphloss,
        weights=weights,
    )

    # reconstructed=zeros, target=ones => glyphloss MSE=1
    expected = 0.5 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0 + 4.0 * 4.0
    assert torch.isclose(total_loss, torch.tensor(expected, dtype=torch.float32))

    assert torch.isclose(terms["glyphloss"], torch.tensor(1.0))
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
        glyphloss_fn=_dummy_glyphloss,
        weights=GtokLossWeights(),
    )

    # Only glyphloss contributes when VQ terms are absent.
    assert torch.isclose(total_loss, torch.tensor(1.0))
    assert torch.isclose(terms["vq"], torch.tensor(0.0))
    assert torch.isclose(terms["commit"], torch.tensor(0.0))
    assert torch.isclose(terms["entropy"], torch.tensor(0.0))
    assert torch.isclose(terms["glyphloss"], torch.tensor(1.0))


def test_compute_gtok_loss_glyphloss_term_logged() -> None:
    """Glyphloss term must be present in returned terms dict."""
    reconstructed = torch.zeros(1, 1, 8, 8)
    target = torch.ones(1, 1, 8, 8)
    vq_loss_info = (None, None, None, 0.0)

    _, terms = compute_gtok_loss(
        reconstructed, target, vq_loss_info,
        glyphloss_fn=_dummy_glyphloss,
        weights=GtokLossWeights(glyphloss=1.0),
    )

    assert "glyphloss" in terms
    assert "weighted_glyphloss" in terms
    # Non-identical images must produce a positive glyphloss.
    assert terms["glyphloss"].item() > 0.0
    assert terms["weighted_glyphloss"].item() > 0.0


def test_compute_gtok_loss_glyphloss_weight_zero_does_not_affect_total() -> None:
    """With glyphloss weight 0.0 the glyphloss term must not contribute to total."""
    reconstructed = torch.zeros(1, 1, 8, 8)
    target = torch.ones(1, 1, 8, 8)
    vq_loss_info = (None, None, None, 0.0)

    total_no_glyphloss, _ = compute_gtok_loss(
        reconstructed, target, vq_loss_info,
        glyphloss_fn=_dummy_glyphloss,
        weights=GtokLossWeights(glyphloss=0.0),
    )
    total_with_glyphloss, _ = compute_gtok_loss(
        reconstructed, target, vq_loss_info,
        glyphloss_fn=_dummy_glyphloss,
        weights=GtokLossWeights(glyphloss=1.0),
    )

    assert total_with_glyphloss.item() > total_no_glyphloss.item()
