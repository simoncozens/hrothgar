"""Tests for AR visual-pretraining loss utilities."""

import pytest
import torch
from hrothgar.ar.losses import ARLossWeights, compute_ar_loss
from hrothgar.ar.model import ARModelOutput


def _dummy_output() -> ARModelOutput:
    logits = torch.tensor(
        [
            [[3.0, 1.0, -2.0], [0.0, 2.0, 1.0]],
            [[1.0, 0.5, 0.0], [2.0, 0.0, -1.0]],
        ],
        dtype=torch.float32,
    )
    reconstructed = torch.zeros((2, 3, 4, 4), dtype=torch.float32)
    soft_embeddings = torch.zeros((2, 2, 8), dtype=torch.float32)
    targets = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)
    return ARModelOutput(
        logits=logits,
        reconstructed_images=reconstructed,
        soft_token_embeddings=soft_embeddings,
        target_token_indices=targets,
    )


def test_compute_ar_loss_uses_output_targets() -> None:
    output = _dummy_output()
    target_images = torch.ones((2, 3, 4, 4), dtype=torch.float32)

    total, terms = compute_ar_loss(output, target_images)

    assert total.ndim == 0
    assert "token_cross_entropy" in terms
    assert "pixel_l1" in terms
    assert "token_accuracy" in terms
    assert torch.isclose(terms["pixel_l1"], torch.tensor(1.0))


def test_compute_ar_loss_with_explicit_targets_and_weights() -> None:
    output = _dummy_output()
    target_images = torch.ones((2, 3, 4, 4), dtype=torch.float32)
    explicit_targets = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    weights = ARLossWeights(token_cross_entropy=2.0, pixel_l1=0.5)

    total, terms = compute_ar_loss(
        output,
        target_images,
        target_token_indices=explicit_targets,
        weights=weights,
    )

    expected = terms["weighted_token_cross_entropy"] + terms["weighted_pixel_l1"]
    assert torch.isclose(total, expected)
    assert torch.isclose(terms["weighted_pixel_l1"], 0.5 * terms["pixel_l1"])


def test_compute_ar_loss_requires_targets() -> None:
    output = _dummy_output()
    output = ARModelOutput(
        logits=output.logits,
        reconstructed_images=output.reconstructed_images,
        soft_token_embeddings=output.soft_token_embeddings,
        target_token_indices=None,
    )
    target_images = torch.ones((2, 3, 4, 4), dtype=torch.float32)

    with pytest.raises(ValueError):
        compute_ar_loss(output, target_images)


def test_compute_ar_loss_validates_shapes() -> None:
    output = _dummy_output()
    target_images = torch.ones((2, 3, 4, 4), dtype=torch.float32)
    bad_targets = torch.tensor([[0, 1, 2]], dtype=torch.long)

    with pytest.raises(ValueError):
        compute_ar_loss(output, target_images, target_token_indices=bad_targets)
