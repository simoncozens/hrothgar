"""Regression tests for upscaler corruption, corner detection, and BCE stability."""

import torch

from hrothgar.upscaler.dataset import UpscalerDatasetMaker
from hrothgar.upscaler.train import compute_upscaler_loss


# ---------------------------------------------------------------------------
# Low-res noise
# ---------------------------------------------------------------------------

def test_low_res_noise_injection_is_monochrome() -> None:
    maker = UpscalerDatasetMaker.__new__(UpscalerDatasetMaker)
    maker.low_res_noise_std = 1.0

    low_res = torch.zeros((2, 3, 8, 8), dtype=torch.float32)
    noised = maker._inject_monochrome_low_res_noise(low_res)

    assert torch.allclose(noised[:, 0, :, :], noised[:, 1, :, :])
    assert torch.allclose(noised[:, 1, :, :], noised[:, 2, :, :])
    assert torch.all((noised >= 0.0) & (noised <= 1.0))


# ---------------------------------------------------------------------------
# BCE stability
# ---------------------------------------------------------------------------

def test_compute_upscaler_loss_handles_invalid_prediction_values() -> None:
    predictions = torch.full((1, 3, 8, 8), 0.5, dtype=torch.float32)
    predictions[0, 0, 0, 0] = float("nan")
    predictions[0, 1, 0, 1] = 1.5
    predictions[0, 2, 0, 2] = -0.5

    targets = torch.zeros_like(predictions)

    loss, terms = compute_upscaler_loss(
        predictions=predictions,
        targets=targets,
        glyphloss_weight=1.0,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(terms["bce"])
    assert torch.isfinite(terms["glyphloss"])


# ---------------------------------------------------------------------------
# Gaussian blur
# ---------------------------------------------------------------------------

def test_gaussian_blur_preserves_shape_and_range() -> None:
    x = torch.rand((2, 3, 32, 32), dtype=torch.float32)
    blurred = UpscalerDatasetMaker._gaussian_blur(x, sigma=2.0)
    assert blurred.shape == x.shape
    assert torch.all((blurred >= 0.0) & (blurred <= 1.0))


def test_gaussian_blur_small_sigma_is_noop() -> None:
    x = torch.rand((1, 3, 16, 16), dtype=torch.float32)
    blurred = UpscalerDatasetMaker._gaussian_blur(x, sigma=0.3)
    # sigma < 0.5 → identity
    assert torch.allclose(blurred, x)


def test_gaussian_blur_reduces_variance() -> None:
    torch.manual_seed(42)
    x = torch.rand((1, 1, 64, 64), dtype=torch.float32)
    blurred = UpscalerDatasetMaker._gaussian_blur(x, sigma=4.0)
    assert blurred.var() < x.var()


# ---------------------------------------------------------------------------
# Corner response
# ---------------------------------------------------------------------------

def _make_edge_image() -> torch.Tensor:
    """Create a sharp vertical edge in a 64×64 image."""
    img = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    img[:, :, :, :32] = 0.0  # left: black
    img[:, :, :, 32:] = 1.0  # right: white
    return img


def _make_corner_image() -> torch.Tensor:
    """Create an L-shaped corner in a 64×64 image."""
    img = torch.ones((1, 3, 64, 64), dtype=torch.float32)  # all white
    img[:, :, 32:, 32:] = 0.0  # bottom-right quadrant: black
    return img


def test_corner_response_is_higher_at_corners() -> None:
    """Corners should produce a stronger Harris response than straight edges."""
    maker = UpscalerDatasetMaker.__new__(UpscalerDatasetMaker)
    maker.harris_window = 5

    edge_img = _make_edge_image()
    corner_img = _make_corner_image()

    edge_response = maker._corner_response(edge_img)
    corner_response = maker._corner_response(corner_img)

    # The maximum response should be higher for the corner image
    assert corner_response.amax() > edge_response.amax()


def test_corner_response_is_in_range() -> None:
    maker = UpscalerDatasetMaker.__new__(UpscalerDatasetMaker)
    maker.harris_window = 5

    img = torch.rand((2, 3, 32, 32), dtype=torch.float32)
    R = maker._corner_response(img)

    assert R.shape == (2, 32, 32)
    assert torch.all((R >= 0.0) & (R <= 1.0))


def test_corner_response_uniform_image_is_zero() -> None:
    """A flat image has no corners, so the normalised response should be ~0."""
    maker = UpscalerDatasetMaker.__new__(UpscalerDatasetMaker)
    maker.harris_window = 5

    # Uniform grey
    img = torch.full((1, 3, 32, 32), 0.5, dtype=torch.float32)
    R = maker._corner_response(img)

    # With no variation the response is a constant (the normalisation
    # divides by (max-min+eps), so all values end up at 0).
    assert torch.allclose(R, torch.tensor(0.0), atol=1e-6)


# ---------------------------------------------------------------------------
# Curvature-weighted corruption
# ---------------------------------------------------------------------------

def test_corrupt_high_res_preserves_shape_and_range() -> None:
    maker = UpscalerDatasetMaker.__new__(UpscalerDatasetMaker)
    maker.outline_noise_edge_threshold = 0.12
    maker.outline_noise_std = 0.05
    maker.terminal_blur_sigma = 2.5
    maker.stem_blur_sigma = 0.75
    maker.blur_mix_min = 0.2
    maker.blur_mix_max = 0.8
    maker.blur_sigma_jitter = 0.0
    maker.mix_spatial_noise = 0.0
    maker.harris_window = 5

    x = torch.rand((2, 3, 64, 64), dtype=torch.float32)
    corrupted = maker._corrupt_high_res_for_conformance(x)

    assert corrupted.shape == x.shape
    assert torch.all((corrupted >= 0.0) & (corrupted <= 1.0))


def test_corrupt_high_res_with_corner_image_blurs_corner_more() -> None:
    """On an image with both a straight edge and a corner, the corner region
    should receive a stronger blur (more deviation from the original)."""
    maker = UpscalerDatasetMaker.__new__(UpscalerDatasetMaker)
    maker.outline_noise_edge_threshold = 0.05
    maker.outline_noise_std = 0.0  # disable additive noise for this test
    maker.terminal_blur_sigma = 3.5
    maker.stem_blur_sigma = 0.5
    maker.blur_mix_min = 0.8  # high mix to amplify the difference
    maker.blur_mix_max = 0.8  # fixed mix to make test deterministic
    maker.blur_sigma_jitter = 0.0  # deterministic
    maker.mix_spatial_noise = 0.0  # deterministic
    maker.harris_window = 5

    img = _make_corner_image()  # (1, 3, 64, 64)
    corrupted = maker._corrupt_high_res_for_conformance(img)

    diff = (corrupted - img).abs()

    # The corner region (bottom-right of top-left quadrant, near (32,32))
    # should have larger differences than the straight-edge region
    # (left side of the image, far from the corner).
    corner_region_diff = diff[:, :, 28:36, 28:36].mean()
    edge_region_diff = diff[:, :, 28:36, 8:16].mean()

    assert corner_region_diff > edge_region_diff, (
        f"Expected corner region ({corner_region_diff:.6f}) to be blurred "
        f"more than edge region ({edge_region_diff:.6f})"
    )


def test_sigma_jitter_produces_different_blur_strength() -> None:
    """With jitter > 0, repeated calls on the same image should produce
    different corruption results because the effective sigma changes."""
    maker = UpscalerDatasetMaker.__new__(UpscalerDatasetMaker)
    maker.outline_noise_edge_threshold = 0.05
    maker.outline_noise_std = 0.0
    maker.terminal_blur_sigma = 3.0
    maker.stem_blur_sigma = 0.5
    maker.blur_mix_min = 0.8
    maker.blur_mix_max = 0.8
    maker.blur_sigma_jitter = 0.5  # ±50% jitter
    maker.mix_spatial_noise = 0.0
    maker.harris_window = 5

    torch.manual_seed(42)
    img = _make_corner_image()

    # Run twice with different seeds
    torch.manual_seed(42)
    corrupted_a = maker._corrupt_high_res_for_conformance(img.clone())
    torch.manual_seed(99)
    corrupted_b = maker._corrupt_high_res_for_conformance(img.clone())

    # The two corruptions should differ (different sigma → different blur)
    assert not torch.allclose(corrupted_a, corrupted_b, atol=1e-4), (
        "Sigma jitter should produce different corruption across calls"
    )


def test_spatial_noise_creates_non_uniform_corruption() -> None:
    """With mix_spatial_noise > 0, different edge pixels within the same
    sample should receive different blur mix strengths."""
    maker = UpscalerDatasetMaker.__new__(UpscalerDatasetMaker)
    maker.outline_noise_edge_threshold = 0.05
    maker.outline_noise_std = 0.0
    maker.terminal_blur_sigma = 3.0
    maker.stem_blur_sigma = 0.5
    maker.blur_mix_min = 0.5
    maker.blur_mix_max = 0.5  # fix scalar mix to isolate spatial noise
    maker.blur_sigma_jitter = 0.0
    maker.mix_spatial_noise = 0.3  # significant spatial noise
    maker.harris_window = 5

    img = _make_edge_image()  # vertical edge — all edge pixels are similar
    corrupted = maker._corrupt_high_res_for_conformance(img)

    diff = (corrupted - img).abs()

    # Compute per-pixel difference along the edge band (column ~32)
    edge_col = diff[:, :, :, 31:33].mean(dim=(0, 1))  # (H, 2) → mean over batch, channel, width
    std_along_edge = edge_col.std()

    assert std_along_edge > 0.0, (
        "Spatial noise should create variation in corruption strength "
        f"along the edge, got std={std_along_edge:.6f}"
    )
