"""Integration tests for the AR phase-1 dataset maker.

Requires the Google Fonts repository to be available; set the
``GOOGLE_FONTS_REPO`` environment variable to its local path before running.
"""

import os

from hrothgar.ar.dataset import ARPhase1DatasetMaker

if "GOOGLE_FONTS_REPO" not in os.environ:
    raise ValueError("GOOGLE_FONTS_REPO environment variable not set, cannot run tests")
REPOSITORY_PATH = os.getenv("GOOGLE_FONTS_REPO")


def test_dataset_maker() -> None:
    """AR dataset train/test sets have plausible sizes."""
    maker = ARPhase1DatasetMaker(REPOSITORY_PATH, batch_size=32)
    train = maker.train_set()
    test = maker.test_set()
    assert len(train) > 100_000
    assert len(test) > 2_000


def test_data_loader_shapes() -> None:
    """Loader returns batches with the expected keys and tensor shapes."""
    style_glyph_count = 4
    maker = ARPhase1DatasetMaker(
        REPOSITORY_PATH, batch_size=8, style_glyph_count=style_glyph_count
    )
    batch = next(iter(maker.test_loader()))

    B = 8
    H = W = 128

    assert batch["char"].shape == (B,)
    assert batch["target_rendering"].shape == (B, 3, H, W)
    assert batch["content_rendering"].shape == (B, 3, H, W)
    assert batch["style_renderings"].shape == (B, style_glyph_count, 3, H, W)
    assert batch["style_chars"].shape == (B, style_glyph_count)
    assert len(batch["description"]) == B


def test_content_rendering_differs_from_target() -> None:
    """Content rendering should come from the reference font, not the target font.

    Across a reasonably-sized batch, at least some items should have a
    content rendering that differs from the target rendering, confirming that
    ``reference_font()`` is being called rather than re-rendering from the
    target font.
    """
    maker = ARPhase1DatasetMaker(REPOSITORY_PATH, batch_size=16)
    batch = next(iter(maker.test_loader()))
    same = (batch["content_rendering"] == batch["target_rendering"]).all(dim=(1, 2, 3))
    assert not same.all(), (
        "Every content rendering was pixel-identical to its target rendering; "
        "reference_font() may not be applied correctly."
    )


def test_no_crash_on_full_loader() -> None:
    """Loader completes several batches without error on a small font subset."""
    maker = ARPhase1DatasetMaker(REPOSITORY_PATH, batch_size=32, canary_size=5)
    for _ in range(3):
        batch = next(iter(maker.test_loader()))
        assert "target_rendering" in batch
