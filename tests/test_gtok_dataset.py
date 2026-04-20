import os

from hrothgar.gtok.dataset import GTokDatasetMaker

if "GOOGLE_FONTS_REPO" not in os.environ:
    raise ValueError("GOOGLE_FONTS_REPO environment variable not set, cannot run tests")
REPOSITORY_PATH = os.getenv("GOOGLE_FONTS_REPO")


def test_dataset_maker():
    maker = GTokDatasetMaker(REPOSITORY_PATH, batch_size=32)
    train = maker.train_set()
    test = maker.test_set()

    # Test lengths are reasonable
    assert len(train) > 100_000
    assert len(test) > 2_000


def test_data_loader():
    maker = GTokDatasetMaker(REPOSITORY_PATH, batch_size=32)
    test_loader = maker.test_loader()
    # Check we can read a few batches
    for _ in range(3):
        batch = next(iter(test_loader))
        assert "gid" in batch
        assert "rendering" in batch
        assert "description" in batch

        assert batch["rendering"].shape == (32, 3, 128, 128)
        assert batch["gid"].shape == (32,)


def test_no_crash_on_space():
    maker = GTokDatasetMaker(REPOSITORY_PATH, batch_size=32)
    maker.test_fonts = [maker.test_fonts[0]]  # Use just one font to speed up the test
    test_loader = maker.test_loader()
    for batch in test_loader:
        assert "rendering" in batch
    # If we make it through all batches, then we know we didn't crash on anything with no outlines
