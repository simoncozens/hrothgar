import os

from hrothgar.dataset import DatasetMaker

if "GOOGLE_FONTS_REPO" not in os.environ:
    raise ValueError("GOOGLE_FONTS_REPO environment variable not set, cannot run tests")
REPOSITORY_PATH = os.getenv("GOOGLE_FONTS_REPO")


def test_dataset_maker():
    maker = DatasetMaker(REPOSITORY_PATH, batch_size=32)
    train = maker.train_set()
    test = maker.test_set()

    # Test lengths are reasonable
    assert len(train) > 100_000
    assert len(test) > 2_000


def test_data_loader():
    maker = DatasetMaker(REPOSITORY_PATH, batch_size=32)
    test_loader = maker.test_loader()
    # Check we can read a few batches
    for _ in range(3):
        batch = next(iter(test_loader))
        assert "char" in batch
        assert "rendering" in batch
        assert "description" in batch

        assert batch["rendering"].shape == (32, 3, 64, 64)
        assert batch["char"].shape == (32,)
