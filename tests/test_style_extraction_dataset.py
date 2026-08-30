"""Tests for the style-extraction font-level dataset (canary precompute)."""

import torch

from hrothgar.style_extraction import dataset as se_dataset
from hrothgar.style_extraction.dataset import StyleExtractionDatasetMaker

REPO = "tests/dummy_repo"
# Printable ASCII, which both dummy fonts cover.
ASCII = list(range(0x20, 0x7F))


def _make_maker(**kwargs):
    defaults = dict(
        repo_url=REPO,
        batch_size=2,
        image_size=32,
        character_set=ASCII,
        num_evidence_glyphs=2,
        split_seed=1234,
        canary_size=2,
    )
    defaults.update(kwargs)
    return StyleExtractionDatasetMaker(**defaults)


def _assert_batches_equal(a: dict, b: dict) -> None:
    for key in (
        "style_images",
        "style_codepoint_idx",
        "target_images",
        "target_codepoint_idx",
        "target_codepoint",
    ):
        assert torch.equal(a[key], b[key]), f"batch tensor {key} differs"
    assert a["family"] == b["family"]


def test_canary_loader_is_precomputed_and_fixed() -> None:
    """Canary batches are identical on every iteration (same fonts + glyphs)."""
    maker = _make_maker()
    loader = maker.train_loader()
    assert len(loader) == 1

    first = list(loader)
    second = list(loader)
    assert len(first) == len(second) == 1
    _assert_batches_equal(first[0], second[0])

    # The batch is shaped (batch_size, num_evidence_glyphs, 1, 32, 32).
    assert first[0]["style_images"].shape == (2, 2, 1, 32, 32)
    assert first[0]["target_images"].shape == (2, 1, 32, 32)


def test_canary_test_loader_is_train_loader() -> None:
    """In canary mode test == train, so the precompute is shared."""
    maker = _make_maker()
    assert maker.test_loader() is maker.train_loader()


def test_canary_loader_is_deterministic_across_runs() -> None:
    """Same split_seed => bit-identical canary data across fresh makers."""
    loader_a = _make_maker(split_seed=1234).train_loader()
    loader_b = _make_maker(split_seed=1234).train_loader()
    _assert_batches_equal(next(iter(loader_a)), next(iter(loader_b)))

    # A different seed => different (but valid) batches.
    loader_c = _make_maker(split_seed=99).train_loader()
    c = next(iter(loader_c))
    assert c["style_images"].shape == (2, 2, 1, 32, 32)


def test_non_canary_loader_renders_valid_batch(monkeypatch) -> None:
    """The regular (collate-based) path still works after the refactor."""
    monkeypatch.setattr(se_dataset, "NUM_WORKERS", 0)
    maker = _make_maker(canary_size=None)
    loader = maker.train_loader()
    assert len(loader) > 0
    batch = next(iter(loader))
    assert batch["style_images"].shape[0] == maker.batch_size
    assert batch["style_images"].shape[1] == maker._num_evidence_glyphs
    assert batch["target_images"].shape[0] == maker.batch_size
    assert set(batch["family"]).issubset({f.family for f in maker.train_fonts})
