"""Tests for shared dataset utilities (``hrothgar.dataset``)."""

import pytest

from hrothgar.dataset import ClassBalancedBatchSampler


def test_class_balanced_batch_sampler_balances_majority_class() -> None:
    """A 20:1 majority class must not dominate any emitted batch."""
    items = ["sans"] * 100 + ["serif"] * 5
    sampler = ClassBalancedBatchSampler(
        items, key=lambda item: item, batch_size=8, drop_last=True
    )

    batches = list(sampler)
    assert batches
    for batch in batches:
        labels = [items[i] for i in batch]
        assert labels.count("sans") == 4
        assert labels.count("serif") == 4


def test_class_balanced_batch_sampler_more_classes_than_slots() -> None:
    """When classes outnumber batch slots, each batch samples distinct classes."""
    items = [f"cls-{i}" for i in range(10)]
    sampler = ClassBalancedBatchSampler(
        items, key=lambda item: item, batch_size=4, drop_last=False
    )

    batches = list(sampler)
    assert batches
    for batch in batches:
        labels = [items[i] for i in batch]
        assert len(labels) == 4
        assert len(set(labels)) == 4


def test_class_balanced_batch_sampler_len() -> None:
    items = list(range(10))
    sampler = ClassBalancedBatchSampler(
        items, key=lambda i: "even" if i % 2 == 0 else "odd",
        batch_size=3, drop_last=True,
    )
    assert len(sampler) == 3  # 10 // 3


def test_class_balanced_batch_sampler_rejects_empty() -> None:
    with pytest.raises(ValueError):
        ClassBalancedBatchSampler(
            [], key=lambda item: item, batch_size=4, drop_last=True
        )


def test_class_balanced_batch_sampler_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError):
        ClassBalancedBatchSampler(
            [1, 2, 3], key=lambda item: item, batch_size=0, drop_last=True
        )


def test_class_balanced_batch_sampler_rng_is_deterministic() -> None:
    """A seeded RNG makes batch composition reproducible (canary mode)."""
    import random

    items = [f"cls-{i % 3}" for i in range(24)]
    key = lambda item: item  # noqa: E731
    sampler_a = ClassBalancedBatchSampler(
        items, key=key, batch_size=6, drop_last=True, rng=random.Random(42)
    )
    sampler_b = ClassBalancedBatchSampler(
        items, key=key, batch_size=6, drop_last=True, rng=random.Random(42)
    )
    assert list(sampler_a) == list(sampler_b)

    # A different seed produces a different (but still balanced) composition.
    sampler_c = ClassBalancedBatchSampler(
        items, key=key, batch_size=6, drop_last=True, rng=random.Random(7)
    )
    assert list(sampler_a) != list(sampler_c)
