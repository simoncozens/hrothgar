"""Tests for shared dataset utilities (``hrothgar.dataset``)."""

import pytest
from hrothgar.dataset import (
    ClassBalancedBatchSampler,
    Instance,
    Unit,
    _regular_instance,
    contrast_instances,
)


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
        items,
        key=lambda i: "even" if i % 2 == 0 else "odd",
        batch_size=3,
        drop_last=True,
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


def _unit(*instances: Instance) -> Unit:
    return Unit(
        family="Test",
        style="normal",
        bucket="sans",
        instances=list(instances),
        classification="SANS_SERIF",
        tag_category="SANS_SERIF",
    )


def _static(weight: int) -> Instance:
    return Instance(
        path=f"Test-{weight}.ttf",
        weight=weight,
        style="normal",
        variable=False,
        axes=None,
        axis_position=None,
        has_target=True,
        coverage=26,
    )


def _variable(wght_min: float, wght_default: float, wght_max: float) -> Instance:
    return Instance(
        path="Test[wght].ttf",
        weight=400,
        style="normal",
        variable=True,
        axes=[["wght", wght_min, wght_default, wght_max]],
        axis_position=None,
        has_target=True,
        coverage=26,
    )


def test_regular_instance_keeps_single_static_family() -> None:
    """A single-static-font family contributes its one file, whatever its weight."""
    inst = _regular_instance(_unit(_static(700)))
    assert inst is not None
    assert inst.weight == 700
    assert inst.variable is False


def test_regular_instance_prefers_static_400() -> None:
    inst = _regular_instance(_unit(_static(300), _static(400), _static(700)))
    assert inst is not None
    assert inst.weight == 400


def test_regular_instance_variable_origin_at_100_is_instantiated_at_400() -> None:
    """A VF whose wght origin/default is 100 must be instantiated at 400, not 100."""
    inst = _regular_instance(_unit(_variable(100.0, 100.0, 900.0)))
    assert inst is not None
    assert inst.weight == 400
    assert inst.axis_position == [400.0]


def test_regular_instance_variable_is_a_fixed_point() -> None:
    """The synthesized regular location carries no wght axis, so the contrast
    logic cannot re-expand it into a weight ladder."""
    inst = _regular_instance(_unit(_variable(200.0, 200.0, 900.0)))
    assert inst is not None
    assert inst.variable is False
    assert inst.axes is None
    assert inst.axis_position == [400.0]


def test_regular_instance_variable_not_reexpanded_by_contrast() -> None:
    """Regression: contrast_instances must not re-synthesize weights from a
    trimmed regular-only unit (previously it produced 200/550/900)."""
    inst = _regular_instance(_unit(_variable(200.0, 200.0, 900.0)))
    assert inst is not None
    trimmed = _unit(inst)
    assert trimmed.weight_range() == (400, 400)
    for k in (1, 2, 3):
        chosen = contrast_instances(trimmed, k)
        assert [c.weight for c in chosen] == [400]
        assert [c.axis_position for c in chosen] == [[400.0]]


def test_regular_instance_variable_default_400_is_instantiated_at_400() -> None:
    inst = _regular_instance(_unit(_variable(100.0, 400.0, 900.0)))
    assert inst is not None
    assert inst.weight == 400
    assert inst.axis_position == [400.0]


def test_regular_instance_drops_multi_static_without_400() -> None:
    """A multi-weight static family with no 400 has no Regular master."""
    assert _regular_instance(_unit(_static(300), _static(700))) is None
