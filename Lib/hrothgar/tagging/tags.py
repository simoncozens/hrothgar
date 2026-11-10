"""Instance-aware loader for ``tags/all/families.csv``.

The tag CSV is long-form: ``family, instance, tag, value``.  ``instance`` is
empty for family-level tags, or an axis location such as ``wght@900`` /
``wdth,wght@100,900`` for per-instance tags (assessed separately because the
tag *changes* with weight/width).  ``value`` is the assessed centile; ``1`` is
the lowest assessed value and a missing row means *unassessed*, never zero — so
we drop the handful of stray ``0`` values and keep absence as absence.

This module turns the CSV into per-``(family, instance)`` *cells*, and maps an
instance string to an fvar-ordered ``axis_position`` for a given font file.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from hrothgar.dataset import fvar_axes


@dataclass
class TagCell:
    """One ``(family, instance)`` cell and its assessed tag values."""

    family: str
    instance: str  # "" = family-level (default instance)
    values: dict[str, float] = field(default_factory=dict)


def parse_instance(instance: str) -> dict[str, float]:
    """Parse ``"wght@400"`` / ``"wdth,wght@100,900"`` into ``{axis: value}``."""
    if not instance:
        return {}
    left, sep, right = instance.partition("@")
    if not sep:
        return {}
    axes = [a for a in left.split(",") if a]
    values = [v for v in right.split(",") if v]
    if len(axes) != len(values):
        raise ValueError(
            f"malformed instance {instance!r}: {len(axes)} axes vs {len(values)} values"
        )
    return {a: float(v) for a, v in zip(axes, values)}


def is_italic_instance(instance: str) -> bool:
    """Whether the instance location selects the italic file (``ital`` nonzero)."""
    return parse_instance(instance).get("ital", 0) != 0


def resolve_axis_position(instance: str, font_path: Path) -> list[float]:
    """Resolve an instance string to an fvar-ordered axis position for ``font_path``.

    Unspecified axes fall back to their fvar default; the ``ital`` axis is
    *not* an fvar axis here (it selects the italic *file*, see
    :func:`is_italic_instance`) so it is dropped.  An empty instance (the
    default/regular location) yields ``[]``, which :func:`render` treats as the
    default instance.
    """
    parsed = parse_instance(instance)
    if not parsed:
        return []
    parsed = {k: v for k, v in parsed.items() if k != "ital"}
    axes = fvar_axes(Path(font_path))
    if not axes:
        return []
    return [parsed.get(tag, default) for tag, _min, default, _max in axes]


def load_tag_cells(repo_path: str | Path) -> list[TagCell]:
    """Read ``tags/all/families.csv`` into per-``(family, instance)`` cells."""
    tag_csv = Path(repo_path) / "tags" / "all" / "families.csv"
    cells: dict[tuple[str, str], dict[str, float]] = {}
    with tag_csv.open() as f:
        for family, instance, tag, value in csv.reader(f):
            v = float(value)
            if v == 0:
                # "0" is not a real assessment — 1 is the lowest assessed value.
                continue
            cells.setdefault((family, instance), {})[tag] = v
    return [TagCell(f, i, vals) for (f, i), vals in cells.items()]
