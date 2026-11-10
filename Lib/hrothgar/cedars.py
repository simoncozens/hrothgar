"""Hand-labelled CEDARS+ glyph features.

``cedars.csv`` holds one row per family of hand labels for the CEDARS+ glyph
features (``e-crossbar``, ``finials``, ``c-overhang``, ``a-shape``, …).  It is
used as *ground truth* when training the small glyph-feature recognisers, and as
the ``feature_source="hand"`` categorical axes in the text similarity arm.

The file lives with the similarity supervision data::

    data/similarity_supervision/cedars.csv

Set ``$CEDARS_CSV`` to point at a different copy.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

#: Values that mean "not assessed" rather than a real label.
MISSING_VALUES = {"", "na"}


def default_cedars_path() -> Path:
    """``$CEDARS_CSV``, else the copy shipped in ``data/similarity_supervision``."""
    override = os.environ.get("CEDARS_CSV")
    if override:
        return Path(override)
    # Lib/hrothgar/cedars.py -> repo root
    return (
        Path(__file__).resolve().parents[2]
        / "data"
        / "similarity_supervision"
        / "cedars.csv"
    )


def load_cedars(path: str | Path | None = None) -> dict[str, dict[str, str]]:
    """Load the hand labels keyed by lowercase family name."""
    path = default_cedars_path() if path is None else Path(path)
    out: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["family"].strip().lower()] = row
    return out


__all__ = ["MISSING_VALUES", "default_cedars_path", "load_cedars"]
