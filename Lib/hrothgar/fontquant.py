"""Per-font metadata from ``fontquant.csv`` (importable without torch).

``fontquant.csv`` is generated over the library (one row per font file, plus one
per axis position) and carries things ``METADATA.pb`` does not: whether a font's
*lowercase slots actually hold lowercase shapes*, whether it is unicase, and
whether its ``a``/``g`` are single- or double-storey.

The content detector needs the first of these: an all-caps font puts an uppercase
shape in every lowercase slot, so training the lowercase anchors on it teaches
"``E`` is an ``e``", and auditing it asks a question with no meaningful answer.
Blank fonts (Adobe Blank) put an empty image in every slot, which is worse still.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

# ``casing/lowercase_shapes`` values are "lowercase" (normal), "uppercase"
# (all-caps design) and "smallcaps".
LOWERCASE_SHAPES = "lowercase"


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


@dataclass(frozen=True)
class FontQuant:
    """One font's quantitative metadata."""

    lowercase_shapes: str
    unicase: bool
    lowercase_a_style: str  # "single_story" | "double_story" | ""
    lowercase_g_style: str
    ascender: float
    x_height: float

    @property
    def has_lowercase(self) -> bool:
        """Whether the lowercase slots hold ordinary lowercase shapes."""
        return self.lowercase_shapes == LOWERCASE_SHAPES

    @property
    def is_blank(self) -> bool:
        """A font with no vertical extent at all (Adobe Blank and friends)."""
        return self.ascender == 0.0 and self.x_height == 0.0


class FontQuantTable:
    """Lookup of :class:`FontQuant` by font filename (basename, case-insensitive)."""

    def __init__(self, rows: dict[str, FontQuant]):
        self._by_basename = rows

    @classmethod
    def load(cls, path) -> FontQuantTable:
        rows: dict[str, FontQuant] = {}
        with Path(path).open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                font = row.get("Font", "")
                if not font:
                    continue
                rows[Path(font).name.lower()] = FontQuant(
                    lowercase_shapes=row.get("casing/lowercase_shapes", ""),
                    unicase=row.get("casing/unicase", "").strip().lower() == "true",
                    lowercase_a_style=row.get("appearance/lowercase_a_style", ""),
                    lowercase_g_style=row.get("appearance/lowercase_g_style", ""),
                    ascender=_to_float(row.get("appearance/ascender", "")),
                    x_height=_to_float(row.get("appearance/x_height", "")),
                )
        return cls(rows)

    @classmethod
    def default_path(cls) -> Path:
        override = os.environ.get("FONTQUANT_CSV")
        if override:
            return Path(override)
        # Lib/hrothgar/fontquant.py -> repo root
        return Path(__file__).resolve().parents[2] / "fontquant.csv"

    @classmethod
    def load_default(cls) -> FontQuantTable:
        return cls.load(cls.default_path())

    def get(self, font_path) -> FontQuant | None:
        return self._by_basename.get(Path(str(font_path)).name.lower())

    def __len__(self) -> int:
        return len(self._by_basename)

    def has_lowercase(self, font_path) -> bool:
        """Whether a font's lowercase slots are usable (unknown fonts: assume yes)."""
        quant = self.get(font_path)
        return True if quant is None else quant.has_lowercase

    def is_blank(self, font_path) -> bool:
        quant = self.get(font_path)
        return bool(quant is not None and quant.is_blank)


def load_default_or_none() -> FontQuantTable | None:
    """Load the default table, or ``None`` (with a warning) when it is missing."""
    try:
        return FontQuantTable.load_default()
    except OSError:
        print(
            "warning: fontquant.csv not found; lowercase-shape / blank filtering off"
        )
        return None


__all__ = [
    "FontQuant",
    "FontQuantTable",
    "LOWERCASE_SHAPES",
    "load_default_or_none",
]
