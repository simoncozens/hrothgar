"""fontquant measurements for a font (importable without torch).

``quantify_row(font)`` is the one entry point everything else uses: it returns
one font's measurements as the ``{column/path: str}`` row shape ``fontquant.csv``
has.  It looks the font up by filename in the measured library
(``data/similarity_supervision/fontquant.csv``, or ``$FONTQUANT_CSV``) and only
runs ``fontquant.quantify`` when that file, or the font's row, is missing — so
the CSV is a cache over an expensive (~1 s/font) measurement, never a hard
requirement.

Why a CSV at all?  The measurements carry things ``METADATA.pb`` does not:
whether a font's *lowercase slots actually hold lowercase shapes*, whether it is
unicase, whether its ``a``/``g`` are single- or double-storey, stroke contrast,
proportions, the parametric axes.  Measuring the whole library once and shipping
the result is much cheaper than re-measuring it on every run; fonts the file has
never seen (a query TTF) are measured on the spot.

:class:`FontQuantTable` is the read-only view the content detector uses: it needs
only the casing/a/g columns, to drop fonts with no real lowercase shapes
(an all-caps font puts an uppercase shape in every lowercase slot, so training
"``E`` is an ``e``" on it teaches nonsense) and blank fonts (Adobe Blank puts an
empty image in every slot).
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# ``casing/lowercase_shapes`` values are "lowercase" (normal), "uppercase"
# (all-caps design) and "smallcaps".
LOWERCASE_SHAPES = "lowercase"


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# ── Measuring a font with the fontquant package ──────────────────────────────
#: Columns in ``fontquant.csv`` that are bookkeeping, not measurements.
_CSV_BOOKKEEPING = {"Font", "Location"}


def default_fontquant_csv() -> Path:
    """``$FONTQUANT_CSV``, else the measured library in ``data/similarity_supervision``."""
    override = os.environ.get("FONTQUANT_CSV")
    if override:
        return Path(override)
    # Lib/hrothgar/fontquant.py -> repo root
    return (
        Path(__file__).resolve().parents[2]
        / "data"
        / "similarity_supervision"
        / "fontquant.csv"
    )


def _flatten(result: dict, prefix: str = "") -> dict:
    """Flatten ``quantify``'s nested ``{category: {key: {"value": X}}}`` dict."""
    flat: dict = {}
    for key, value in result.items():
        column = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict) and "value" in value:
            flat[column] = value["value"]
        elif isinstance(value, dict):
            flat.update(_flatten(value, column))
        else:
            flat[column] = value
    return flat


def _stringify(flat: dict) -> dict[str, str]:
    """Render the flattened values the way ``fontquant.csv`` writes them."""
    row: dict[str, str] = {}
    for column, value in flat.items():
        if isinstance(value, bool):
            row[column] = "true" if value else "false"
        elif isinstance(value, (list, tuple)):
            row[column] = ",".join(str(v) for v in value)
        elif value is None:
            row[column] = ""
        else:
            row[column] = str(value)
    return row


@lru_cache(maxsize=4)
def _csv_table(csv_path: str, mtime: float) -> dict[str, dict[str, str]]:
    """The measured library, keyed by lowercase font basename.

    Variable fonts have one row per axis location; the default-instance row
    (empty ``Location``) wins, matching what ``quantify`` reports for the file.
    """
    rows: dict[str, dict[str, str]] = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("Font", "")
            if not name:
                continue
            key = Path(name).name.lower()
            if key not in rows or row.get("Location", "") == "":
                rows[key] = row
    return rows


def _csv_row(font_path: Path) -> dict[str, str] | None:
    """The font's row from the measured library, if the file and row exist."""
    csv_path = default_fontquant_csv()
    try:
        mtime = csv_path.stat().st_mtime
    except OSError:
        return None
    return _csv_table(str(csv_path), mtime).get(font_path.name.lower())


@lru_cache(maxsize=None)
def _cached_quantify(path: str, mtime: float) -> dict[str, str]:
    # Imported lazily: ``fontquant`` pulls in a lot, and most imports of this
    # module only want the CSV table.
    from fontquant import quantify

    try:
        return _stringify(_flatten(quantify(path)))
    except Exception:
        # A font fontquant cannot measure is treated exactly like one missing
        # from the CSV: no data (rather than killing a library-wide scan).
        return {}


def quantify_row(font_or_path) -> dict[str, str]:
    """One font's fontquant measurements, as a CSV-shaped ``{column: str}`` row.

    Looked up by filename in the measured library
    (``data/similarity_supervision/fontquant.csv``, or ``$FONTQUANT_CSV``).  Only
    when that file is missing, or it has no row for this font, is the font
    measured with ``fontquant.quantify`` — measuring costs about a second, so the
    CSV is the fast path and live measurement the fallback.

    Accepts a font object (anything with a ``.path``) or a path.
    """
    path = Path(getattr(font_or_path, "path", font_or_path)).resolve()
    cached = _csv_row(path)
    if cached is not None:
        return {k: v for k, v in cached.items() if k not in _CSV_BOOKKEEPING}
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return dict(_cached_quantify(str(path), mtime))


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
    """Lookup of :class:`FontQuant` by font filename (basename, case-insensitive).

    The CSV is the fast path; a font with no row (or a missing CSV entirely) is
    measured with :func:`quantify_row` on first lookup and remembered.
    """

    def __init__(self, rows: dict[str, FontQuant], live_fallback: bool = True):
        self._by_basename = rows
        self._live_fallback = live_fallback
        self._measured: dict[str, FontQuant | None] = {}

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
        return default_fontquant_csv()

    @classmethod
    def load_default(cls) -> FontQuantTable:
        return cls.load(cls.default_path())

    def get(self, font_path) -> FontQuant | None:
        key = Path(str(font_path)).name.lower()
        quant = self._by_basename.get(key)
        if quant is not None or not self._live_fallback:
            return quant
        if key not in self._measured:
            self._measured[key] = _quant_from_row(quantify_row(font_path))
        return self._measured[key]

    def __len__(self) -> int:
        return len(self._by_basename)

    def has_lowercase(self, font_path) -> bool:
        """Whether a font's lowercase slots are usable (unknown fonts: assume yes)."""
        quant = self.get(font_path)
        return True if quant is None else quant.has_lowercase

    def is_blank(self, font_path) -> bool:
        quant = self.get(font_path)
        return bool(quant is not None and quant.is_blank)


def _quant_from_row(row: dict[str, str]) -> FontQuant | None:
    """Build a :class:`FontQuant` from a CSV-shaped row (``None`` when empty)."""
    if not row:
        return None
    return FontQuant(
        lowercase_shapes=row.get("casing/lowercase_shapes", ""),
        unicase=row.get("casing/unicase", "").strip().lower() == "true",
        lowercase_a_style=row.get("appearance/lowercase_a_style", ""),
        lowercase_g_style=row.get("appearance/lowercase_g_style", ""),
        ascender=_to_float(row.get("appearance/ascender", "")),
        x_height=_to_float(row.get("appearance/x_height", "")),
    )


def load_default_or_none() -> FontQuantTable | None:
    """Load the measured library; when it is absent, measure fonts on demand."""
    try:
        return FontQuantTable.load_default()
    except OSError:
        print(
            "fontquant.csv not found; measuring fonts as they are needed "
            "(slower, same results)"
        )
        return FontQuantTable({})


__all__ = [
    "FontQuant",
    "FontQuantTable",
    "LOWERCASE_SHAPES",
    "default_fontquant_csv",
    "load_default_or_none",
    "quantify_row",
]
