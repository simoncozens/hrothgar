#!/usr/bin/env python3
"""Resolve the ``Latin family`` column of ``non-latins.csv`` to canonical
Google Fonts family names.

The CSV's "Latin family" values are a mixture of:

* family names        ("Montserrat", "Noto Sans", "Baloo 2")
* full font names     ("Hind Light", "Roboto Bold", "IBM Plex Sans Thin")
* renamed/legacy names ("Source Sans Pro")
* explicit non-GF names ("AMS Euler (non GF)")

This script indexes every ``METADATA.pb`` in the Google Fonts ``ofl`` tree and
rewrites the CSV so that:

* ``Latin family``    is the canonical ``metadata.name`` (family name), or the
                      raw string if it could not be resolved.
* ``Latin full name`` is the matched font ``full_name`` (weight-preserving),
                      when the input looked like a full font name.
* ``Resolve status``  is one of ``ok`` / ``alias`` / ``non-gf`` /
                      ``unresolved`` / ``empty`` for auditing.

The original raw value is preserved in ``Latin family raw``.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

from gftools.util.google_fonts import Metadata

# As it appears in the source CSV (two spaces).
RAW_LATIN_FAMILY = "Latin  family"

# Legacy / renamed family names whose ``metadata.name`` no longer matches.
# Keyed by the lowercased, whitespace-collapsed input string.
MANUAL_ALIASES = {
    "source sans pro": "Source Sans 3",
    "andada": "Andada Pro",
}

_NON_GF_RE = re.compile(r"\(non[- ]gf\)", re.IGNORECASE)

# Trailing tokens that look like weight/style rather than part of a family name.
_WEIGHTISH = {
    "thin",
    "extralight",
    "ultralight",
    "light",
    "regular",
    "normal",
    "medium",
    "semibold",
    "demibold",
    "bold",
    "extrabold",
    "ultrabold",
    "black",
    "heavy",
    "italic",
    "oblique",
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _key(s: str) -> str:
    return _norm(s).lower()


def _token_key(tok: str) -> str:
    return re.sub(r"[^a-z0-9]", "", tok.lower())


def build_indexes(repo: Path) -> tuple[dict, dict, dict]:
    """Index families and font full names from the ``ofl`` METADATA.pb files.

    Returns ``(family_index, fullname_index, psname_index)`` where values are
    canonical family names (for the family index) or ``(family, full_name)``
    tuples (for the full-name / PostScript-name indexes).
    """
    base = repo / "ofl" if (repo / "ofl").is_dir() else repo

    family_index: dict[str, str] = {}
    fullname_index: dict[str, tuple[str, str]] = {}
    psname_index: dict[str, tuple[str, str]] = {}

    for pb in sorted(base.glob("*/METADATA.pb")):
        try:
            m = Metadata(str(pb))
        except Exception:
            continue
        family = m.name
        family_index[_key(family)] = family
        for font in m.fonts:
            full = font.full_name
            fullname_index[_key(full)] = (family, full)
            ps = font.post_script_name
            if ps:
                psname_index[_key(ps)] = (family, full)

    return family_index, fullname_index, psname_index


def strip_weightish(name: str) -> str:
    """Drop trailing weight/style tokens (e.g. "Hind Light" -> "Hind")."""
    parts = _norm(name).split()
    while parts and _token_key(parts[-1]) in _WEIGHTISH:
        parts.pop()
    return " ".join(parts)


def resolve(
    raw: str,
    family_index: dict,
    fullname_index: dict,
    psname_index: dict,
) -> tuple[str, str | None, str]:
    """Resolve one raw Latin-family string.

    Returns ``(family, full_name, status)``.
    """
    raw = raw or ""
    s = raw.strip()
    if not s:
        return "", None, "empty"

    if _NON_GF_RE.search(s):
        return s, None, "non-gf"

    k = _key(s)

    if k in fullname_index:
        family, full = fullname_index[k]
        return family, full, "ok"

    if k in family_index:
        return family_index[k], None, "ok"

    if k in psname_index:
        family, full = psname_index[k]
        return family, full, "ok"

    # Alias / family match on the weight-stripped form (e.g.
    # "Andada Bold" -> "Andada" -> alias "Andada Pro").
    base = strip_weightish(s)
    base_key = _key(base) if base else ""
    if base_key and base_key != k:
        if base_key in MANUAL_ALIASES:
            alias = MANUAL_ALIASES[base_key]
            if _key(alias) in family_index:
                return family_index[_key(alias)], None, "alias"
        if base_key in family_index:
            return family_index[base_key], None, "ok"

    # Alias on the raw form (e.g. "Source Sans Pro").
    if k in MANUAL_ALIASES:
        alias = MANUAL_ALIASES[k]
        if _key(alias) in family_index:
            return family_index[_key(alias)], None, "alias"

    return s, None, "unresolved"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts"),
        help="Google Fonts repo root, or the ofl/ directory directly",
    )
    parser.add_argument(
        "--csv",
        default="non-latins.csv",
        help="Input CSV (default: non-latins.csv)",
    )
    parser.add_argument(
        "--out",
        default="non-latins.resolved.csv",
        help="Output CSV (default: non-latins.resolved.csv)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input CSV instead of writing --out",
    )
    args = parser.parse_args()

    family_index, fullname_index, psname_index = build_indexes(Path(args.repo))

    in_path = Path(args.csv)
    out_path = in_path if args.inplace else Path(args.out)

    with in_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    # Locate the Latin-family column robustly (1 or 2 spaces).
    latin_key = None
    for name in fieldnames:
        if re.sub(r"\s+", "", name).lower() == "latinfamily":
            latin_key = name
            break
    if latin_key is None:
        latin_key = RAW_LATIN_FAMILY

    out_fields = [
        "Family",
        "Font",
        "Has Latin?",
        "Latin origin",
        "Latin family raw",
        "Latin family",
        "Latin full name",
        "Resolve status",
    ]

    unresolved: dict[str, int] = {}

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            raw = row.get(latin_key, "")
            family, full, status = resolve(
                raw, family_index, fullname_index, psname_index
            )
            if status == "unresolved":
                unresolved[raw] = unresolved.get(raw, 0) + 1
            writer.writerow(
                {
                    "Family": row.get("Family", ""),
                    "Font": row.get("Font", ""),
                    "Has Latin?": row.get("Has Latin?", ""),
                    "Latin origin": row.get("Latin origin", ""),
                    "Latin family raw": raw,
                    "Latin family": family,
                    "Latin full name": full or "",
                    "Resolve status": status,
                }
            )

    print(f"Wrote {out_path} ({len(rows)} rows)")
    print(
        f"Indexed {len(family_index)} families, "
        f"{len(fullname_index)} full names, {len(psname_index)} PostScript names"
    )
    if unresolved:
        print("\nUnresolved 'Latin family' values:")
        for value, n in sorted(unresolved.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {n:3d}  {value!r}")
    else:
        print("\nAll values resolved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
