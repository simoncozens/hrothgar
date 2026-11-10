"""Tests for scripts/labeling/build_cedars_families.py.

``scripts/`` is not an installed package, so put the repo root on the path before
importing it. The fixture is ``tests/dummy_repo`` (a miniature Google Fonts
checkout), so nothing here needs network access.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.labeling.build_cedars_families import (  # noqa: E402
    _covers,
    _inks,
    build_families,
    default_face,
    exclusion_reason,
    is_excluded_family,
    load_tags,
    parse_metadata,
)

DUMMY = ROOT / "tests" / "dummy_repo"


def _write_fontquant(tmp_path: Path) -> Path:
    path = tmp_path / "fontquant.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Font",
                "Location",
                "appearance/lowercase_a_style",
                "appearance/lowercase_g_style",
                "casing/lowercase_shapes",
                "casing/unicase",
            ]
        )
        writer.writerow(
            ["ABeeZee-Regular.ttf", "", "double_story", "single_story", "lowercase", "false"]
        )
        writer.writerow(
            [
                "PlayfairDisplay[wght].ttf",
                "",
                "double_story",
                "double_story",
                "lowercase",
                "false",
            ]
        )
        writer.writerow(
            ["Roboto[wdth,wght].ttf", "", "double_story", "single_story", "lowercase", "false"]
        )
        writer.writerow([
            "ShantellSans[BNCE,INFM,SPAC,wght].ttf",
            "",
            "single_story",
            "single_story",
            "lowercase",
            "false",
        ])
    return path


def test_build_families_from_dummy_repo(tmp_path):
    design = build_families(
        DUMMY, fontquant_csv=_write_fontquant(tmp_path), progress=False
    )
    families = {record["family"]: record for record in design["families"]}

    assert set(families) == {
        "ABeeZee",
        "Playfair Display",
        "Roboto",
        "Shantell Sans",
    }
    assert design["count"] == 4
    assert design["dataset"] == "dummy_repo"
    assert design["excluded_count"] == 0
    assert design["excluded_families"] == []
    assert design["exclusion_policy"] == {
        "prefixes": ["Noto Sans", "Noto Serif", "Playwrite", "Rubik "],
        "suffixes": ["SC"],
    }

    playfair = families["Playfair Display"]
    assert playfair["serif"] is True
    assert playfair["serif_tags"] == ["/Serif/Didone"]
    assert playfair["tags"]["/Serif/Didone"] == 100

    assert families["ABeeZee"]["serif"] is False
    assert families["Roboto"]["serif"] is False
    assert families["Shantell Sans"]["serif"] is False

    for record in families.values():
        assert _covers(record["glyphs"], "alt")
        assert _covers(record["glyphs"], "lmn")
        assert _covers(record["ink"], "alt")
        assert _covers(record["ink"], "lmn")
        assert _inks(record["ink"], "alt")
        assert _inks(record["ink"], "lmn")
        assert record["file"] and not record["file"].startswith("/")
        assert record["fontquant"] is not None
        assert "error" not in record

    assert families["ABeeZee"]["fontquant"]["a_style"] == "double_story"
    assert families["ABeeZee"]["fontquant"]["g_style"] == "single_story"
    assert families["Shantell Sans"]["fontquant"]["a_style"] == "single_story"
    assert families["Roboto"]["fontquant"]["unicase"] is False


def test_missing_fontquant_csv_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        build_families(
            DUMMY, fontquant_csv=tmp_path / "nope.csv", progress=False
        )


def test_default_face_prefers_upright_regular():
    files = [
        Path("Foo-Italic.ttf"),
        Path("Foo[wght].ttf"),
        Path("Foo-Regular.ttf"),
    ]
    assert default_face(files).name == "Foo-Regular.ttf"
    assert default_face([Path("Foo-Italic.ttf"), Path("Foo-Bold.ttf")]).name == "Foo-Bold.ttf"


def test_default_face_handles_variable_font():
    files = [
        Path("Roboto-Italic[wdth,wght].ttf"),
        Path("Roboto[wdth,wght].ttf"),
    ]
    assert default_face(files).name == "Roboto[wdth,wght].ttf"


def test_parse_metadata_ignores_inner_font_names():
    metadata = DUMMY / "ofl" / "roboto" / "METADATA.pb"
    name, category, stroke, filenames = parse_metadata(metadata)
    assert name == "Roboto"
    assert category == "SANS_SERIF"
    assert stroke is None
    assert filenames == [
        "Roboto[wdth,wght].ttf",
        "Roboto-Italic[wdth,wght].ttf",
    ], "inner fonts{} blocks must not contribute a second family name"


def test_exclusion_policy():
    # Noto Sans/Serif cuts reuse the base Latin outlines.
    assert is_excluded_family("Noto Sans")
    assert is_excluded_family("Noto Sans Devanagari")
    assert is_excluded_family("Noto Serif Display")
    assert exclusion_reason("Noto Serif Display") == "Noto Sans/Serif cut"

    # Small-caps companions cannot be judged independently of their base.
    assert is_excluded_family("Alegreya SC")
    assert is_excluded_family("Playfair Display SC")
    assert exclusion_reason("Alegreya SC") == "small-caps companion"

    # Playwrite variants are programmatically derived from one design system.
    assert is_excluded_family("Playwrite AR")
    assert is_excluded_family("Playwrite US Modern Guides")
    assert (
        exclusion_reason("Playwrite AR")
        == "programmatically derived Playwrite variant"
    )

    # Decorative Rubik variants are excluded, but the base Rubik is kept.
    assert is_excluded_family("Rubik Bubbles")
    assert is_excluded_family("Rubik Glitch Pop")
    assert is_excluded_family("Rubik Mono One")
    assert exclusion_reason("Rubik Bubbles") == "programmatically derived Rubik variant"
    assert not is_excluded_family("Rubik")

    # Other Noto families are not Latin cuts and stay in the snapshot.
    assert not is_excluded_family("Noto Kufi Arabic")
    assert not is_excluded_family("Noto Music")
    assert not is_excluded_family("Roboto")
    assert exclusion_reason("Roboto") is None


def test_load_tags_folds_instances_and_drops_zero(tmp_path):
    tag_csv = tmp_path / "families.csv"
    with tag_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Family", "", "/Serif/Modern", "70"])
        writer.writerow(["Family", "wght@900", "/Serif/Modern", "90"])
        writer.writerow(["Family", "", "/Serif/Slab", "0"])
        writer.writerow(["Family", "", "/Expressive/Loud", "12.5"])
        writer.writerow(["Other", "", "/Sans/Geometric", "100"])

    tags = load_tags(tag_csv)
    assert tags["Family"]["/Serif/Modern"] == 90
    assert "/Serif/Slab" not in tags["Family"]
    assert tags["Family"]["/Expressive/Loud"] == 12.5
    assert tags["Other"] == {"/Sans/Geometric": 100}
