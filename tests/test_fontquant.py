"""Tests for the ``fontquant.csv`` metadata loader."""

from __future__ import annotations

from hrothgar import fontquant
from hrothgar.fontquant import FontQuantTable

CSV = """Font,Location,appearance/ascender,appearance/x_height,casing/lowercase_shapes,casing/unicase,appearance/lowercase_a_style,appearance/lowercase_g_style
Aboreto-Regular.ttf,,700,700,uppercase,false,double_story,
AdobeBlank-Regular.ttf,,0,0,uppercase,true,,
Roboto[wdth,wght].ttf,,750,528,lowercase,false,double_story,single_story
Gruppo-Regular.ttf,,544,374,lowercase,false,single_story,single_story
"""


def _table(tmp_path) -> FontQuantTable:
    path = tmp_path / "fontquant.csv"
    path.write_text(CSV)
    return FontQuantTable.load(path)


def test_has_lowercase(tmp_path) -> None:
    table = _table(tmp_path)
    assert table.has_lowercase("ofl/aboreto/Aboreto-Regular.ttf") is False
    assert table.has_lowercase("ofl/roboto/Roboto[wdth,wght].ttf") is True
    # Unknown fonts are assumed usable rather than dropped.
    assert table.has_lowercase("ofl/unknown/Whatever.ttf") is True


def test_is_blank(tmp_path) -> None:
    table = _table(tmp_path)
    assert table.is_blank("ofl/adobeblank/AdobeBlank-Regular.ttf") is True
    assert table.is_blank("ofl/roboto/Roboto[wdth,wght].ttf") is False


def test_lookup_is_case_insensitive_on_basename(tmp_path) -> None:
    table = _table(tmp_path)
    quant = table.get("/some/prefix/gruppo-regular.ttf")
    assert quant is not None
    assert quant.lowercase_a_style == "single_story"
    assert quant.unicase is False


def test_load_default_or_none_missing(monkeypatch) -> None:
    monkeypatch.setenv("FONTQUANT_CSV", "/nonexistent/fontquant.csv")
    assert fontquant.load_default_or_none() is None
