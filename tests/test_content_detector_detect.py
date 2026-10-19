"""Regression tests for the library content-error detection script.

``scripts/`` is not an installed package, so put the repo root on the path before
importing it.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.content_detector.detect_content_errors import (  # noqa: E402
    _print_findings,
    is_blank_render,
    is_excluded_font,
    sweep_counts,
)


class _Args:
    max_print = 10
    ambiguous = 0.5
    reject_below = 0.30
    confident_above = 0.70
    margin = 0.0


def _stats(**overrides) -> dict:
    stats = {
        "fonts_scanned": 3,
        "slots_checked": 3,
        "skipped_caps": [],
        "skipped_blank": [],
        "skipped_special": [],
        "tallies": Counter(),
    }
    stats.update(overrides)
    return stats


def test_print_findings_handles_all_three_labels(capsys) -> None:
    # A blank finding has no meaningful *competing* anchor, which previously
    # crashed the printer when the field was assumed to be a codepoint.
    findings = [
        {
            "label": "wrong",
            "family": "Laila",
            "path": "ofl/laila/Laila-Regular.ttf",
            "claimed": 0xE1,
            "claimed_probability": 0.08,
            "top": 0xE0,
            "top_probability": 0.71,
        },
        {
            "label": "ambiguous",
            "family": "Barcode",
            "path": "ofl/barcode/Barcode.ttf",
            "claimed": 0x61,
            "claimed_probability": 0.11,
            "top": 0x6F,
            "top_probability": 0.12,
        },
        {
            "label": "blank",
            "family": "Adobe Blank",
            "path": "ofl/adobeblank/AdobeBlank-Regular.ttf",
            "claimed": 0x65,
            "claimed_probability": 0.0,
            "top": 0x65,
            "top_probability": 0.0,
        },
    ]
    _print_findings(
        findings,
        _Args(),
        _stats(tallies=Counter({"wrong:á": 1, "ambiguous:a": 1, "blank:e": 1})),
    )

    out = capsys.readouterr().out
    assert "wrong content : 1" in out
    assert "ambiguous     : 1" in out
    assert "blank         : 1" in out
    assert "looks like à" in out  # the Laila accent-swap reading
    assert "blank slots" in out


def test_print_findings_reports_skipped_fonts(capsys) -> None:
    _print_findings([], _Args(), _stats(skipped_caps=["Aboreto", "Bungee"]))
    out = capsys.readouterr().out
    assert "skipped 2 all-caps/small-caps fonts" in out


def test_is_blank_render() -> None:
    white = torch.ones(1, 16, 16)
    assert is_blank_render(white)
    inked = torch.ones(1, 16, 16)
    inked[0, 4:12, 4:12] = 0.0
    assert not is_blank_render(inked)


def test_is_excluded_font() -> None:
    class _FakeFont:
        def __init__(self, path: str, family: str):
            self.path = path
            self.family = family

    tags = {"Barcode 39": {"/Special use/Barcode": 100}}
    assert is_excluded_font(_FakeFont("ofl/roboto/Roboto.ttf", "Roboto"), {}) is None
    assert (
        is_excluded_font(_FakeFont("ofl/barcode/X.ttf", "Barcode 39"), tags)
        == "special-use"
    )
    assert (
        is_excluded_font(_FakeFont("ofl/foo/Foo-SC.ttf", "Foo"), {})
        == "small-caps/guide"
    )
    assert (
        is_excluded_font(_FakeFont("ofl/betania/BetaniaPatmosGuideLine.ttf", "X"), {})
        == "small-caps/guide"
    )


def test_sweep_counts_requires_rejection_and_confidence() -> None:
    # (claimed, top, top_is_claimed, blank)
    records = [
        (0.00, 1.00, False, False),  # the Laila case: always flagged
        (0.89, 0.99, False, False),  # a legitimate a where a neighbour ranks higher
        (0.10, 0.40, False, False),  # nothing confident: never flagged
        (0.90, 0.10, True, False),  # correct: never flagged
        (0.00, 1.00, False, True),  # blank: never flagged
    ]
    table = sweep_counts(records, (0.3,), (0.7,))
    assert table == [(0.3, 0.7, 1)]
