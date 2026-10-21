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
    SlotRecord,
    _print_findings,
    is_blank_render,
    is_excluded_font,
    sweep_counts,
)


class _Args:
    max_print = 10
    ambiguous = 0.5
    wrong_margin = 0.35
    confident_above = 0.70


def _stats(**overrides) -> dict:
    stats = {
        "fonts_scanned": 3,
        "slots_checked": 3,
        "skipped_caps": [],
        "skipped_blank": [],
        "skipped_special": [],
        "skipped_collapsed": [],
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
            "claimed_probability": 0.70,
            "top": 0xE0,
            "top_probability": 1.00,
            "cosine_excess": 0.66,
        },
        {
            "label": "ambiguous",
            "family": "Barcode",
            "path": "ofl/barcode/Barcode.ttf",
            "claimed": 0x61,
            "claimed_probability": 0.11,
            "top": 0x6F,
            "top_probability": 0.12,
            "cosine_excess": 0.05,
        },
        {
            "label": "blank",
            "family": "Adobe Blank",
            "path": "ofl/adobeblank/AdobeBlank-Regular.ttf",
            "claimed": 0x65,
            "claimed_probability": 0.0,
            "top": 0x65,
            "top_probability": 0.0,
            "cosine_excess": 0.0,
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


def test_sweep_counts_requires_a_cosine_margin_and_confidence() -> None:
    def record(cosine_excess, competitor_probability, *, same=False, blank=False):
        return SlotRecord(
            claimed_probability=0.70,
            top_probability=1.00,
            top_is_claimed=same,
            competitor_probability=competitor_probability,
            cosine_excess=cosine_excess,
            blank=blank,
        )

    records = [
        record(0.66, 1.00),  # the Laila case: always flagged
        record(0.08, 0.99),  # a legitimate a whose neighbour ranks a little higher
        record(0.50, 0.65),  # big margin, but the competitor is not confident
        record(-0.80, 0.10, same=True),  # correct: never flagged
        record(0.66, 1.00, blank=True),  # blank: never flagged
    ]
    table = sweep_counts(records, (0.35,), (0.70,))
    assert table == [(0.35, 0.70, 1)]


# --------------------------------------------------------------------------- #
# Font centring and collapse detection
# --------------------------------------------------------------------------- #


def test_collapsed_font_detects_a_whole_font_failure() -> None:
    from scripts.content_detector.detect_content_errors import collapsed_font

    anchors = [ord(c) for c in "aegofc"]
    # Two slots win their own anchor, four are won by `e`: a font whose style the
    # encoder cannot read, not four individually mis-encoded slots.
    cosine = torch.zeros(len(anchors), len(anchors))
    for i in range(len(anchors)):
        cosine[i, i if i < 2 else 1] = 1.0

    result = collapsed_font(cosine, anchors, anchors)
    assert result is not None
    assert result["n_slots"] == len(anchors)
    assert result["n_disagreeing"] == len(anchors) - 2
    assert result["top_foreign"] == ord("e")
    assert result["top_foreign_count"] == len(anchors) - 2


def test_collapsed_font_ignores_a_healthy_font() -> None:
    from scripts.content_detector.detect_content_errors import collapsed_font

    anchors = [ord(c) for c in "aego"]
    cosine = torch.eye(len(anchors))
    assert collapsed_font(cosine, anchors, anchors) is None

    # Even a couple of odd slots is not a collapse — most claims must disagree.
    cosine[1, 1] = 0.0
    cosine[1, 0] = 1.0
    assert collapsed_font(cosine, anchors, anchors) is None


def test_collapsed_font_needs_enough_slots_to_judge() -> None:
    from scripts.content_detector.detect_content_errors import collapsed_font

    anchors = [ord("a"), ord("e")]
    assert collapsed_font(torch.zeros(2, 2), anchors, anchors) is None


def test_font_scores_centres_on_the_font_and_handles_a_single_glyph() -> None:
    from hrothgar.content_detector.config import ContentDetectorConfig
    from hrothgar.content_detector.model import ContentDetector
    from scripts.content_detector.detect_content_errors import font_scores

    config = ContentDetectorConfig(anchors=[ord("a"), ord("e")], glyph_size=32)
    model = ContentDetector(config).eval()
    images = torch.rand(4, 1, 32, 32)

    raw_cos, raw_logits = font_scores(model, images, "cpu", centre=False)
    centred_cos, centred_logits = font_scores(model, images, "cpu", centre=True)
    assert raw_cos.shape == centred_cos.shape == (4, 2)
    assert not torch.allclose(raw_cos, centred_cos)  # the font mean was removed

    # Logits are just ``scale * cosine + bias``, so centring cannot reorder the
    # geometry-dependent part; only the reference point moved.
    expected = model.logit_scale.detach() * raw_cos + model.anchor_bias.detach()
    assert torch.allclose(raw_logits, expected, atol=1e-5)
    assert not torch.allclose(raw_logits, centred_logits)

    # A single glyph has nothing to centre on, so centring is a no-op.
    one = images[:1]
    assert torch.allclose(
        font_scores(model, one, "cpu", centre=True)[0],
        font_scores(model, one, "cpu", centre=False)[0],
    )
