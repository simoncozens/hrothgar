"""Behavioural tests for the content detector (model, curation, loss)."""

from __future__ import annotations

import random

import pytest
import torch
import torch.nn.functional as F

from hrothgar.content_detector.config import ContentDetectorConfig
from hrothgar.content_detector.constants import (
    EXCLUDED_FAMILIES,
    OFF_SCRIPT_CODEPOINTS,
    PILOT_ANCHORS,
    PILOT_LOWERCASE,
    PILOT_NEGATIVES,
    PILOT_UPPERCASE,
    nfd_base_codepoint,
)
from hrothgar.content_detector.dataset import (
    ContentDatasetMaker,
    axis_positions_for_font,
    diacritic_variants,
    exclude_families,
    exclude_unusable_fonts,
    hard_negative_pool,
    is_shape_identical,
    normalised_height,
    targets_for_codepoints,
)
from hrothgar.content_detector.evaluation import (
    DEFAULT_TARGET_FAR,
    _auc,
    _summary,
    build_content_panel,
    content_report,
    off_sample_buckets,
)
from hrothgar.content_detector.losses import height_loss, margin_hinges, ovr_bce_loss
from hrothgar.content_detector.model import ContentDetector

DUMMY_REPO = "tests/dummy_repo"
ROBOTO_VF = "tests/dummy_repo/ofl/roboto/Roboto[wdth,wght].ttf"
ABEEZEE = "tests/dummy_repo/ofl/abeezee/ABeeZee-Italic.ttf"


def _config(**overrides) -> ContentDetectorConfig:
    defaults = dict(
        anchors=[ord(c) for c in "aeg"],
        glyph_size=32,
        anchors_per_batch=3,
        fonts_per_anchor=2,
        num_workers=0,
        panel_fonts=4,
    )
    defaults.update(overrides)
    return ContentDetectorConfig(**defaults)


# --------------------------------------------------------------------------- #
# Curation
# --------------------------------------------------------------------------- #


def test_diacritic_variants_are_base_letter_siblings() -> None:
    pool = [0xE9, 0xE8, ord("e"), ord("o")]
    assert set(diacritic_variants(ord("e"), pool)) == {0xE9, 0xE8}
    # A precomposed anchor behaves the same way: á's siblings are the a-forms,
    # including the unaccented ``a`` (a "missing diacritic" negative).
    pool = [0xE0, ord("a"), ord("e")]
    assert set(diacritic_variants(0xE1, pool)) == {0xE0, ord("a")}


def test_hard_negative_pool_includes_confusables_and_diacritics() -> None:
    pool = [0xE9, 0xE8]
    hard = hard_negative_pool(ord("o"), pool)
    assert ord("O") in hard  # case sibling
    assert 0x30 in hard  # digit zero
    assert set(hard_negative_pool(ord("e"), pool)) >= {0xE9, 0xE8}


def test_cross_script_homoglyphs_are_not_negatives() -> None:
    # Cyrillic/Greek homoglyphs share the Latin outline in most fonts, so they
    # are impossible (not hard) negatives and are kept out of the pool.
    off_script = set(OFF_SCRIPT_CODEPOINTS)
    for anchor in "aecos":
        assert not (set(hard_negative_pool(ord(anchor), PILOT_NEGATIVES)) & off_script)
    assert not (set(PILOT_NEGATIVES) & off_script)


def test_nfd_base_of_homoglyph_is_itself() -> None:
    # Cyrillic а must not be treated as a diacritic variant of Latin a.
    assert nfd_base_codepoint(0x0430) == 0x0430
    assert hard_negative_pool(0x0430, []) == []


def test_targets_are_multi_hot_over_anchors() -> None:
    anchors = [ord("a"), ord("e")]
    targets = targets_for_codepoints([ord("e"), ord("x"), ord("a")], anchors)
    assert torch.equal(targets, torch.tensor([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]))


# --------------------------------------------------------------------------- #
# Model + loss
# --------------------------------------------------------------------------- #


def test_model_probabilities_are_independent_and_bounded() -> None:
    config = _config()
    model = ContentDetector(config).eval()
    with torch.no_grad():
        probs = model.predict_proba(torch.randn(5, 1, 32, 32))
    assert probs.shape == (5, len(config.anchors))
    assert (probs >= 0).all() and (probs <= 1).all()
    # One-vs-rest: the columns need not sum to 1 (unlike a softmax).
    assert not torch.allclose(probs.sum(dim=1), torch.ones(5))


def test_probability_for_selects_the_right_slot() -> None:
    config = _config()
    model = ContentDetector(config).eval()
    images = torch.randn(4, 1, 32, 32)
    with torch.no_grad():
        probs = model.predict_proba(images)
        assert torch.allclose(model.probability_for(images, ord("e")), probs[:, 1])


def test_margin_hinges_are_zero_when_satisfied() -> None:
    cosine = torch.tensor([[0.5, -0.4, -0.5], [-0.3, 0.6, -0.2]])
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ranking, positive = margin_hinges(
        cosine, targets, ranking_margin=0.1, positive_margin=0.1
    )
    assert float(ranking) == 0.0
    assert float(positive) == 0.0


def test_margin_hinges_punish_a_competing_anchor() -> None:
    # The glyph's own anchor (index 0) is *beaten* by index 1.
    cosine = torch.tensor([[0.10, 0.60, -0.5]])
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    ranking, positive = margin_hinges(
        cosine, targets, ranking_margin=0.1, positive_margin=0.0
    )
    assert float(ranking) == pytest.approx(0.6)
    assert float(positive) == 0.0


def test_ranking_hinge_ignores_a_global_shift() -> None:
    # This is the point of writing the hinge in raw cosine: a per-anchor bias is a
    # constant shift, so it cannot buy any of the ranking margin.
    cosine = torch.tensor([[0.10, 0.60, -0.5]])
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    flat = margin_hinges(cosine, targets, ranking_margin=0.1, positive_margin=0.0)[0]
    shifted = margin_hinges(
        cosine + 5.0, targets, ranking_margin=0.1, positive_margin=0.0
    )[0]
    assert torch.allclose(flat, shifted)
    # The positive hinge, by design, is *not* invariant — it pins the absolute level.
    pos_flat = margin_hinges(cosine, targets, ranking_margin=0.1, positive_margin=0.5)[1]
    pos_shifted = margin_hinges(
        cosine + 5.0, targets, ranking_margin=0.1, positive_margin=0.5
    )[1]
    assert float(pos_flat) > float(pos_shifted)


def test_margin_hinges_ignore_rows_without_a_claimed_anchor() -> None:
    cosine = torch.tensor([[0.1, 0.2], [0.05, 0.9]])
    targets = torch.tensor([[0.0, 0.0], [1.0, 0.0]])  # row 0 has no anchor
    ranking, positive = margin_hinges(
        cosine, targets, ranking_margin=0.1, positive_margin=0.1
    )
    assert float(ranking) == pytest.approx(0.95)  # 0.1 - (0.05 - 0.9)
    assert float(positive) == pytest.approx(0.05)  # 0.1 - 0.05
    # An all-negative batch has nothing to hinge on.
    empty = margin_hinges(
        cosine, torch.zeros_like(targets), ranking_margin=0.1, positive_margin=0.1
    )
    assert float(empty[0]) == 0.0 and float(empty[1]) == 0.0


def test_ovr_bce_prefers_correct_targets() -> None:
    anchors = [ord("a"), ord("e"), ord("g")]
    targets = targets_for_codepoints([ord("a"), ord("e"), ord("a")], anchors)
    good = torch.tensor([[6.0, -6.0, -6.0], [-6.0, 6.0, -6.0], [6.0, -6.0, -6.0]])
    bad = torch.flip(good, dims=[1])
    assert ovr_bce_loss(good, targets) < ovr_bce_loss(bad, targets)
    # A glyph that is none of the anchors is a negative for every column.
    assert torch.isfinite(ovr_bce_loss(good, torch.zeros_like(targets)))


# --------------------------------------------------------------------------- #
# Height cue
# --------------------------------------------------------------------------- #


def test_normalised_height_is_relative_to_x_height() -> None:
    # The target is ink-height / x-height, clamped at 2 and halved into [0, 1], so
    # the ratio is what matters, not the absolute size: an x-height glyph and a
    # capital score the same in a small-x-height font and a large one.
    assert normalised_height(0.5, 0.5) == pytest.approx(0.5)  # ratio 1
    assert normalised_height(0.75, 0.5) == pytest.approx(0.75)  # ratio 1.5 (a cap)
    assert normalised_height(2.0, 0.5) == pytest.approx(1.0)  # clamped
    # A font with no x-height metric must not blow the ratio up.
    assert normalised_height(0.5, 0.0) == pytest.approx(1.0)


def test_height_loss_matches_mse_and_respects_mask() -> None:
    predicted = torch.tensor([0.0, 1.0], requires_grad=True)
    target = torch.tensor([1.0, 1.0])
    assert float(height_loss(predicted, target).detach()) == pytest.approx(0.5)
    # Masking the wrong row leaves the correct one, which costs nothing here.
    assert (
        float(height_loss(predicted, target, torch.tensor([0.0, 1.0])).detach())
        == 0.0
    )
    # A fully-masked batch is a graph-carrying zero, not a crash.
    zero = height_loss(predicted, target, torch.zeros(2))
    assert zero.requires_grad and float(zero.detach()) == 0.0
    # An empty batch likewise.
    assert float(height_loss(torch.zeros(0), torch.zeros(0))) == 0.0


def test_forward_with_aux_predicts_a_bounded_height() -> None:
    config = _config()
    model = ContentDetector(config).eval()
    images = torch.randn(4, 1, 32, 32)
    with torch.no_grad():
        logits, cosine, height = model.forward_with_aux(images)
        scores_logits, scores_cosine = model.scores(images)
    assert height.shape == (4,)
    assert ((height >= 0) & (height <= 1)).all()  # sigmoid-bounded, like the target
    # ``scores`` is the same two outputs as ``forward_with_aux`` minus the aux head.
    assert torch.allclose(logits, scores_logits)
    assert torch.allclose(cosine, scores_cosine)


def test_training_step_trains_the_height_head() -> None:
    config = _config()
    maker = ContentDatasetMaker(DUMMY_REPO, config)
    model = ContentDetector(config)
    batch = next(iter(maker.train_loader()))
    _logits, _cosine, height = model.forward_with_aux(batch["images"])
    loss = height_loss(height, batch["height"], batch["height_mask"])
    loss.backward()
    assert model.height_head[0].weight.grad is not None


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


def test_dataset_batch_has_cross_style_positives_and_negatives() -> None:
    config = _config()
    maker = ContentDatasetMaker(DUMMY_REPO, config)
    batch = next(iter(maker.train_loader()))

    images, targets, codepoints = batch["images"], batch["targets"], batch["codepoints"]
    assert images.shape[1:] == (1, 32, 32)
    assert targets.shape == (images.shape[0], len(config.anchors))
    assert codepoints.shape == (images.shape[0],)
    # Every row is either a positive for exactly one anchor or a pure negative.
    assert (targets.sum(dim=1) <= 1).all()
    # Each sampled anchor contributes at least one positive.
    for anchor in config.anchors:
        assert (codepoints == anchor).any()


def test_dataset_batch_carries_height_targets_and_mask() -> None:
    config = _config()
    maker = ContentDatasetMaker(DUMMY_REPO, config)
    batch = next(iter(maker.train_loader()))

    heights, mask = batch["height"], batch["height_mask"]
    assert heights.shape == (batch["images"].shape[0],)
    assert mask.shape == heights.shape
    assert ((heights >= 0) & (heights <= 1)).all()
    assert (mask > 0).any()  # in-font glyphs are real renders with a real height


def test_content_frame_is_the_tight_global_window() -> None:
    from hrothgar.content_detector.dataset import content_frame
    from hrothgar.render import UNIFORM_FRAME, Frame

    frame = content_frame(_config(anchors=[ord("o")]))
    # The dataset-global window measured by scripts/analyze_glyph_extents.py.
    assert frame == Frame(-0.20, 1.20, -0.36, 1.04)
    assert frame.em_span == pytest.approx(1.40)
    # 1.40em instead of the legacy 2.0em is 1.43x more pixels per em.
    assert UNIFORM_FRAME.em_span / frame.em_span == pytest.approx(1.43, abs=0.01)


def test_dataset_draws_glyphs_into_the_configured_frame() -> None:
    from hrothgar.content_detector.dataset import (
        Cell,
        _CellDataset,
        _render_font,
        content_frame,
    )
    from hrothgar.render import render

    config = _config(anchors=[ord("o")], negatives=[ord("x")], augment=False)
    font = _render_font(ROBOTO_VF)
    framed, _ = render(
        font, codepoint=ord("o"), size=config.glyph_size, frame=content_frame(config)
    )

    item = _CellDataset([Cell(ROBOTO_VF, ord("o"), "sans")], config)[0]
    assert item["codepoints"][0] == ord("o")  # the positive comes first
    assert item["images"].shape[1:] == (1, config.glyph_size, config.glyph_size)
    assert torch.allclose(item["images"][0, 0], framed)
    # ... and the tight frame really is a different picture from the legacy one.
    legacy_image, _ = render(font, codepoint=ord("o"), size=config.glyph_size)
    assert not torch.allclose(framed, legacy_image)


def test_panel_is_stored_compactly_and_scored_in_chunks() -> None:
    from hrothgar.content_detector.evaluation import build_content_panel, score_panel
    from hrothgar.googlefonts import GoogleFont

    config = _config(anchors=[ord("o")], negatives=[ord("x")])
    panel = build_content_panel([GoogleFont(ROBOTO_VF)], config, max_fonts=1)
    assert len(panel.images) == len(panel.codepoints) > 0
    # uint8 keeps the panel ~4x smaller than a float32 copy.
    assert panel.images.dtype == torch.uint8
    assert float(panel.batch(0, 1, "cpu").max()) <= 1.0

    # Chunked scoring must be numerically identical to scoring in one pass, or
    # the memory fix would silently change every reported metric.
    model = ContentDetector(config).eval()
    whole_e, whole_l = score_panel(model, panel, "cpu", batch_size=len(panel.images))
    chunk_e, chunk_l = score_panel(model, panel, "cpu", batch_size=1)
    assert torch.allclose(whole_e, chunk_e, atol=1e-5)
    assert torch.allclose(whole_l, chunk_l, atol=1e-5)


def test_axis_positions_default_for_non_variable_font() -> None:
    assert axis_positions_for_font(ABEEZEE, 3, random.Random(0)) == [()]
    assert axis_positions_for_font(ROBOTO_VF, 1, random.Random(0)) == [()]


def test_axis_positions_for_variable_font() -> None:
    from pathlib import Path

    from hrothgar.dataset import fvar_axes

    n_axes = len(fvar_axes(Path(ROBOTO_VF)))
    positions = axis_positions_for_font(ROBOTO_VF, 3, random.Random(0))
    assert positions[0] == ()  # the default instance comes first
    assert len(positions) == 3
    assert len(set(positions)) == 3  # all distinct
    for position in positions[1:]:
        assert len(position) == n_axes  # full fvar-order position


def test_axis_positions_are_deterministic() -> None:
    first = axis_positions_for_font(ROBOTO_VF, 4, random.Random(7))
    second = axis_positions_for_font(ROBOTO_VF, 4, random.Random(7))
    assert first == second


def test_axis_positions_fill_with_random_combinations() -> None:
    # Roboto has two axes, so single-axis extremes can't fill six positions;
    # the remainder comes from random per-axis combinations.
    positions = axis_positions_for_font(ROBOTO_VF, 6, random.Random(0))
    assert len(positions) == 6
    assert len(set(positions)) == 6
    assert () not in positions[1:]  # the default is never duplicated


def test_cell_with_axis_position_renders() -> None:
    from hrothgar.content_detector.dataset import Cell, _CellDataset

    config = _config(anchors=[ord("o")], negatives=[0x043E])
    cell = Cell(ROBOTO_VF, ord("o"), "sans", ((100.0, 900.0),))
    item = _CellDataset([cell], config)[0]
    assert item["images"].shape[1:] == (1, 32, 32)
    assert item["codepoints"].tolist()[0] == ord("o")


def test_cell_positions_rotate_with_epoch() -> None:
    from hrothgar.content_detector.dataset import Cell

    positions = ((), (100.0, 900.0), (75.0, 100.0))
    cell = Cell(ROBOTO_VF, ord("o"), "sans", positions, slot=1)
    assert cell.position_for(0) == (100.0, 900.0)
    assert cell.position_for(1) == (75.0, 100.0)
    assert cell.position_for(2) == ()  # wraps


def test_dataset_resamples_with_the_epoch() -> None:
    from hrothgar.content_detector.dataset import Cell, _CellDataset

    # A cell whose axis position changes must render differently per epoch, and
    # its negatives/augmentation must not be frozen either.
    config = _config(anchors=[ord("o")], negatives=[0x043E])
    cell = Cell(ROBOTO_VF, ord("o"), "sans", ((), (100.0, 900.0)), slot=0)
    dataset = _CellDataset([cell], config)
    dataset.epoch = 0
    first = dataset[0]["images"]
    dataset.epoch = 1
    second = dataset[0]["images"]
    assert not torch.allclose(first, second)


def test_unusable_fonts_are_dropped() -> None:
    from hrothgar.fontquant import FontQuant, FontQuantTable

    class _FakeFont:
        def __init__(self, path: str):
            self.path = path

    table = FontQuantTable(
        {
            # all-caps: lowercase slots hold uppercase shapes
            "aboreto-regular.ttf": FontQuant("uppercase", False, "", "", 700.0, 700.0),
            # blank: no vertical extent at all
            "adobeblank-regular.ttf": FontQuant("uppercase", True, "", "", 0.0, 0.0),
            "roboto.ttf": FontQuant(
                "lowercase", False, "double_story", "single_story", 750.0, 528.0
            ),
        }
    )
    fonts = [
        _FakeFont("ofl/aboreto/Aboreto-Regular.ttf"),
        _FakeFont("ofl/adobeblank/AdobeBlank-Regular.ttf"),
        _FakeFont("ofl/roboto/Roboto.ttf"),
    ]
    assert {font.path for font in exclude_unusable_fonts(fonts, table)} == {
        "ofl/roboto/Roboto.ttf"
    }


def test_training_step_backpropagates() -> None:
    config = _config()
    maker = ContentDatasetMaker(DUMMY_REPO, config)
    model = ContentDetector(config)

    batch = next(iter(maker.train_loader()))
    loss = ovr_bce_loss(model(batch["images"]), batch["targets"])
    assert torch.isfinite(loss)
    loss.backward()
    assert model.anchor_bank.grad is not None


def test_is_shape_identical_uses_a_tight_tolerance() -> None:
    blank = torch.zeros(8, 8)
    assert is_shape_identical(blank, blank.clone())
    slightly_different = blank.clone()
    slightly_different[0, 0] = 1.0
    assert not is_shape_identical(blank, slightly_different)


def test_cell_dataset_drops_shape_identical_negatives() -> None:
    from hrothgar.content_detector.dataset import Cell, _CellDataset

    # Cyrillic о is the same outline as Latin o in this font, so it must not
    # survive as a negative for the o anchor.
    config = _config(
        anchors=[ord("o")],
        negatives=[0x043E],
        hard_negatives_per_cell=0,
        random_negatives_per_cell=1,
    )
    roboto = "tests/dummy_repo/ofl/roboto/Roboto[wdth,wght].ttf"
    item = _CellDataset([Cell(roboto, ord("o"), "sans")], config)[0]
    assert item["codepoints"].tolist() == [ord("o")]
    assert item["images"].shape[0] == 1


def test_panel_drops_shape_identical_off_sample() -> None:
    from hrothgar.googlefonts import GoogleFont

    font = GoogleFont("tests/dummy_repo/ofl/roboto/Roboto[wdth,wght].ttf")
    config = _config(anchors=[ord("o")], negatives=[0x043E])
    panel = build_content_panel([font], config, max_fonts=1)
    assert ord("o") in panel.codepoints
    assert 0x043E not in panel.codepoints


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_auc_separates_and_ranks_ties() -> None:
    assert _auc([0.9, 0.8], [0.2, 0.1]) == 1.0
    assert _auc([0.1, 0.2], [0.8, 0.9]) == 0.0
    assert _auc([0.5], [0.5]) == 0.5
    assert _auc([], [0.5]) is None


def test_off_sample_buckets_partition_the_off_anchor_pool() -> None:
    config = _config(anchors=[ord(c) for c in "aeog"])
    hard, easy = off_sample_buckets(config)
    anchors = set(config.anchors)
    assert hard and easy
    assert not (set(hard) & anchors)  # anchors are never off-sample
    assert not (set(hard) & set(easy))  # the two buckets are disjoint
    assert 0x30 in hard  # digit zero confuses 'o'
    assert ord("z") in easy  # unrelated letter


def test_panel_is_stratified_and_bucketed() -> None:
    config = _config()
    maker = ContentDatasetMaker(DUMMY_REPO, config)
    panel = build_content_panel(maker.test_fonts, config, max_fonts=4)
    n = len(panel.codepoints)
    assert n > 0
    assert len(panel.strata) == n and len(panel.buckets) == n
    assert len(panel.paths) == n
    assert "anchor" in panel.buckets
    assert set(panel.buckets) <= {"anchor", "off_hard", "off_easy"}


def test_content_report_has_anchor_and_stratum_breakdowns() -> None:
    config = _config()
    maker = ContentDatasetMaker(DUMMY_REPO, config)
    panel = build_content_panel(maker.test_fonts, config, max_fonts=4)
    model = ContentDetector(config).eval()
    report = content_report(model, panel, config, "cpu")
    assert set(report["per_anchor"]) == {str(a) for a in config.anchors}
    assert report["per_stratum"]
    # Embedding-space diagnostics: similarity is a cosine, the margin a difference.
    similarity = report["embedding_similarity"]
    assert -1.0 <= similarity["same_letter"] <= 1.0
    assert -1.0 <= similarity["cross_letter"] <= 1.0
    assert -2.0 <= report["anchor_margin_cosine"] <= 2.0
    for metrics in report["per_anchor"].values():
        assert "bias" in metrics and "positive_recall" in metrics
    for metrics in report["per_stratum"].values():
        assert "auc_hard" in metrics and "positive_recall" in metrics


def test_content_report_recall_at_far_and_geometry() -> None:
    config = _config()
    maker = ContentDatasetMaker(DUMMY_REPO, config)
    panel = build_content_panel(maker.test_fonts, config, max_fonts=4)
    model = ContentDetector(config).eval()
    report = content_report(model, panel, config, "cpu")

    assert report["target_far"] == DEFAULT_TARGET_FAR
    assert 0.0 <= report["recall_at_far"] <= 1.0
    assert set(report["anchor_neighbours"]) == {str(a) for a in config.anchors}
    for metrics in report["per_anchor"].values():
        assert "recall_at_far" in metrics
        assert set(metrics["positive_cosine"]) == {
            "mean",
            "std",
            "p10",
            "p25",
            "p50",
            "p75",
            "p90",
        }


def test_summary_handles_values_and_empty() -> None:
    summary = _summary([0.0, 1.0, 2.0, 3.0])
    assert summary["mean"] == 1.5
    assert summary["p50"] == 1.5
    assert summary["p10"] is not None and summary["p90"] is not None
    assert _summary([])["mean"] is None


def test_content_report_confusable_breakdown_and_top_negatives() -> None:
    config = _config(anchors=[ord(c) for c in "aeog"])
    maker = ContentDatasetMaker(DUMMY_REPO, config)
    panel = build_content_panel(maker.test_fonts, config, max_fonts=4)
    model = ContentDetector(config).eval()
    report = content_report(model, panel, config, "cpu", top_negatives=3)

    # 'o' has curated confusables (O, 0, ...), so its breakdown must be populated.
    o_negatives = report["per_anchor"][str(ord("o"))]["negatives"]
    assert o_negatives
    for metrics in o_negatives.values():
        assert metrics["n"] > 0
        assert {"auc", "mean_cosine", "above_threshold"} <= set(metrics)

    # Top negatives are per anchor and never the anchor's own codepoint.
    assert set(report["top_negatives"]) == {str(a) for a in config.anchors}
    for anchor, entries in report["top_negatives"].items():
        assert all(entry["codepoint"] != int(anchor) for entry in entries)
        assert all("path" in entry for entry in entries)


def test_pilot_anchors_are_latin_letters() -> None:
    assert all(chr(a).isalpha() for a in PILOT_ANCHORS)
    assert len(set(PILOT_ANCHORS)) == len(PILOT_ANCHORS)  # no duplicates


def test_pilot_uppercase_tier_mirrors_the_lowercase_tier() -> None:
    assert set(PILOT_LOWERCASE) <= set(PILOT_ANCHORS)
    assert set(PILOT_UPPERCASE) == {ord(chr(a).upper()) for a in PILOT_LOWERCASE}
    assert set(PILOT_UPPERCASE) <= set(PILOT_ANCHORS)


def test_case_siblings_are_mutual_confusables() -> None:
    for lower in "aegofcs":
        upper = lower.upper()
        assert ord(upper) in hard_negative_pool(ord(lower), PILOT_NEGATIVES)
        assert ord(lower) in hard_negative_pool(ord(upper), PILOT_NEGATIVES)


def test_uppercase_anchors_get_diacritic_variants() -> None:
    # The uppercase precomposed forms live in the pool, so They are derived.
    hard = hard_negative_pool(ord("A"), PILOT_NEGATIVES)
    assert 0xC0 in hard and 0xC1 in hard  # À, Á


def test_pilot_includes_accented_a_pairs() -> None:
    # á and à are anchors so the detector can catch fonts (e.g. Laila) that
    # have them swapped.
    assert 0xE1 in PILOT_ANCHORS and 0xE0 in PILOT_ANCHORS
    # Their case siblings are negatives, not anchors.
    assert 0xC1 in hard_negative_pool(0xE1, PILOT_NEGATIVES)
    assert 0xC1 not in PILOT_ANCHORS


def test_accented_a_hard_negatives_are_each_other_and_base() -> None:
    for anchor in (0xE1, 0xE0):
        hard = hard_negative_pool(anchor, PILOT_NEGATIVES)
        other = 0xE0 if anchor == 0xE1 else 0xE1
        assert other in hard  # à is a negative for á, and vice versa
        assert ord("a") in hard  # the unaccented base is too


def test_excluded_families_are_dropped() -> None:
    class _FakeFont:
        def __init__(self, family: str):
            self.family = family

    fonts = [_FakeFont("Laila"), _FakeFont("Roboto"), _FakeFont("ABeeZee")]
    kept = {font.family for font in exclude_families(fonts)}
    assert "Laila" not in kept
    assert kept == {"Roboto", "ABeeZee"}
    assert "Laila" in EXCLUDED_FAMILIES
