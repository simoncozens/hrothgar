"""Tests for the frozen text-arm index.

Two things matter here:

* the *query* path (one anchor vs the frozen library) must compute exactly the
  same per-axis similarities as the *build* path (the pairwise tensor), or a
  frozen index would score differently from the pipeline it was trained on; and
* the index must survive a save/load round trip and still rank sensibly.
"""

from __future__ import annotations

import os

import torch

from hrothgar.similarity.axis_combiner import (
    _categorical_sim,
    _numeric_sim,
    _standardize_tensor,
    bank_similarity_tensor,
    categorical_sim_row,
    numeric_sim_row,
    tag_cosine_row,
)
from hrothgar.similarity.metrics import masked_pairwise_cosine
from hrothgar.similarity.text_index import TextIndex, index_is_stale

FAMILIES = ["alpha", "beta", "gamma", "delta"]


def _toy_bank() -> dict:
    """A tiny self-describing bank: one numeric, one vector, two categorical axes, one tag axis."""
    values = torch.tensor(
        [
            [1.0, 1.0],
            [1.1, 1.0],
            [5.0, 5.0],
            [float("nan"), 1.2],
        ],
        dtype=torch.float32,
    )
    numbers = values[~torch.isnan(values)]
    means = torch.stack([numbers.mean(), numbers.mean()])
    stds = torch.stack([numbers.std(), numbers.std()])

    return {
        "families": list(FAMILIES),
        "axis_names": ["weight", "a-storey", "monospaced", "tags-style"],
        "numeric": {
            "weight": {
                "columns": ["appearance/weight", "appearance/width"],
                "values": values,
                "means": means,
                "stds": stds,
            }
        },
        "categorical": {
            "a-storey": {
                "vocab": ["single_story", "double_story"],
                "boolean": False,
                "codes": torch.tensor([0, 0, 1, -1]),
            },
            "monospaced": {
                "vocab": ["false", "true"],
                "boolean": True,
                "codes": torch.tensor([0, 0, 1, 1]),
            },
        },
        "tags": {
            "tags-style": {
                "tags": ["/Sans/Geometric", "/Base/Serif", "/Script/Formal"],
                "values": torch.tensor(
                    [[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
                ),
                "mask": torch.tensor(
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
                ),
            }
        },
    }


def _toy_index(bank: dict | None = None) -> TextIndex:
    bank = bank or _toy_bank()
    return TextIndex(
        families=list(bank["families"]),
        axis_names=list(bank["axis_names"]),
        numeric=bank["numeric"],
        categorical=bank["categorical"],
        tags=bank["tags"],
        w=torch.tensor([0.5, 0.3, 0.2, 0.1]),
        b=torch.tensor(0.05),
        meta={"created_mtime": 1.0},
    )


# ── Query path == build path ──────────────────────────────────────────────────
def test_numeric_row_matches_pairwise_with_missing_values() -> None:
    bank = _toy_bank()["numeric"]["weight"]
    V, P = _standardize_tensor(bank["values"], bank["means"], bank["stds"])
    matrix = _numeric_sim(V, P)
    for i in range(V.shape[0]):
        assert torch.allclose(numeric_sim_row(V[i], P[i], V, P), matrix[i])


def test_categorical_row_matches_pairwise() -> None:
    for codes, boolean in (
        (torch.tensor([0, 0, 1, -1, 1]), False),
        (torch.tensor([0, 0, 1, -1, 1]), True),
    ):
        matrix = _categorical_sim(codes, boolean)
        for i in range(codes.numel()):
            assert torch.equal(categorical_sim_row(int(codes[i]), codes, boolean), matrix[i])


def test_tag_row_matches_pairwise_off_diagonal() -> None:
    axis = _toy_bank()["tags"]["tags-style"]
    matrix, _comparable = masked_pairwise_cosine(axis["values"], axis["mask"])
    for i in range(axis["values"].shape[0]):
        row = tag_cosine_row(axis["values"][i], axis["mask"][i], axis["values"], axis["mask"])
        off = [j for j in range(matrix.shape[0]) if j != i]
        # The pairwise helper zeroes the diagonal; self-similarity is 1 in the row form.
        assert torch.allclose(row[off], matrix[i][off])


def test_index_similarity_row_reproduces_the_build_tensor() -> None:
    bank = _toy_bank()
    S_ref, _names = bank_similarity_tensor(bank)
    index = _toy_index(bank)
    for i, family in enumerate(FAMILIES):
        af = index.features_for_family(family)
        assert af is not None
        S_row = index.similarity_row(af)
        off = [j for j in range(len(FAMILIES)) if j != i]
        assert torch.allclose(S_row[off], S_ref[i][off])


def test_index_score_matches_the_frozen_combiner() -> None:
    bank = _toy_bank()
    S_ref, _names = bank_similarity_tensor(bank)
    index = _toy_index(bank)
    expected = S_ref[0] @ index.w + index.b
    af = index.features_for_family("alpha")
    assert af is not None
    # The anchor's own row differs on the tag diagonal by design (see
    # ``TextIndex.similarity_row``); ``rank`` excludes it.
    assert torch.allclose(index.score(af)[1:], expected[1:])


# ── Lookup / ranking ─────────────────────────────────────────────────────────
def test_rank_is_ordered_and_excludes_the_anchor() -> None:
    index = _toy_index()
    af = index.features_for_family("alpha")
    assert af is not None
    ranked = index.rank(af, k=3)
    assert [family for family, _score in ranked][0] == "beta"
    assert "alpha" not in [family for family, _score in ranked]
    scores = [score for _family, score in ranked]
    assert scores == sorted(scores, reverse=True)


def test_unknown_family_is_reported_missing() -> None:
    index = _toy_index()
    assert index.family_index("Alpha") == 0  # case-insensitive
    assert index.features_for_family("not-a-family") is None


def test_summary_reads_raw_values_and_vocabulary() -> None:
    index = _toy_index()
    af = index.features_for_family("delta")
    assert af is not None
    summary = index.summary(af)
    assert "—" in summary["weight"]  # NaN renders as an em dash
    assert summary["a-storey"] == "—"  # code -1 == not comparable
    assert summary["monospaced"] == "true"
    assert summary["tags-style"] == "none"


def test_rank_only_considers_rankable_rows() -> None:
    index = TextIndex(
        families=list(FAMILIES),
        axis_names=["a-storey"],
        numeric={},
        categorical={
            "a-storey": {
                "vocab": ["single_story", "double_story"],
                "boolean": False,
                "codes": torch.tensor([0, 0, 1, 1]),
            }
        },
        tags={},
        w=torch.tensor([1.0]),
        b=torch.tensor(0.0),
        rankable=[0, 1],
    )
    af = index.features_for_family("delta")
    assert af is not None
    ranked = index.rank(af, k=10)
    assert [family for family, _score in ranked] == ["alpha", "beta"]


# ── Persistence ──────────────────────────────────────────────────────────────
def test_save_load_roundtrip_preserves_scores(tmp_path) -> None:
    index = _toy_index()
    index.rankable = [0, 2]
    index.__post_init__()
    path = index.save(tmp_path / "index.pth")
    loaded = TextIndex.load(path)
    assert loaded.families == index.families
    assert loaded.axis_names == index.axis_names
    assert loaded.rankable == [0, 2]
    assert torch.equal(loaded.rankable_mask, index.rankable_mask)
    assert torch.allclose(loaded.w, index.w)
    assert torch.allclose(loaded.b, index.b)
    for family in FAMILIES:
        a = index.features_for_family(family)
        b = loaded.features_for_family(family)
        assert a is not None and b is not None
        assert torch.allclose(index.score(a), loaded.score(b))


def test_load_rejects_a_foreign_checkpoint(tmp_path) -> None:
    path = tmp_path / "not-an-index.pth"
    torch.save({"kind": "something-else", "version": 1}, path)
    try:
        TextIndex.load(path)
    except ValueError as e:
        assert "not a hrothgar text index" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


# ── Staleness ────────────────────────────────────────────────────────────────
def test_stale_when_an_input_is_newer(tmp_path) -> None:
    source = tmp_path / "judgements.csv"
    source.write_text("a\n")

    index = _toy_index()
    index.meta = {"created_mtime": 2000.0, "inputs": {str(source): 0.0}}

    os.utime(source, (1000, 1000))
    stale, reasons = index_is_stale(tmp_path / "index.pth", index=index)
    assert stale is False and reasons == []

    os.utime(source, (3000, 3000))
    stale, reasons = index_is_stale(tmp_path / "index.pth", index=index)
    assert stale is True
    assert "is newer than the index" in reasons[0]

    index.meta["created_mtime"] = 4000.0
    stale, reasons = index_is_stale(tmp_path / "index.pth", index=index)
    assert stale is False and reasons == []


def test_missing_supervision_file_is_not_stale(tmp_path) -> None:
    index = _toy_index()
    index.meta = {
        "created_mtime": 2000.0,
        "inputs": {str(tmp_path / "gone.csv"): 1000.0},
    }
    stale, reasons = index_is_stale(tmp_path / "index.pth", index=index)
    assert stale is False and reasons == []
