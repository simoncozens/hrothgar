"""Tests for the proposal-game round builder.

The model-backed path needs a Google Fonts checkout and is exercised by running
the builder itself; these tests cover the pure policy pieces: which families
count as positive supervision, which neighbours can fill a round, and how the
anchor's superfamily is excluded.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.similarity.build_proposal_rounds import (  # noqa: E402
    TOP_K,
    build_rounds,
    load_excluded_families,
    load_positive_families,
    propose,
)


def _write_similarities(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {"include": ["Alpha", "Beta"], "exclude": ["Gamma"]},
                {"include": ["Delta"], "exclude": ["Epsilon"]},
            ]
        )
    )


def _write_judgements(path: Path) -> None:
    fields = ["skipped", "no_clear_answer", "none_alike", "positive"]
    rows = [
        {"skipped": "true", "no_clear_answer": "false", "none_alike": "false", "positive": '["Ignored"]'},
        {"skipped": "false", "no_clear_answer": "true", "none_alike": "false", "positive": '["AlsoIgnored"]'},
        {"skipped": "false", "no_clear_answer": "false", "none_alike": "true", "positive": "[]"},
        {"skipped": "false", "no_clear_answer": "false", "none_alike": "false", "positive": '["Zeta", "Eta"]'},
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_positive_and_exclude_sources(tmp_path: Path) -> None:
    sim = tmp_path / "similarities.json"
    judge = tmp_path / "judgements.csv"
    _write_similarities(sim)
    _write_judgements(judge)

    includes, judgements = load_positive_families(sim, judge)
    assert includes == {"alpha", "beta", "delta"}
    # Skipped, unjudgeable and none-alike rows contribute no positives.
    assert judgements == {"zeta", "eta"}
    assert load_excluded_families(sim) == {"gamma", "epsilon"}


def test_propose_reserves_the_anchor_superfamily() -> None:
    families = [
        "Anchor",
        "Anchor Wide",   # same stem: never a candidate
        "Alpha",
        "Beta",
        "Gamma",
        "Delta",
        "Epsilon",
    ]
    scores = torch.tensor([1.0, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5])
    picked = propose(0, scores, families, list(range(len(families))))

    assert [family for family, _score in picked] == [
        "Alpha", "Beta", "Gamma", "Delta", "Epsilon",
    ]
    assert len(picked) == TOP_K
    assert "Anchor Wide" not in {family for family, _score in picked}


def test_propose_returns_short_when_pool_is_small() -> None:
    families = ["Anchor", "Alpha", "Beta"]
    scores = torch.tensor([1.0, 0.9, 0.8])
    picked = propose(0, scores, families, [0, 1, 2])
    assert [family for family, _score in picked] == ["Alpha", "Beta"]


def test_build_rounds_fixes_the_anchor_in_slot_zero() -> None:
    families = ["Anchor", "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]
    sim = torch.zeros(len(families), len(families))
    sim[0] = torch.tensor([0.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])

    rounds, _ = build_rounds(
        sim, families, set(families), ["Anchor"], seed=3
    )
    round_ = rounds["p0000"]
    assert round_["shown"][0] == "Anchor"
    assert round_["shown"][0] == round_["anchor"]
    assert len(round_["shown"]) == TOP_K + 1
    assert {c["family"] for c in round_["candidates"]} == set(round_["shown"]) - {
        "Anchor"
    }
