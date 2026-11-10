"""Tests for the cluster-proposal merge tool."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.similarity.merge_cluster_proposals import (  # noqa: E402
    canonical_shown,
    consensus,
    group_judgements,
    make_block,
    merge_blocks,
    parse_rows,
    select_voters,
)


def _row(
    *,
    round_id="p0001",
    anchor="Anchor",
    shown=("Anchor", "A", "B", "C", "D"),
    include=("Anchor", "A", "B"),
    exclude=("C", "D"),
    browser_id="browser-1",
    session_id="session-1",
    created_at="2026-11-06T10:00:00+00:00",
    position=0,
    **extra,
):
    return {
        "id": f"{round_id}-{browser_id}-{position}",
        "round_id": round_id,
        "anchor": anchor,
        "shown": json.dumps(list(shown)),
        "block": json.dumps({"include": list(include), "exclude": list(exclude)}),
        "browser_id": browser_id,
        "session_id": session_id,
        "created_at": created_at,
        "position": str(position),
        "skipped": "false",
        "no_clear_answer": "false",
        "anchor_outlier": "false",
        **extra,
    }


def test_parse_rows_drops_unusable_answers() -> None:
    rows = [
        _row(),
        _row(skipped="true"),
        _row(no_clear_answer="true"),
        _row(anchor_outlier="true"),
        _row(
            include=("Anchor", "A"),
            exclude=("B", "C"),  # does not partition shown: D is missing
        ),
        _row(include=("A", "B", "C", "D"), exclude=("Anchor",)),  # anchor omitted
        _row(block="{not json"),
    ]
    judgements, report = parse_rows(rows)

    assert report.total == 7
    assert report.valid == 1
    assert report.drops["skipped"] == 1
    assert report.drops["unjudgeable-only"] == 1
    assert report.drops["anchor-outlier"] == 1
    assert report.drops["not-a-partition"] == 1
    assert report.drops["anchor-not-included"] == 1
    assert report.drops["bad-block"] == 1
    assert judgements[0].include == ("Anchor", "A", "B")


def test_parse_rows_accepts_include_exclude_columns() -> None:
    row = _row()
    row.pop("block")
    row["include"] = '["Anchor", "A", "B"]'
    row["exclude"] = '["C", "D"]'
    judgements, report = parse_rows([row])
    assert report.valid == 1
    assert judgements[0].include == ("Anchor", "A", "B")


def test_majority_leaves_disagreement_neutral() -> None:
    rows = [
        _row(
            browser_id="b1",
            include=("Anchor", "A", "B", "C", "D"),
            exclude=(),
        ),
        _row(
            browser_id="b2",
            include=("Anchor", "A", "B"),
            exclude=("C", "D"),
        ),
    ]
    judgements, _ = parse_rows(rows)
    group = group_judgements(judgements)[("p0001", "Anchor")]
    shown = canonical_shown(group)
    voters, mismatches = select_voters(
        group, policy="majority", vote_unit="browser", shown=shown
    )
    include, exclude, ties = consensus(
        voters, shown=shown, anchor="Anchor", policy="majority"
    )

    assert mismatches == 0
    assert include == ("Anchor", "A", "B")
    assert exclude == ()
    assert ties == ("C", "D")


def test_browser_vote_uses_latest_valid_judgement() -> None:
    rows = [
        _row(
            created_at="2026-11-06T10:00:00+00:00",
            include=("Anchor", "A", "B", "C", "D"),
            exclude=(),
        ),
        _row(
            created_at="2026-11-06T11:00:00+00:00",
            include=("Anchor", "A"),
            exclude=("B", "C", "D"),
        ),
    ]
    judgements, _ = parse_rows(rows)
    group = group_judgements(judgements)[("p0001", "Anchor")]
    voters, _ = select_voters(
        group, policy="majority", vote_unit="browser", shown=canonical_shown(group)
    )

    assert len(voters) == 1
    assert voters[0].include == ("Anchor", "A")
    assert voters[0].exclude == ("B", "C", "D")


def test_merge_replaces_old_generated_block_and_keeps_hand_written() -> None:
    hand = {"include": ["Hand", "Written"], "exclude": ["No"]}
    old_generated = {
        "include": ["Anchor", "A"],
        "exclude": ["B", "C", "D"],
        "note": "proposal game p0001 (anchor Anchor, browser 12345678)",
    }
    duplicate = {**old_generated}
    new = make_block(
        round_id="p0001",
        anchor="Anchor",
        include=("Anchor", "A", "B"),
        exclude=("C", "D"),
        ties=(),
        voters=2,
        judgements=2,
        policy="majority",
        vote_unit="browser",
    )

    merged, stats = merge_blocks([hand, old_generated, duplicate], [new])

    assert merged[0] == hand
    assert merged[1] == new
    assert len(merged) == 2
    assert stats["replaced"] == 1
    assert stats["generated-removed"] == 1
    assert stats["hand-written"] == 1
    assert stats["added"] == 0


def test_make_block_note_records_ties_and_provenance() -> None:
    block = make_block(
        round_id="p0002",
        anchor="Anchor",
        include=("Anchor", "A"),
        exclude=("B",),
        ties=("C", "D"),
        voters=2,
        judgements=3,
        policy="majority",
        vote_unit="browser",
    )
    assert block["round_id"] == "p0002"
    assert block["anchor"] == "Anchor"
    assert "2 voters" in block["note"]
    assert "3 judgements" in block["note"]
    assert "ties omitted: C, D" in block["note"]
