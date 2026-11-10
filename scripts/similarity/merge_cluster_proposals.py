#!/usr/bin/env python3
"""Merge Supabase cluster-proposal judgements into similarities.json.

The review app writes one row per judgement. A round can be judged by several
browsers/sessions, and the same browser/session can meet the same round more
than once (the pool is sampled with replacement). This tool turns those rows
into one or more similarities.json-shaped blocks.

Why voting, when the clusters are "soft hints"?  Because the model's conflict
resolution is not soft.  ``build_supervision`` turns every include-include pair
into a positive and every include-exclude pair into a negative, then:

    for k in neg:
        if k in pos or (k[1], k[0]) in pos:
            del neg[k]

A single inclusion therefore beats any number of exclusions. Appending every
judgement block would let one dissenting reviewer force a positive pair. Instead
this tool collapses each round to a consensus block and leaves disputed faces
*unassigned* (neither include nor exclude), which is the model's genuinely soft
state.

Policies:

* ``majority`` (default) — a face is included if more voters keep it than mark
  it, excluded if more mark it, and omitted on a tie.
* ``unanimous`` — include only if every voter keeps it, exclude only if every
  voter marks it, otherwise omit.
* ``latest`` — no vote: use the most recent valid judgement for the round.

Vote units:

* ``browser`` (default) — one vote per browser per round, using its latest valid
  judgement. Repeated exposure in one session cannot stack.
* ``session`` — one vote per session per round, using its latest valid judgement.
* ``judgement`` — every valid row is a vote (useful for comparison, but repeated
  judgements from one person are not independent).

Usage::

    python scripts/similarity/merge_cluster_proposals.py \\
        --csv cluster_proposals_rows.csv \\
        --list similarities.json

The merge is idempotent: generated blocks carry ``round_id`` and ``anchor``, and
re-running replaces the generated block for that round rather than appending a
duplicate. Hand-written blocks are never touched.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

POLICIES = ("majority", "unanimous", "latest")
VOTE_UNITS = ("browser", "session", "judgement")

ROUND_NOTE_RE = re.compile(r"\bproposal game (p\d+)\b")
ANCHOR_NOTE_RE = re.compile(r"\(anchor ([^;,)]+)")


# ── Rows ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Judgement:
    round_id: str
    anchor: str
    shown: tuple[str, ...]
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    browser_id: str
    session_id: str
    created_at: str
    position: int
    row_id: str


@dataclass
class ParseReport:
    total: int = 0
    valid: int = 0
    drops: Counter = field(default_factory=Counter)

    def drop(self, reason: str) -> None:
        self.drops[reason] += 1


def _bool(value: object) -> bool:
    return str(value or "").strip().lower() in {"true", "t", "1", "yes"}


def _json_list(value: object) -> tuple[str, ...] | None:
    """Parse a JSON array of strings from a CSV cell; ``None`` if malformed."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        return None
    return tuple(data)


def _timestamp(value: str) -> float:
    """Best-effort timestamp for 'latest'. Invalid/empty sorts oldest."""
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return 0.0


def _latest_key(judgement: Judgement) -> tuple[float, int, str]:
    return (_timestamp(judgement.created_at), judgement.position, judgement.row_id)


def parse_rows(rows: Iterable[dict[str, str]]) -> tuple[list[Judgement], ParseReport]:
    """Validate rows and keep only real, anchored, partitioning judgements."""
    report = ParseReport()
    judgements: list[Judgement] = []

    for row in rows:
        report.total += 1
        if _bool(row.get("skipped")):
            report.drop("skipped")
            continue
        if _bool(row.get("no_clear_answer")):
            report.drop("unjudgeable-only")
            continue
        # Rows from the short-lived app version where the anchor was selectable.
        if _bool(row.get("anchor_outlier")):
            report.drop("anchor-outlier")
            continue

        round_id = str(row.get("round_id") or "").strip()
        anchor = str(row.get("anchor") or "").strip()
        browser_id = str(row.get("browser_id") or "").strip()
        session_id = str(row.get("session_id") or "").strip()
        if not (round_id and anchor and browser_id and session_id):
            report.drop("missing-fields")
            continue

        shown = _json_list(row.get("shown"))
        if shown is None or len(set(shown)) != len(shown):
            report.drop("bad-shown")
            continue

        try:
            block = json.loads(str(row.get("block") or ""))
        except json.JSONDecodeError:
            # Query-1 exports have include/exclude columns instead of `block`.
            block = None
        if block is None and row.get("include") is not None:
            include_list = _json_list(row.get("include"))
            exclude_list = _json_list(row.get("exclude"))
            if include_list is not None and exclude_list is not None:
                block = {"include": list(include_list), "exclude": list(exclude_list)}
        if not isinstance(block, dict):
            report.drop("bad-block")
            continue
        include = block.get("include")
        exclude = block.get("exclude")
        if not isinstance(include, list) or not isinstance(exclude, list):
            report.drop("bad-block")
            continue
        include = tuple(str(item) for item in include)
        exclude = tuple(str(item) for item in exclude)

        shown_set = set(shown)
        include_set = set(include)
        exclude_set = set(exclude)
        if len(include_set) != len(include) or len(exclude_set) != len(exclude):
            report.drop("duplicate-names")
            continue
        if include_set & exclude_set:
            report.drop("overlap")
            continue
        if not include_set <= shown_set or not exclude_set <= shown_set:
            report.drop("names-not-shown")
            continue
        if include_set | exclude_set != shown_set:
            report.drop("not-a-partition")
            continue
        if anchor not in shown_set:
            report.drop("anchor-not-shown")
            continue
        if anchor not in include_set:
            report.drop("anchor-not-included")
            continue

        try:
            position = int(row.get("position") or 0)
        except (TypeError, ValueError):
            position = 0

        judgements.append(
            Judgement(
                round_id=round_id,
                anchor=anchor,
                shown=tuple(shown),
                include=include,
                exclude=exclude,
                browser_id=browser_id,
                session_id=session_id,
                created_at=str(row.get("created_at") or ""),
                position=position,
                row_id=str(row.get("id") or ""),
            )
        )
        report.valid += 1

    return judgements, report


# ── Voting ──────────────────────────────────────────────────────────────────


def group_judgements(
    judgements: Sequence[Judgement],
) -> dict[tuple[str, str], list[Judgement]]:
    groups: dict[tuple[str, str], list[Judgement]] = defaultdict(list)
    for judgement in judgements:
        groups[(judgement.round_id, judgement.anchor)].append(judgement)
    return dict(groups)


def canonical_shown(group: Sequence[Judgement]) -> tuple[str, ...]:
    """The face set/order for a round.

    The design is frozen, but old and new app versions can order ``shown``
    differently (the anchor used to sit at a random position).  Use the latest
    valid row's order, and only include voters that saw the same face set.
    """
    latest = max(group, key=_latest_key)
    return latest.shown


def select_voters(
    group: Sequence[Judgement],
    *,
    policy: str,
    vote_unit: str,
    shown: Sequence[str],
) -> tuple[list[Judgement], int]:
    """Return the judgements that count as votes and the number of mismatches."""
    shown_set = set(shown)
    same_design = [j for j in group if set(j.shown) == shown_set]
    mismatches = len(group) - len(same_design)

    if policy == "latest":
        return [max(same_design, key=_latest_key)], mismatches
    if vote_unit == "judgement":
        return same_design, mismatches

    key_attr = "browser_id" if vote_unit == "browser" else "session_id"
    latest: dict[str, Judgement] = {}
    for judgement in same_design:
        key = getattr(judgement, key_attr)
        current = latest.get(key)
        if current is None or _latest_key(judgement) > _latest_key(current):
            latest[key] = judgement
    return list(latest.values()), mismatches


def consensus(
    voters: Sequence[Judgement],
    *,
    shown: Sequence[str],
    anchor: str,
    policy: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return ``(include, exclude, ties)`` for one round.

    ``ties`` are omitted from both sides; the model leaves unassigned pairs
    neutral.  The anchor is always included.
    """
    if policy == "latest":
        judgement = voters[0]
        include = (anchor,) + tuple(
            face for face in shown if face != anchor and face in judgement.include
        )
        exclude = tuple(
            face for face in shown if face != anchor and face in judgement.exclude
        )
        return include, exclude, ()

    voters_n = len(voters)
    keep = Counter()
    drop = Counter()
    for judgement in voters:
        keep.update(judgement.include)
        drop.update(judgement.exclude)

    include: list[str] = [anchor]
    exclude: list[str] = []
    ties: list[str] = []
    for face in shown:
        if face == anchor:
            continue
        kept = keep[face]
        dropped = drop[face]
        if policy == "majority":
            if kept > dropped:
                include.append(face)
            elif dropped > kept:
                exclude.append(face)
            else:
                ties.append(face)
        elif policy == "unanimous":
            if kept == voters_n:
                include.append(face)
            elif dropped == voters_n:
                exclude.append(face)
            else:
                ties.append(face)
        else:  # pragma: no cover - argparse restricts this
            raise ValueError(f"unknown policy {policy!r}")

    return tuple(include), tuple(exclude), tuple(ties)


def make_block(
    *,
    round_id: str,
    anchor: str,
    include: Sequence[str],
    exclude: Sequence[str],
    ties: Sequence[str],
    voters: int,
    judgements: int,
    policy: str,
    vote_unit: str,
) -> dict:
    if policy == "latest":
        note = f"proposal game {round_id} (anchor {anchor}; latest judgement)"
    else:
        note = (
            f"proposal game {round_id} (anchor {anchor}; "
            f"{voters} voter{'s' if voters != 1 else ''}, "
            f"{judgements} judgement{'s' if judgements != 1 else ''}; "
            f"{policy}, {vote_unit})"
        )
    if ties:
        note += f"; ties omitted: {', '.join(ties)}"
    return {
        "include": list(include),
        "exclude": list(exclude),
        "round_id": round_id,
        "anchor": anchor,
        "note": note,
    }


# ── Merging ─────────────────────────────────────────────────────────────────


def _block_round_id(block: dict) -> str | None:
    value = block.get("round_id")
    if isinstance(value, str) and value:
        return value
    match = ROUND_NOTE_RE.search(str(block.get("note", "")))
    return match.group(1) if match else None


def _block_anchor(block: dict) -> str | None:
    """The round's anchor, from the block or the old SQL export note."""
    value = block.get("anchor")
    if isinstance(value, str) and value:
        return value
    match = ANCHOR_NOTE_RE.search(str(block.get("note", "")))
    return match.group(1).strip() if match else None


def merge_blocks(
    existing: Sequence[dict], new_blocks: Sequence[dict]
) -> tuple[list[dict], Counter]:
    """Replace generated blocks in place; preserve hand-written ones.

    Generated blocks are identified by ``(round_id, anchor)``.  The anchor is
    part of the key so a regenerated design that reuses a round id for a
    different cell does not overwrite the old cell's block.  Old SQL exports
    carry the round id and anchor in their note; those are parsed.

    If a round has several old generated blocks (one per judgement), the first
    is replaced and the rest are removed.
    """
    by_key: dict[tuple[str, str], dict] = {}
    for block in new_blocks:
        key = (block["round_id"], block["anchor"])
        if key in by_key:
            raise ValueError(f"duplicate new block for {key[0]} / {key[1]}")
        by_key[key] = block

    stats: Counter = Counter()
    used: set[tuple[str, str]] = set()
    merged: list[dict] = []
    for block in existing:
        round_id = _block_round_id(block)
        if round_id is None:
            merged.append(block)
            stats["hand-written"] += 1
            continue
        key = (round_id, _block_anchor(block))
        if key not in by_key and key[1] is None:
            # A generated block with a round id but no parseable anchor: if the
            # round maps to exactly one new cell, treat it as that cell rather
            # than leaving a duplicate behind.
            candidates = [k for k in by_key if k[0] == round_id]
            if len(candidates) == 1:
                key = candidates[0]
        if key not in by_key:
            merged.append(block)
            stats["generated-kept"] += 1
            continue
        if key in used:
            stats["generated-removed"] += 1
            continue
        merged.append(by_key[key])
        used.add(key)
        stats["replaced"] += 1

    for key in sorted(set(by_key) - used):
        merged.append(by_key[key])
        stats["added"] += 1

    return merged, stats


# ── Entry point ─────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("cluster_proposals_rows.csv"),
        help="Supabase CSV export of cluster_proposals",
    )
    parser.add_argument(
        "--list", type=Path, default=Path("data/similarity_supervision/similarities.json"), help="cluster JSON"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output JSON (default: overwrite --list)",
    )
    parser.add_argument(
        "--policy",
        choices=POLICIES,
        default="majority",
        help="how to resolve multiple judgements (default: majority)",
    )
    parser.add_argument(
        "--vote-unit",
        choices=VOTE_UNITS,
        default="browser",
        help="what counts as one vote (default: browser)",
    )
    parser.add_argument(
        "--min-voters",
        type=int,
        default=1,
        help="skip rounds with fewer than this many voters (default: 1)",
    )
    parser.add_argument(
        "--min-include",
        type=int,
        default=2,
        help="skip blocks with fewer than this many included faces "
        "(default: 2; use 1 to keep anchor-only negative blocks)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report without writing"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="do not write similarities.json.bak before an in-place write",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        parser.error(f"CSV not found: {args.csv}")
    if not args.list.exists():
        parser.error(f"cluster JSON not found: {args.list}")
    if args.min_voters < 1:
        parser.error("--min-voters must be at least 1")
    if args.min_include < 1:
        parser.error("--min-include must be at least 1")

    with args.csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {"round_id", "anchor", "shown", "browser_id", "session_id"}
        missing = required - fieldnames
        if missing:
            parser.error(f"CSV is missing columns: {', '.join(sorted(missing))}")
        if "block" not in fieldnames and not {"include", "exclude"} <= fieldnames:
            parser.error("CSV needs a `block` column or `include`/`exclude` columns")
        rows = list(reader)

    judgements, parse_report = parse_rows(rows)
    groups = group_judgements(judgements)

    new_blocks: list[dict] = []
    skipped: Counter = Counter()
    ties_total = 0
    voter_histogram: Counter = Counter()
    mismatched = 0

    for (round_id, anchor), group in sorted(groups.items()):
        shown = canonical_shown(group)
        voters, group_mismatch = select_voters(
            group, policy=args.policy, vote_unit=args.vote_unit, shown=shown
        )
        mismatched += group_mismatch
        voter_histogram[len(voters)] += 1

        if len(voters) < args.min_voters:
            skipped["min-voters"] += 1
            continue

        include, exclude, ties = consensus(
            voters, shown=shown, anchor=anchor, policy=args.policy
        )
        if len(include) < args.min_include:
            skipped["include-too-small"] += 1
            continue

        new_blocks.append(
            make_block(
                round_id=round_id,
                anchor=anchor,
                include=include,
                exclude=exclude,
                ties=ties,
                voters=len(voters),
                judgements=len(group),
                policy=args.policy,
                vote_unit=args.vote_unit,
            )
        )
        ties_total += len(ties)

    existing = json.loads(args.list.read_text())
    if not isinstance(existing, list):
        parser.error(f"{args.list} must contain a JSON array of blocks")

    merged, merge_stats = merge_blocks(existing, new_blocks)

    print(f"CSV: {args.csv} ({parse_report.total} rows)")
    print(f"  valid: {parse_report.valid}")
    for reason, count in parse_report.drops.most_common():
        print(f"  dropped ({reason}): {count}")
    print(
        f"Rounds: {len(groups)}  "
        f"voters per round: "
        + ", ".join(f"{n}→{c}" for n, c in sorted(voter_histogram.items()))
    )
    print(
        f"Policy: {args.policy}; vote unit: "
        f"{'latest' if args.policy == 'latest' else args.vote_unit}; "
        f"min voters: {args.min_voters}"
    )
    if args.policy != "latest" and args.vote_unit != "judgement":
        print(
            "  (repeated judgements from one browser/session collapse to the "
            "latest valid one)"
        )
    if mismatched:
        print(f"  skipped voters with a different shown set: {mismatched}")
    if skipped:
        print(
            "Skipped rounds: "
            + ", ".join(f"{reason}={count}" for reason, count in skipped.items())
        )
    print(
        f"Blocks: {len(new_blocks)}  "
        f"added={merge_stats['added']} replaced={merge_stats['replaced']} "
        f"removed-old={merge_stats['generated-removed']} "
        f"kept-generated={merge_stats['generated-kept']} "
        f"hand-written={merge_stats['hand-written']}"
    )
    if ties_total:
        print(f"Ties omitted (left neutral): {ties_total} face(s)")

    output = args.output or args.list
    if args.dry_run:
        print(f"Dry run: would write {len(merged)} blocks to {output}")
        return 0

    if output.resolve() == args.list.resolve() and not args.no_backup:
        backup = args.list.with_name(args.list.name + ".bak")
        shutil.copy2(args.list, backup)
        print(f"Backup: {backup}")

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    tmp.write_text(json.dumps(merged, indent=2) + "\n")
    os.replace(tmp, output)
    print(f"Wrote {len(merged)} blocks to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
