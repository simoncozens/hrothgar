"""Text-arm similarity: top-k similar fonts for a given font.

Given a font — either a Google Fonts family name (``--font``) or a local TTF
(``--ttf``) — returns the top-k most similar fonts from the *text* arm of the
library, using the per-axis combiner (:mod:`hrothgar.similarity.axis_combiner`)
with auto-extractable features: fontquant geometry quantifiers plus the small
feature recognisers.

By default this queries a **frozen index** (:mod:`hrothgar.similarity.text_index`)
that ships with the package: the library's per-font positions and the learned
combiner weights.  A query costs one ``fontquant`` run plus four glyph
recognisers, and needs no Google Fonts checkout.

Building/refreshing the index is the expensive path::

    # rebuild from supervision data and write the packaged index
    python -m hrothgar.similarity.text --rebuild-index --repo ~/others-repos/fonts

    # rebuild only if the supervision data or measured library is newer
    python -m hrothgar.similarity.text --refresh-if-stale --repo ~/others-repos/fonts

Query::

    python -m hrothgar.similarity.text --font Inter
    python -m hrothgar.similarity.text --ttf path/to/Font-Regular.ttf --k 10

Use ``--live`` to ignore the frozen index and redo the whole pipeline in-process
(the pre-freezing behaviour), which is useful for verifying the index.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hrothgar.cedars import load_cedars
from hrothgar.googlefonts import StandaloneFont, families_to_fonts
from hrothgar.similarity.axis_combiner import (
    _load_recognisers,
    _pair_weight_matrices,
    _score_matrix,
    _train_combiner,
    build_similarity_tensor,
    font_feature_summary,
)
from hrothgar.similarity.supervision import (
    JUDGEMENTS_PATH,
    SIMILARITIES_PATH,
    build_supervision,
    text_half_supervision,
)
from hrothgar.utils import device
from hrothgar.similarity.text_index import (
    DEFAULT_INDEX_PATH,
    TextIndex,
    anchor_features,
    build_index,
    default_repo,
    index_is_stale,
)


def _resolve_anchor(name: str | None, ttf: str | None, repo: str):
    """Return a ``Font`` for the anchor (a GF regular instance or a standalone TTF)."""
    if ttf:
        return StandaloneFont(ttf)
    fam2font = families_to_fonts(repo)
    return fam2font.get(name.strip().lower())


def _display_name(ttf: str) -> str:
    """The font's typographic family name, falling back to the file stem."""
    try:
        from fontTools.ttLib import TTFont

        font = TTFont(ttf, lazy=True)
        try:
            name = font["name"].getDebugName(16) or font["name"].getDebugName(1)
        finally:
            font.close()
        return name or Path(ttf).stem
    except Exception:  # malformed/unreadable name table: the stem is good enough
        return Path(ttf).stem


def _print_ranking(index: TextIndex, af, k: int) -> None:
    results = index.rank(af, k=k)
    total = int(index.rankable_mask.sum())
    if af.family is not None and index.family_index(af.family) is not None:
        if bool(index.rankable_mask[index.family_index(af.family)]):
            total -= 1
    print(f"\nTop {len(results)} text-arm matches for {af.label!r} "
          f"(against {total} library fonts):")
    for rank, (family, score) in enumerate(results):
        print(f"  {rank + 1:2d}. {family:<32s} score={score:.4f}")


def _row_breakdown(S_row: torch.Tensor, names: list[str], w: torch.Tensor) -> list[tuple[str, float, float, float]]:
    """``(axis, similarity, weight, contribution)`` for one anchor row."""
    contrib = S_row * w
    rows = [
        (names[k], float(S_row[k]), float(w[k]), float(contrib[k]))
        for k in range(len(names))
    ]
    rows.sort(key=lambda r: -abs(r[3]))
    return rows


def _explain(index: TextIndex, af, k: int, explain_fonts: str | None) -> None:
    S = index.similarity_row(af)  # (N, K)
    scores = S @ index.w + index.b

    print(f"\n===== anchor {af.label!r} =====")
    for axis, val in index.summary(af).items():
        print(f"  {axis:<18} {val}")

    if explain_fonts:
        targets = [t.strip() for t in explain_fonts.split(",") if t.strip()]
        missing = [t for t in targets if index.family_index(t) is None]
        if missing:
            print(f"\n(warning: not in the library: {missing})")
        targets = [t for t in targets if index.family_index(t) is not None]
    else:
        order = torch.argsort(index.rank_scores(af), descending=True)
        targets = [index.families[int(i)] for i in order[:k]]

    for family in targets:
        i = index.family_index(family)
        other = index.features_for_family(family)
        if i is None or other is None:  # pragma: no cover - guarded above
            continue
        print(f"\n===== {family!r} score={float(scores[i]):.4f} =====")
        for axis, val in index.summary(other).items():
            print(f"  {axis:<18} {val}")
        print("  --- per-axis contribution vs anchor (sorted by |contrib|) ---")
        for name, s_ij, w_k, c in _row_breakdown(S[i], index.axis_names, index.w):
            print(f"    {name:<18} sim={s_ij:+.3f} weight={w_k:+.3f} contrib={c:+.3f}")
        print(f"    {'<bias>':<18} {'':>10} {'':>8} {float(index.b):+.3f}")


def _run_frozen(args, p: argparse.ArgumentParser) -> int:
    index_path = Path(args.index or DEFAULT_INDEX_PATH)

    built: TextIndex | None = None
    rebuild = args.rebuild_index
    if not rebuild and args.refresh_if_stale:
        stale, reasons = index_is_stale(index_path)
        if stale:
            print("text index is stale:")
            for reason in reasons:
                print(f"  - {reason}")
            rebuild = True
    if rebuild:
        repo = args.repo or default_repo()
        if not repo:
            p.error("--repo (or $GOOGLE_FONTS_REPO) is required to rebuild the index")
        print(f"rebuilding text index from {repo} ...", flush=True)
        built = build_index(
            repo,
            args.list,
            args.judgements,
            combiner=args.combiner,
            steps=args.steps,
            lr=args.lr,
            margin=args.margin,
            gamma=args.gamma,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )
        out = Path(args.index_out or index_path)
        print(
            f"wrote {built.save(out)} ({len(built)} families, "
            f"{int(built.rankable_mask.sum())} in the text arm, "
            f"{len(built.axis_names)} axes)"
        )

    index = built if built is not None else TextIndex.load(index_path)

    # ── Resolve the anchor ────────────────────────────────────────────────────
    if args.font:
        af = index.features_for_family(args.font)
        if af is None:
            repo = args.repo or default_repo()
            if not repo:
                print(
                    f"font {args.font!r} is not in the frozen text-arm index; "
                    "pass --ttf, or --repo to quantify it from the library",
                    flush=True,
                )
                return 1
            anchor = _resolve_anchor(args.font, None, repo)
            if anchor is None:
                print(f"font {args.font!r} not found in the library", flush=True)
                return 1
            af = anchor_features(anchor, index, _load_recognisers())
    else:
        anchor = StandaloneFont(args.ttf)
        af = anchor_features(anchor, index, _load_recognisers())
        af.label = _display_name(args.ttf)

    _print_ranking(index, af, args.k)

    if args.explain:
        _explain(index, af, args.k, args.explain_fonts)
    return 0


def _run_live(args, p: argparse.ArgumentParser) -> int:
    """The pre-freezing pipeline: build, train and query in one process."""
    repo = args.repo or default_repo()
    if not repo:
        p.error("--repo (or $GOOGLE_FONTS_REPO) is required for --live")

    dev = device(args.device)

    # ── Text-arm library universe (full library, then text-like half) ─────────
    sup = build_supervision(repo, args.list, args.judgements, universe="full")
    sup = text_half_supervision(sup)

    # ── Resolve the anchor and place it in the universe ────────────────────────
    anchor = _resolve_anchor(args.font, args.ttf, repo)
    if anchor is None:
        print(f"font {args.font!r} not found in the library", flush=True)
        return 1

    anchor_idx = None
    for i, fam in enumerate(sup.families):
        if fam.lower() == anchor.family.lower():
            anchor_idx = i
            break
    if anchor_idx is None:
        anchor_idx = len(sup.fonts)
        sup.fonts.append(anchor)
        sup.families.append(anchor.family)
        sup.font_ids.append(str(anchor.path))

    # ── Build the similarity tensor (anchor included) and train the combiner ──
    S, names = build_similarity_tensor(sup, feature_source="recogniser")
    S = S.to(dev)
    pos, neg = _pair_weight_matrices(sup, dev)
    w, b = _train_combiner(
        S, pos, neg, combiner=args.combiner, steps=args.steps, lr=args.lr,
        margin=args.margin, gamma=args.gamma, seed=args.seed,
        weight_decay=args.weight_decay,
    )
    sim = _score_matrix(S, w, b)

    # ── Rank the anchor against the library (exclude the anchor itself) ───────
    anchor_scores = sim[anchor_idx].clone()
    anchor_scores[anchor_idx] = float("-inf")
    order = torch.argsort(anchor_scores, descending=True)

    k = min(args.k, len(sup.fonts) - 1)
    print(f"\nTop {k} text-arm matches for {anchor.family!r} "
          f"(against {len(sup.fonts) - 1} library fonts):")
    for rank in range(k):
        idx = int(order[rank])
        print(f"  {rank + 1:2d}. {sup.families[idx]:<32s} score={anchor_scores[idx].item():.4f}")

    # ── Explain the inputs / per-axis contributions (debug) ────────────────────
    if args.explain:
        cedars = load_cedars()
        recognisers = _load_recognisers()
        fam_to_idx = {f.lower(): i for i, f in enumerate(sup.families)}

        if args.explain_fonts:
            targets = [t.strip().lower() for t in args.explain_fonts.split(",") if t.strip()]
            explain_idxs, missing = [], []
            for t in targets:
                if t in fam_to_idx:
                    explain_idxs.append(fam_to_idx[t])
                else:
                    missing.append(t)
            if missing:
                print(f"\n(warning: not in text-half universe: {missing}")
        else:
            explain_idxs = [int(order[r]) for r in range(k)]

        print(f"\n===== anchor {anchor.family!r} ({anchor.path.name}) =====")
        for axis, val in font_feature_summary(anchor, cedars, recognisers).items():
            print(f"  {axis:<18} {val}")

        for idx in explain_idxs:
            print(f"\n===== {sup.families[idx]!r} ({Path(sup.fonts[idx].path).name}) "
                  f"score={sim[anchor_idx, idx].item():.4f} =====")
            for axis, val in font_feature_summary(sup.fonts[idx], cedars, recognisers).items():
                print(f"  {axis:<18} {val}")
            print("  --- per-axis contribution vs anchor (sorted by |contrib|) ---")
            contrib = S[anchor_idx, idx] * w
            rows = [
                (names[k_i], float(S[anchor_idx, idx][k_i]), float(w[k_i]), float(contrib[k_i]))
                for k_i in range(len(names))
            ]
            rows.sort(key=lambda r: -abs(r[3]))
            for name, s_ij, w_k, c in rows:
                print(f"    {name:<18} sim={s_ij:+.3f} weight={w_k:+.3f} contrib={c:+.3f}")
            print(f"    {'<bias>':<18} {'':>10} {'':>8} {b.item():+.3f}")

    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--font", help="Google Fonts family name to rank against")
    p.add_argument("--ttf", help="local TTF to rank against")
    p.add_argument("--k", type=int, default=20, help="number of results to return")
    p.add_argument("--repo", default=default_repo(), help="Google Fonts checkout (rebuild/--live only)")
    p.add_argument("--list", default=SIMILARITIES_PATH, help="supervision: agreed similarities")
    p.add_argument("--judgements", default=JUDGEMENTS_PATH, help="supervision: expert judgements")
    p.add_argument(
        "--index",
        default=None,
        help=f"frozen index to query (default: the packaged {Path(DEFAULT_INDEX_PATH).name})",
    )
    p.add_argument(
        "--rebuild-index",
        action="store_true",
        help="recompute the frozen index from supervision data, then query it",
    )
    p.add_argument(
        "--refresh-if-stale",
        action="store_true",
        help="recompute the frozen index first if the supervision data is newer than it",
    )
    p.add_argument("--index-out", default=None, help="where --rebuild-index writes (default: --index)")
    p.add_argument(
        "--live",
        action="store_true",
        help="ignore the frozen index; rebuild and retrain in-process (slow; for verification)",
    )
    p.add_argument(
        "--combiner",
        choices=("linear", "softmax"),
        default="linear",
        help="how the per-axis similarities are combined (linear = signed weights, "
        "softmax = non-negative convex combination)",
    )
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--margin", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=128.0)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="--live only; the frozen index scores on CPU")
    p.add_argument(
        "--explain",
        action="store_true",
        help="dump the raw feature inputs and per-axis contribution for the anchor "
        "and the explained fonts (default: the top-k results)",
    )
    p.add_argument(
        "--explain-fonts",
        help="comma-separated family names to explain instead of the top-k "
        "(e.g. 'Roboto Flex,Vazirmatn,Yantramanav'); useful for tracing why a "
        "particular font does *not* rank",
    )
    args = p.parse_args()

    if not args.font and not args.ttf:
        p.error("one of --font or --ttf is required")

    if args.live:
        return _run_live(args, p)
    return _run_frozen(args, p)


if __name__ == "__main__":
    raise SystemExit(main())
