#!/usr/bin/env python
"""Measure how much the factorized diffusion model's conditioning tables are
being asked to represent.

Loads a trained ``(codepoint, font-id)`` diffusion checkpoint and its sidecars
(``.codepoints.json``, ``.instances.json``), then reports, for each learned
embedding table (``codepoint_emb``, ``family_emb``, ``style_emb``,
``italic_family_emb``):

  * **effective rank** (participation ratio ``(Σλ)² / Σλ²`` of the centred
    table's singular values) — how many independent directions the table is
    actually using.  ``PR ≈ min(N, D)`` means the table is *saturated* (every
    dimension is being worked); ``PR ≪ min(N, D)`` means the vectors have
    collapsed onto a low-dimensional subspace and the table has headroom.
  * **variance concentration** — how many components explain 50% / 90% / 99%
    of the variance.  A table that needs only 3 components to hit 90% is
    carrying far less than its width suggests.
  * **pairwise cosine geometry** — the distribution of angles between rows,
    plus the most collinear pairs.  Collinear *codepoint* rows mean two
    codepoints are being asked to represent near-identical shapes; collinear
    *family* rows mean two families have been squeezed into nearly the same
    style point.

The intuition being probed: "is the codepoint vector easy and the style vector
hard, or the other way round?"  Compare ``PR / min(N, D)`` across the codepoint
and family tables — the one closer to saturation is the branch under more
representational pressure.  (This is the *static*, post-hoc measurement; it does
not need the set-membership / content-detector models.)

This reads the four embedding tensors directly out of the checkpoint's state
dict (matched by key suffix), so it does not instantiate the UNet.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

# Keys are matched by suffix so any checkpoint prefix (e.g. a ``module.``
# DataParallel wrapper) still resolves.  The leading dot anchors the match to a
# full component name: ``.family_emb.weight`` must not also match
# ``italic_family_emb.weight`` (whose preceding component is ``italic_``).
TABLES = {
    "codepoint": ".codepoint_emb.weight",
    "family": ".family_emb.weight",
    "style": ".style_emb.weight",
    "italic_family": ".italic_family_emb.weight",
}


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_table(state_dict: dict, table: str) -> torch.Tensor:
    suffix = TABLES[table]
    matches = [k for k in state_dict if k.endswith(suffix)]
    if not matches:
        raise KeyError(f"No state-dict key ending in {suffix!r} (table {table!r})")
    if len(matches) > 1:
        raise KeyError(f"Ambiguous keys for {table!r}: {matches}")
    return state_dict[matches[0]]


def _codepoint_labels(codepoints: list[int]) -> list[str]:
    out = []
    for cp in codepoints:
        try:
            ch = chr(cp)
            if ch.isprintable() and not ch.isspace():
                out.append(f"{ch!r} U+{cp:04X}")
            else:
                out.append(f"U+{cp:04X}")
        except (ValueError, OverflowError):
            out.append(f"U+{cp:04X}")
    return out


def _family_labels(instances: list[dict], n_rows: int) -> list[str]:
    """family_id -> family name from the instances sidecar (first occurrence).

    Returns exactly ``n_rows`` labels, falling back to ``family_<id>`` for any
    family id present in the table but absent from the sidecar.
    """
    by_id: dict[int, str] = {}
    for inst in instances:
        fid = inst.get("family_id")
        name = inst.get("family")
        if fid is not None and name is not None and fid not in by_id:
            by_id[fid] = name
    return [by_id.get(i, f"family_{i}") for i in range(n_rows)]


def _effective_rank(weight: torch.Tensor) -> dict:
    """Participation ratio + variance concentration for a ``(N, D)`` table."""
    w = weight.detach().to(torch.float32).cpu()
    n, d = w.shape
    if not torch.isfinite(w).all():
        raise ValueError("table contains NaN/Inf")

    norms = w.norm(dim=1)
    dead = int((norms < 1e-6).sum().item())

    centred = w - w.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(centred)
    lam = s.square()
    total = lam.sum().item()

    if total <= 0:
        pr = 0.0
        cum = torch.zeros_like(lam)
    else:
        pr = (total * total) / (lam.square().sum().item())
        cum = torch.cumsum(lam, dim=0) / total

    def n_for(frac: float) -> int:
        hits = (cum >= frac).nonzero()
        if hits.numel() == 0:
            return int(cum.numel())
        return int(hits[0].item()) + 1

    return {
        "n": n,
        "dim": d,
        "rank_bound": min(n, d),
        "participation_ratio": pr,
        "fraction_of_capacity": pr / min(n, d) if min(n, d) > 0 else 0.0,
        "components_50": n_for(0.50),
        "components_90": n_for(0.90),
        "components_99": n_for(0.99),
        "row_norm": {
            "min": float(norms.min()),
            "median": float(norms.median()),
            "max": float(norms.max()),
        },
        "dead_rows": dead,
    }


def _cosine_geometry(
    weight: torch.Tensor, labels: list[str], *, top: int, threshold: float
) -> dict:
    """Pairwise cosine statistics and the most collinear pairs."""
    w = weight.detach().to(torch.float32).cpu()
    n = w.shape[0]
    if n < 2:
        return {"note": "fewer than 2 rows; skipped"}

    w = torch.nn.functional.normalize(w, dim=1)
    cos = w @ w.t()  # (N, N), diagonal = 1

    i_idx, j_idx = torch.triu_indices(n, n, offset=1)
    off = cos[i_idx, j_idx]
    abs_off = off.abs()

    def _frac(mask: torch.Tensor) -> float:
        return float(mask.float().mean()) if mask.numel() else 0.0

    k = min(top, off.numel())
    vals, order = off.topk(k)
    top_pairs = [
        {
            "cosine": float(v),
            "a": labels[int(i_idx[o])],
            "b": labels[int(j_idx[o])],
        }
        for v, o in zip(vals.tolist(), order.tolist())
        if float(v) >= threshold
    ]

    return {
        "n_pairs": int(off.numel()),
        "mean": float(off.mean()),
        "median": float(off.median()),
        "std": float(off.std(unbiased=True)),
        "min": float(off.min()),
        "max": float(off.max()),
        "frac_abs_below_0_1": _frac(abs_off < 0.1),
        "frac_above_0_5": _frac(off > 0.5),
        "frac_above_0_9": _frac(off > 0.9),
        "top_collinear": top_pairs,
    }


def _print_table(name: str, er: dict, geo: dict) -> None:
    print(f"\n=== {name}  (N x dim = {er['n']} x {er['dim']}) ===")
    if "note" in geo:
        print(f"  cosine geometry: {geo['note']}")
        return
    print(
        "  row norms: "
        f"min {er['row_norm']['min']:.3f}  "
        f"median {er['row_norm']['median']:.3f}  "
        f"max {er['row_norm']['max']:.3f}  "
        f"({er['dead_rows']} dead)"
    )
    print(
        "  effective rank: "
        f"{er['participation_ratio']:.1f} / {er['rank_bound']} "
        f"({er['fraction_of_capacity'] * 100:.1f}% of capacity used)"
    )
    print(
        "  components for 50/90/99% variance: "
        f"{er['components_50']} / {er['components_90']} / {er['components_99']}"
    )
    print(
        "  pairwise cosine: "
        f"mean {geo['mean']:.3f}  median {geo['median']:.3f}  "
        f"std {geo['std']:.3f}  min {geo['min']:.3f}  max {geo['max']:.3f}"
    )
    print(
        f"    |cos|<0.1: {geo['frac_abs_below_0_1'] * 100:.1f}%   "
        f"cos>0.5: {geo['frac_above_0_5'] * 100:.1f}%   "
        f"cos>0.9: {geo['frac_above_0_9'] * 100:.1f}%"
    )
    if geo["top_collinear"]:
        print("  most collinear pairs:")
        for p in geo["top_collinear"]:
            print(f"    {p['cosine']:.3f}   {p['a']}  ~  {p['b']}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model-path", type=str, required=True,
                   help="Path to the trained diffusion checkpoint (sidecars derived)")
    p.add_argument("--table", type=str, action="append",
                   choices=list(TABLES), default=None,
                   help="Which table(s) to analyse (repeatable; default: all)")
    p.add_argument("--top", type=int, default=10,
                   help="How many most-collinear pairs to list (default 10)")
    p.add_argument("--cos-threshold", type=float, default=0.9,
                   help="Only list collinear pairs above this cosine (default 0.9)")
    p.add_argument("--output-json", type=str, default=None,
                   help="Write the full report as JSON to this path")
    args = p.parse_args()

    model_path = Path(args.model_path)
    checkpoint = torch.load(str(model_path), map_location="cpu")

    # Build labels from the sidecars (best-effort; a missing sidecar just means
    # rows are named by index instead of codepoint/family).
    codepoints = _load_json(Path(str(model_path) + ".codepoints.json"))
    instances = _load_json(Path(str(model_path) + ".instances.json"))
    cp_labels = _codepoint_labels(codepoints)

    tables = args.table or list(TABLES)
    report: dict = {"model": str(model_path), "tables": {}}

    for table in tables:
        weight = _load_table(checkpoint, table)
        n = weight.shape[0]
        if table == "codepoint":
            labels = cp_labels if len(cp_labels) == n else [str(i) for i in range(n)]
        elif table in ("family", "italic_family"):
            labels = _family_labels(instances, n)
        else:  # style (roman/italic)
            labels = ["roman", "italic"][:n]

        er = _effective_rank(weight)
        geo = _cosine_geometry(weight, labels, top=args.top, threshold=args.cos_threshold)
        _print_table(table, er, geo)
        report["tables"][table] = {"effective_rank": er, "cosine": geo}

    # A one-line side-by-side so the codepoint-vs-style comparison is immediate.
    cp = report["tables"].get("codepoint")
    fam = report["tables"].get("family")
    if cp is not None and fam is not None:
        print("\n=== summary ===")
        print(
            "  codepoint capacity used: "
            f"{cp['effective_rank']['fraction_of_capacity'] * 100:.1f}%   "
            f"(PR {cp['effective_rank']['participation_ratio']:.1f} / "
            f"{cp['effective_rank']['rank_bound']})"
        )
        print(
            "  family    capacity used: "
            f"{fam['effective_rank']['fraction_of_capacity'] * 100:.1f}%   "
            f"(PR {fam['effective_rank']['participation_ratio']:.1f} / "
            f"{fam['effective_rank']['rank_bound']})"
        )
        more = "family" if fam["effective_rank"]["fraction_of_capacity"] \
            >= cp["effective_rank"]["fraction_of_capacity"] else "codepoint"
        print(f"  -> the {more} branch is under more representational pressure")

    if args.output_json:
        out = Path(args.output_json)
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
