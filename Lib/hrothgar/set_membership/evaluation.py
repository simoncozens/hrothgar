"""Validation panel and membership breakdowns.

Builds a fixed, seeded panel of test instances (multi-instance families first)
and scores membership three ways:

* ``overall``       — query glyph vs every instance prototype;
* ``same_family``   — query glyph vs its family's own sibling instances (weight);
* ``same_category`` — query glyph vs one prototype per same-category family.

The last two are the anti-collapse checks: can the model tell Sans-A from
Sans-B (family-specific detail), and can it tell Bold from Regular?
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from hrothgar.render import geometry_tensor, render
from hrothgar.set_membership.config import SetMembershipConfig


@dataclass
class MembershipEvalPanel:
    reference_images: torch.Tensor  # (I, R, 1, H, W)
    reference_geometry: torch.Tensor  # (I, R, 5)
    query_images: torch.Tensor  # (I, Q, 1, H, W)
    query_geometry: torch.Tensor  # (I, Q, 5)
    families: list[str]  # (I,)
    categories: list[str]  # (I,)


def build_membership_panel(
    instances,
    config: SetMembershipConfig,
    *,
    max_instances: int = 200,
) -> MembershipEvalPanel:
    """Select a panel deterministically, multi-instance families first.

    Multi-instance families come first so the same-family (weight) breakdown has
    enough sibling pairs to score.
    """
    by_family: dict[str, list] = defaultdict(list)
    for instance in instances:
        by_family[instance.family].append(instance)

    selected: list = []
    for family in sorted(by_family, key=lambda f: (-len(by_family[f]), f)):
        for instance in sorted(by_family[family], key=lambda x: str(x.font.path)):
            selected.append(instance)
            if len(selected) >= max_instances:
                break
        if len(selected) >= max_instances:
            break

    ref_cps = config.input_codepoints[: config.reference_glyphs]
    query_cps = config.input_codepoints[
        config.reference_glyphs : config.reference_glyphs + config.query_glyphs
    ]

    ref_images, ref_geometry, query_images, query_geometry = [], [], [], []
    families, categories = [], []
    for instance in selected:
        ri, rg, qi, qg = [], [], [], []
        for cp in ref_cps:
            image, geom = render(
                instance.font,
                codepoint=cp,
                size=config.glyph_size,
                axis_position=instance.axis_position,
            )
            ri.append(image.unsqueeze(0))
            rg.append(geometry_tensor(geom))
        for cp in query_cps:
            image, geom = render(
                instance.font,
                codepoint=cp,
                size=config.glyph_size,
                axis_position=instance.axis_position,
            )
            qi.append(image.unsqueeze(0))
            qg.append(geometry_tensor(geom))
        ref_images.append(torch.stack(ri))
        ref_geometry.append(torch.stack(rg))
        query_images.append(torch.stack(qi))
        query_geometry.append(torch.stack(qg))
        families.append(instance.family)
        categories.append(instance.category)

    return MembershipEvalPanel(
        reference_images=torch.stack(ref_images),
        reference_geometry=torch.stack(ref_geometry),
        query_images=torch.stack(query_images),
        query_geometry=torch.stack(query_geometry),
        families=families,
        categories=categories,
    )


def embed_glyphs(
    model, images: torch.Tensor, device, chunk: int = 32
) -> torch.Tensor:
    out = []
    for start in range(0, images.shape[0], chunk):
        with torch.no_grad():
            z = model(images[start : start + chunk].to(device))
        out.append(z.cpu())
    return torch.cat(out)


def _rank_metrics(
    query: torch.Tensor, candidates: torch.Tensor, own: torch.Tensor
) -> dict:
    """Rank each query row against ``candidates``; ``own`` is the correct column."""
    n = query.shape[0]
    if n == 0 or candidates.shape[0] < 2:
        return {"r1": 0.0, "mrr": 0.0, "auc": 0.0, "n": int(n)}

    sim = F.normalize(query, dim=-1) @ F.normalize(candidates, dim=-1).T  # (Nq, Nc)
    rows = torch.arange(n)
    correct = sim[rows, own]
    rank = (sim > correct.unsqueeze(1)).sum(dim=1) + 1

    others = sim.clone()
    others[rows, own] = float("-inf")
    auc = (correct.unsqueeze(1) > others).float().mean(dim=1)
    return {
        "r1": float((rank <= 1).float().mean()),
        "mrr": float((1.0 / rank.float()).mean()),
        "auc": float(auc.mean()),
        "n": int(n),
    }


def _aggregate(metrics: list[dict]) -> dict:
    if not metrics:
        return {"r1": 0.0, "mrr": 0.0, "auc": 0.0, "n": 0}
    total = sum(m["n"] for m in metrics) or 1
    return {
        "r1": sum(m["r1"] * m["n"] for m in metrics) / total,
        "mrr": sum(m["mrr"] * m["n"] for m in metrics) / total,
        "auc": sum(m["auc"] * m["n"] for m in metrics) / total,
        "n": int(total),
    }


def _same_family_metrics(queries, prototypes, family_to_idx) -> dict:
    metrics = []
    for idxs in family_to_idx.values():
        if len(idxs) < 2:
            continue
        candidates = prototypes[idxs]  # (F, D)
        col = {instance_idx: j for j, instance_idx in enumerate(idxs)}
        for instance_idx in idxs:
            own = torch.full((queries.shape[1],), col[instance_idx], dtype=torch.long)
            metrics.append(_rank_metrics(queries[instance_idx], candidates, own))
    return _aggregate(metrics)


def _same_category_metrics(queries, prototypes, families, categories) -> dict:
    family_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, family in enumerate(families):
        family_to_idx[family].append(i)
    family_proto = {
        family: F.normalize(prototypes[idxs].mean(0), dim=-1)
        for family, idxs in family_to_idx.items()
    }
    category_to_families: dict[str, set[str]] = defaultdict(set)
    for i, family in enumerate(families):
        category_to_families[categories[i]].add(family)

    metrics = []
    for family_names in category_to_families.values():
        family_names = sorted(family_names)
        if len(family_names) < 2:
            continue
        candidates = torch.stack([family_proto[f] for f in family_names])  # (F, D)
        col = {family: j for j, family in enumerate(family_names)}
        for i, family in enumerate(families):
            if family not in col:
                continue
            own = torch.full((queries.shape[1],), col[family], dtype=torch.long)
            metrics.append(_rank_metrics(queries[i], candidates, own))
    return _aggregate(metrics)


def membership_breakdown(model, panel: MembershipEvalPanel, device) -> dict:
    """Score overall, same-family, and same-category membership."""
    reference = embed_glyphs(
        model,
        panel.reference_images.flatten(0, 1),
        device,
    )
    r = panel.reference_images.shape[1]
    i = panel.reference_images.shape[0]
    prototypes = F.normalize(reference.view(i, r, -1).mean(dim=1), dim=-1)  # (I, D)

    query = embed_glyphs(
        model,
        panel.query_images.flatten(0, 1),
        device,
    )
    q = panel.query_images.shape[1]
    queries = query.view(i, q, -1)  # (I, Q, D)

    flat_queries = queries.reshape(-1, queries.shape[-1])  # (I*Q, D)
    own = torch.arange(i).unsqueeze(1).expand(i, q).reshape(-1)

    family_to_idx: dict[str, list[int]] = defaultdict(list)
    for idx, family in enumerate(panel.families):
        family_to_idx[family].append(idx)

    return {
        "overall": _rank_metrics(flat_queries, prototypes, own),
        "same_family": _same_family_metrics(queries, prototypes, family_to_idx),
        "same_category": _same_category_metrics(
            queries, prototypes, panel.families, panel.categories
        ),
    }


def main() -> int:
    """Evaluate a set-membership checkpoint against the held-out test instances."""
    import torch

    from hrothgar.set_membership.dataset import SetMembershipDatasetMaker
    from hrothgar.set_membership.model import SetMembershipModel
    from hrothgar.utils import pick_device

    parser = argparse.ArgumentParser(
        description="Evaluate an instance-membership checkpoint."
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts"),
    )
    parser.add_argument("--model", required=True, help="path to the .pth checkpoint")
    parser.add_argument(
        "--max-instances",
        type=int,
        default=200,
        help="eval panel size (0 = all test instances)",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=None, help="write metrics as JSON")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    config = SetMembershipConfig.from_sidecar(args.model)
    model = SetMembershipModel(config)
    model.load(args.model, device)
    model.to(device)
    model.eval()

    maker = SetMembershipDatasetMaker(args.repo, batch_size=64, config=config)
    max_instances = args.max_instances or len(maker.test_instances)
    panel = build_membership_panel(
        maker.test_instances, config, max_instances=max_instances
    )
    print(
        f"model: {args.model}  panel: {len(panel.families)} instances  device: {device}"
    )

    breakdown = membership_breakdown(model, panel, device)
    for scope in ("overall", "same_family", "same_category"):
        m = breakdown[scope]
        print(
            f"  {scope:12} r1={m['r1']:.3f}  mrr={m['mrr']:.3f}  "
            f"auc={m['auc']:.3f}  n={m['n']}"
        )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(breakdown, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
