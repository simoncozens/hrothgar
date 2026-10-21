#!/usr/bin/env python3
"""Evaluation suite for the font similarity embedder (Job 1).

Answers the question the training `Validation/retrieval_*` metric does *not*:
given a font held out of training, are its nearest neighbours actually the
fonts a designer would call similar?  The training metric only checks that two
masked views of the *same* font land near each other, which is trivially
satisfiable and never touches retrieval.

The suite is defined once in ``hrothgar.similarity.evaluation`` and shared
with the training loop, which logs the same numbers under
``Validation/Similarity/*``.  Reported here:

* **A. Family-sibling retrieval** — for each held-out font that has a sibling
  (same family, different file), is the sibling the nearest neighbour?
* **B. Category agreement** — do a font's neighbours share its category?
* **C. Library transfer** — for an unseen font, do its nearest neighbours in
  the *training* library share its category?
* **D. Embedding geometry** — pairwise-cosine spread and effective dimension.
* **E. Variable-font invariance** — a font rendered at two axis positions
  should stay nearest to its own family (invariance) without collapsing.
* **F. Glyph-sampling stability** — do two disjoint glyph subsets embed alike?
* **G. Examples** — a qualitative top-5 dump for eyeballing.

Usage::

    PYTHONPATH=Lib python scripts/similarity_eval.py \
        --repo ~/others-repos/fonts --model models/similarity.pth \
        --space summary --out similarity_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from hrothgar.dataset import DatasetMaker
from hrothgar.googlefonts import GoogleFonts
from hrothgar.similarity.config import FontStyleEmbedderConfig
from hrothgar.similarity.evaluation import (
    build_eval_panel,
    embed_panel,
    run_suite,
)
from hrothgar.similarity.metrics import (
    cross_set_purity,
    nearest_neighbours,
    sibling_retrieval,
)
from hrothgar.similarity.model import FontStyleEmbedder
from hrothgar.utils import pick_device


def _load_model(path: str, device) -> FontStyleEmbedder:
    config = FontStyleEmbedderConfig.from_sidecar(path)
    model = FontStyleEmbedder(config)
    model.load(path, device)
    model.to(device)
    model.eval()
    return model


def _retrieval_by_category(embeddings, families, categories, ks=(1, 3, 5)):
    out = {}
    for category in sorted(set(categories)):
        rows = [i for i, c in enumerate(categories) if c == category]
        score = sibling_retrieval(embeddings[rows], [families[i] for i in rows], ks=ks)
        if score.n_queries:
            out[category] = score
    return out


def _fmt(recall: dict, mrr: float, n: int) -> str:
    return (
        f"R@1={recall[1]:.3f} R@3={recall[3]:.3f} R@5={recall[5]:.3f} "
        f"MRR={mrr:.3f} (n={n})"
    )


def _family_section(panel, spaces) -> dict:
    by_category = _retrieval_by_category(
        spaces["summary"], panel.families, panel.categories
    )
    section = {"by_category": {c: s.as_dict() for c, s in by_category.items()}}
    for name, vectors in spaces.items():
        score = sibling_retrieval(vectors, panel.families)
        section[name] = score.as_dict()
        print(f"  {name:11} {_fmt(score.recall, score.mrr, score.n_queries)}")
    for category, score in by_category.items():
        print(f"    {category:12} {_fmt(score.recall, score.mrr, score.n_queries)}")
    return section


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--repo",
        default=os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts"),
    )
    p.add_argument("--model", default="models/similarity.pth")
    p.add_argument("--device", default=None)
    p.add_argument("--space", choices=["summary", "projection"], default="summary")
    p.add_argument("--limit", type=int, default=None, help="cap fonts scanned (canary)")
    p.add_argument("--split-seed", type=int, default=1234)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--examples", type=int, default=6)
    p.add_argument("--stability-fonts", type=int, default=96)
    p.add_argument(
        "--membership-codepoints",
        type=int,
        default=8,
        help="glyph slots per font for the set-membership eval (0 = skip)",
    )
    p.add_argument(
        "--transfer-index",
        type=int,
        default=800,
        help="train fonts for suite C (0 = skip)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="write all metrics as JSON here")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    device = pick_device() if args.device is None else args.device
    rng = random.Random(args.seed)

    model = _load_model(args.model, device)
    config = model.config
    print(f"model: {args.model}   space: {args.space}   device: {device}")

    gf = GoogleFonts(
        args.repo, having=set(config.input_codepoints), max_fonts=args.limit
    )
    train_fonts, test_fonts = DatasetMaker._split_fonts_by_family(
        list(gf.fonts), split_seed=args.split_seed
    )
    print(
        f"fonts scanned: {len(gf.fonts)}  split: {len(train_fonts)} train / "
        f"{len(test_fonts)} test (family-disjoint, seed {args.split_seed})"
    )

    panel = build_eval_panel(
        test_fonts,
        config,
        seed=args.seed,
        stability_fonts=args.stability_fonts,
        with_descriptors=True,
    )
    print(
        f"eval panel: {len(panel.fonts)} fonts (skipped {len(test_fonts) - len(panel.fonts)} blank)"
    )

    spaces = {
        name: embed_panel(model, panel, device, space=name)
        for name in ("summary", "projection")
    }
    emb = spaces[args.space]

    report = run_suite(
        model,
        panel,
        device,
        space=args.space,
        top_k=args.top_k,
        embeddings=emb,
        membership_codepoints=args.membership_codepoints,
        seed=args.seed,
    )

    print("\n[A] Family-sibling retrieval (held-out index)")
    report["family_sibling"] = _family_section(panel, spaces)

    print(f"\n[B] Category agreement of top-{args.top_k} neighbours (held-out index)")
    cat = report["category_agreement"]
    print(f"  overall {cat['overall']:.3f}")
    for label, value in cat["per_label"].items():
        print(f"    {label:12} {value:.3f}")

    if args.transfer_index > 0 and train_fonts:
        train_panel = build_eval_panel(
            train_fonts, config, max_fonts=args.transfer_index, seed=args.seed
        )
        train_emb = embed_panel(model, train_panel, device, space=args.space)
        report["library_transfer"] = cross_set_purity(
            emb, panel.categories, train_emb, train_panel.categories, k=args.top_k
        )
        print(f"\n[C] Library transfer (index = {len(train_panel.fonts)} train fonts)")
        print(f"  category agreement {report['library_transfer']['overall']:.3f}")

    print("\n[D] Embedding geometry")
    geom = report["geometry"]
    print(
        f"  cosine raw {geom['mean_cosine_raw']:.3f}±{geom['std_cosine_raw']:.3f}  "
        f"centered {geom['mean_cosine_centered']:.3f}±{geom['std_cosine_centered']:.3f}"
    )
    print(f"  participation ratio {geom['participation_ratio']:.1f} / {geom['dim']}")

    if "attribution" in report:
        att = report["attribution"]
        print(
            "\n[D2] Signal attribution (embedding similarity vs reference descriptors)"
        )
        for name in ("proportion", "shape"):
            if name in att:
                print(
                    f"  {name:11} pearson {att[name]['pearson']:+.3f}  "
                    f"spearman {att[name]['spearman']:+.3f}"
                )
        joint = att.get("joint", {})
        betas = ", ".join(
            f"{k[len('beta_'):]}={v:+.2f}"
            for k, v in joint.items()
            if k.startswith("beta_")
        )
        print(f"  joint R² {joint.get('r2', 0.0):.3f}  ({betas})")

    if "axis_invariance" in report:
        print("\n[E] Variable-font invariance (perturbed render vs held-out index)")
        for tag, axis in report["axis_invariance"].items():
            family, anchor = axis["family"], axis["anchor"]
            print(
                f"  {tag:5} family {_fmt(family['recall'], family['mrr'], family['n_pairs'])}  "
                f"anchor R@1={anchor['recall'][1]:.3f}  cos(pair)={anchor['cos_same_pair']:.3f} "
                f"cos(unrel)={anchor['cos_unrelated']:.3f}"
            )

    if "sampling_stability" in report:
        stab = report["sampling_stability"]
        print(f"\n[F] Glyph-sampling stability (n={stab['n']})")
        print(
            f"  mean cos {stab['mean']:.3f}  min {stab['min']:.3f}  p10 {stab['p10']:.3f}"
        )

    if "membership" in report:
        memb = report["membership"]
        print(
            f"\n[H] Set membership — single glyph ranks its own font first? (n={memb['n']})"
        )
        print(
            f"  R@1 {memb['recall'][1]:.3f}  R@3 {memb['recall'][3]:.3f}  "
            f"R@5 {memb['recall'][5]:.3f}  MRR {memb['mrr']:.3f}  AUC {memb['auc']:.3f}"
        )

    print(f"\n[G] Examples (top-{args.top_k}, {args.space})")
    queries = rng.sample(range(len(panel.fonts)), min(args.examples, len(panel.fonts)))
    report["examples"] = []
    for qi, row in zip(
        queries, nearest_neighbours(emb, panel.families, queries, k=args.top_k)
    ):
        print(f"  {panel.fonts[qi].path.name}")
        for _idx, family, sim in row:
            mark = "=" if family == panel.families[qi] else " "
            print(f"    {mark} {sim:+.3f}  {family}")
        report["examples"].append(
            {
                "query": panel.families[qi],
                "neighbours": [{"family": f, "cosine": s} for _, f, s in row],
            }
        )

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
