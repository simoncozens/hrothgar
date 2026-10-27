#!/usr/bin/env python3
"""Probe: where does the fine-style signal (geometric vs humanist) get lost?

Measures the within-vs-between class separation of a fine-style contrast at
three stages of the pipeline, to localise where the signal decays:

* **raw** — the raw (tight-frame) glyph pixels, mean-centered per glyph: the
  signal that actually exists in the data;
* **per-glyph** — the encoder's per-glyph pooled features (before aggregation);
* **summary** — the set-encoder summary (what the model actually ships).

If raw separates but per-glyph/summary do not, the encoder is discarding the
signal.  If per-glyph separates but summary does not, the aggregation is losing
it.  If raw is already ~0, the distinction is genuinely subtle in the pixels.

Usage::

    PYTHONPATH=Lib python scripts/similarity/probe_pooling.py \
        --repo ~/others-repos/fonts --model models/similarity.pth
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F

from hrothgar.googlefonts import GoogleFonts
from hrothgar.render import frame_for_codepoints, render
from hrothgar.similarity.config import FontStyleEmbedderConfig
from hrothgar.similarity.model import FontStyleEmbedder
from hrothgar.utils import pick_device


def _load_model(path: str, device) -> FontStyleEmbedder:
    config = FontStyleEmbedderConfig.from_sidecar(path)
    model = FontStyleEmbedder(config)
    model.load(path, device)
    model.to(device)
    model.eval()
    return model


def _render_set(font, config: FontStyleEmbedderConfig, device) -> torch.Tensor:
    """Render the full input glyph set as ``(1, G, 1, H, W)``, as training does."""
    frame = frame_for_codepoints(font, config.input_codepoints)
    images = torch.stack(
        [
            render(font, codepoint=cp, size=config.glyph_size, frame=frame)[0]
            for cp in config.input_codepoints
        ]
    )
    return images.unsqueeze(1).unsqueeze(0).to(device)  # (1, G, 1, H, W)


def _per_glyph_features(model, images: torch.Tensor) -> torch.Tensor:
    """Per-glyph pooled features ``(G, S, cs, cs)`` for one font's glyph set."""
    g = images.shape[1]
    flat = images.flatten(0, 1)  # (G, 1, H, W)
    with torch.no_grad():
        features = model.encoder(flat)
        proj = model.spatial_proj(features)
        cs = model.config.coarse_grid_size
        proj = F.adaptive_max_pool2d(proj, (cs, cs))
    return proj.reshape(g, model.config.spatial_channels, cs, cs)


def _raw_dist(raw_i: torch.Tensor, raw_j: torch.Tensor) -> float:
    """Aligned per-glyph raw-pixel distance, mean-centered per glyph (no DC)."""
    g = raw_i.shape[0]
    a = raw_i.reshape(g, -1)
    b = raw_j.reshape(g, -1)
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    a = F.normalize(a, p=2, dim=-1, eps=1e-8)
    b = F.normalize(b, p=2, dim=-1, eps=1e-8)
    return float((1.0 - (a * b).sum(-1)).mean())


def _per_glyph_dist(fi: torch.Tensor, fj: torch.Tensor) -> float:
    g = fi.shape[0]
    a = F.normalize(fi.reshape(g, -1), p=2, dim=-1, eps=1e-8)
    b = F.normalize(fj.reshape(g, -1), p=2, dim=-1, eps=1e-8)
    return float((1.0 - (a * b).sum(-1)).mean())


def _matrix(dist_fn, feats: list[torch.Tensor]) -> torch.Tensor:
    n = len(feats)
    d = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            v = dist_fn(feats[i], feats[j])
            d[i, j] = v
            d[j, i] = v
    return d


def _summary_matrix(summaries: list[torch.Tensor]) -> torch.Tensor:
    """Centered, L2-normalized pairwise cosine distance of the summaries."""
    x = torch.stack(summaries)  # (N, D)
    x = x - x.mean(dim=0, keepdim=True)
    x = F.normalize(x, p=2, dim=-1, eps=1e-8)
    return 1.0 - (x @ x.T)


def _separation(dist_matrix: torch.Tensor, labels: list[str], a: str, b: str) -> dict:
    idx_a = torch.tensor([l == a for l in labels]).nonzero().flatten()
    idx_b = torch.tensor([l == b for l in labels]).nonzero().flatten()
    within_a = dist_matrix[idx_a][:, idx_a]
    within_b = dist_matrix[idx_b][:, idx_b]
    within_a = within_a[~torch.eye(len(idx_a), dtype=torch.bool)]
    within_b = within_b[~torch.eye(len(idx_b), dtype=torch.bool)]
    between = dist_matrix[idx_a][:, idx_b]
    within_all = torch.cat([within_a.flatten(), within_b.flatten()])
    return {
        "within": float(within_all.mean()),
        "between": float(between.mean()),
        "gap": float(between.mean() - within_all.mean()),
    }


def _per_glyph_gaps(
    per_font: list[torch.Tensor],
    labels: list[str],
    a: str,
    b: str,
    codepoints: list[int],
    center: bool,
) -> dict[str, float]:
    """Per-glyph fine-style gap ``{char: gap}`` using only that glyph's features.

    ``per_font[i]`` is ``(G, ...)``, aligned by glyph index across fonts.  For
    each glyph the within-vs-between class gap is computed on that glyph alone,
    so we can see which letters carry the geometric-vs-humanist signal.
    """
    n = len(per_font)
    g = per_font[0].shape[0]
    normed: list[torch.Tensor] = []
    for x in per_font:
        x = x.reshape(g, -1).float()
        if center:
            x = x - x.mean(dim=1, keepdim=True)
        normed.append(F.normalize(x, p=2, dim=-1, eps=1e-8))

    gaps: dict[str, float] = {}
    for gi in range(g):
        d = torch.zeros(n, n)
        for i in range(n):
            for j in range(i + 1, n):
                cos = (normed[i][gi] * normed[j][gi]).sum()
                d[i, j] = d[j, i] = 1.0 - cos
        gaps[chr(codepoints[gi])] = _separation(d, labels, a, b)["gap"]
    return gaps


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
    p.add_argument("--limit-per-class", type=int, default=10)
    p.add_argument("--tag-a", default="/Sans/Geometric", help="fine-style class A tag")
    p.add_argument("--tag-b", default="/Sans/Humanist", help="fine-style class B tag")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    device = pick_device() if args.device is None else args.device

    model = _load_model(args.model, device)
    config = model.config
    print(f"model: {args.model}   device: {device}")

    gf = GoogleFonts(args.repo, having=set(config.input_codepoints))
    by_tag: dict[str, list] = {args.tag_a: [], args.tag_b: []}
    values: dict[str, tuple[float, float]] = {}
    for font in gf.fonts:
        tags = font.tags()
        va = tags.get(args.tag_a, 0.0)
        vb = tags.get(args.tag_b, 0.0)
        if va == vb:  # neither tagged, or genuinely ambiguous
            continue
        winner = args.tag_a if va > vb else args.tag_b
        if len(by_tag[winner]) < args.limit_per_class:
            by_tag[winner].append(font)
            values[font.path.name] = (va, vb)

    print(f"{args.tag_a}: {len(by_tag[args.tag_a])} fonts  "
          f"{args.tag_b}: {len(by_tag[args.tag_b])} fonts")

    raws: list[torch.Tensor] = []
    feats: list[torch.Tensor] = []
    summaries: list[torch.Tensor] = []
    labels: list[str] = []
    names: list[str] = []
    for tag, fonts in by_tag.items():
        for font in fonts:
            images = _render_set(font, config, device)
            if bool((images.amin(dim=(-2, -1)) > 0.995).any()):
                continue  # blank glyph
            with torch.no_grad():
                summary = model.encode(images).squeeze(0).cpu()
            raws.append(images[0].cpu())  # (G, 1, H, W)
            feats.append(_per_glyph_features(model, images).cpu())
            summaries.append(summary)
            labels.append(tag)
            names.append(font.path.name)

    if len(feats) < 4 or len(set(labels)) < 2:
        print("Not enough labelled fonts found; bailing.")
        return 1

    print(f"\nscored {len(feats)} fonts (tag centiles "
          f"{args.tag_a.split('/')[-1]} / {args.tag_b.split('/')[-1]}):")
    for n, l in zip(names, labels):
        va, vb = values.get(n, (0.0, 0.0))
        print(f"  [{l.split('/')[-1]:9}] {n:40} {va:5.0f}/{vb:5.0f}")

    sep_raw = _separation(_matrix(_raw_dist, raws), labels, args.tag_a, args.tag_b)
    sep_pg = _separation(_matrix(_per_glyph_dist, feats), labels, args.tag_a, args.tag_b)
    sep_sum = _separation(_summary_matrix(summaries), labels, args.tag_a, args.tag_b)

    for name, sep in (("raw", sep_raw), ("per-glyph", sep_pg), ("summary", sep_sum)):
        print(
            f"\n{name:10} within={sep['within']:.4f}  "
            f"between={sep['between']:.4f}  gap={sep['gap']:+.4f}"
        )

    raw_gaps = _per_glyph_gaps(
        raws, labels, args.tag_a, args.tag_b, config.input_codepoints, center=True
    )
    feat_gaps = _per_glyph_gaps(
        feats, labels, args.tag_a, args.tag_b, config.input_codepoints, center=False
    )
    for title, gaps in (("raw pixels", raw_gaps), ("encoder features", feat_gaps)):
        ordered = sorted(gaps.items(), key=lambda kv: -kv[1])
        print(f"\nper-glyph gaps ({title}) — top 15 / bottom 5:")
        for char, gap in ordered[:15]:
            print(f"    {char!r:8} {gap:+.4f}")
        print("    ...")
        for char, gap in ordered[-5:]:
            print(f"    {char!r:8} {gap:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
