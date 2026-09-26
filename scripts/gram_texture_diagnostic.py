#!/usr/bin/env python3
"""Gram-matrix texture diagnostic: does destroying spatial order help?

For each strong family we compute a "texture" embedding from the frozen
``GlyphEncoder`` features via a per-glyph Gram matrix (channel correlations,
spatial order destroyed), for BOTH the non-Latin glyphs and the Latin
companion.  We then measure cosine similarity between the non-Latin texture
and the Latin texture, to compare against the summary/projection baseline
(whose median was ~0.016).

The Gram matrix is mean-pooled over spatial locations and averaged across
glyphs, so it is density-normalized by construction; we also L2-normalize the
flattened upper triangle for cosine comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fontTools.unicodedata import script as codepoint_script

from hrothgar.googlefonts import StandaloneFont
from hrothgar.style_embedding.config import FontStyleEmbedderConfig
from hrothgar.style_embedding.model import FontStyleEmbedder
from hrothgar.style_embedding.render_utils import render_input_set

STRONG_ORIGINS = {"Custom", "Designed to match Latin"}
TARGET_SCRIPTS = {
    "Deva": "Devanagari",
    "Arab": "Arabic",
    "Thai": "Thai",
    "Taml": "Tamil",
    "Guru": "Gurmukhi",
    "Telu": "Telugu",
}
_NON_SCRIPT = {"Latn", "Zyyy", "Zinh", "Zzzz", "Zsym", "Zpun"}
MIN_GLYPHS = 8


def build_filename_index(repo: Path) -> dict[str, str]:
    base = repo / "ofl" if (repo / "ofl").is_dir() else repo
    return {p.name: str(p) for p in base.glob("*/*.ttf")}


def detect_script(font: StandaloneFont) -> str | None:
    counts: Counter[str] = Counter()
    for cp in font.codepoints:
        counts[codepoint_script(chr(cp))] += 1
    for code in _NON_SCRIPT:
        counts.pop(code, None)
    return counts.most_common(1)[0][0] if counts else None


def nonlatin_letters(font: StandaloneFont, code: str) -> list[int]:
    return [
        cp
        for cp in sorted(font.codepoints)
        if codepoint_script(chr(cp)) == code
        and unicodedata.category(chr(cp)).startswith("L")
    ]


def gram_texture(
    model: FontStyleEmbedder,
    font: StandaloneFont,
    codepoints: list[int],
    glyph_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    imgs = render_input_set(font, codepoints, glyph_size)  # (G, 1, H, W)
    imgs = imgs[~(imgs.amin(dim=(-2, -1)) > 0.995).squeeze(1)]
    if imgs.shape[0] < MIN_GLYPHS:
        return None
    with torch.no_grad():
        feat = model.encoder(imgs.to(device))  # (G, F, h, w)
        g, fd, h, w = feat.shape
        feat = feat.reshape(g, fd, h * w)
        gram = torch.bmm(feat, feat.transpose(1, 2)) / (h * w)  # (G, F, F)
        gram = gram.mean(dim=0)  # (F, F) mean over glyphs
        idx = torch.triu_indices(fd, fd)
        tri = gram[idx[0], idx[1]]
        return F.normalize(tri, p=2, dim=0).cpu()


def pick_canonical(fonts: list[str]) -> str:
    fonts = sorted(fonts)
    for f in fonts:
        if "regular" in f.lower():
            return f
    return fonts[0]


def stats(xs: list[float]) -> str:
    a = np.asarray(xs)
    return f"mean={a.mean():.3f}  median={np.median(a):.3f}  n={len(a)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts"),
    )
    parser.add_argument("--csv", default="non-latins.resolved.csv")
    parser.add_argument("--manifest", default="latin_companion_embeddings.json")
    parser.add_argument("--model", default="models/style_embedding_finetune.pth")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    repo = Path(args.repo)

    cfg = FontStyleEmbedderConfig.from_sidecar(args.model)
    model = FontStyleEmbedder(cfg)
    model.load(args.model, device)
    model.to(device)
    model.eval()

    filename_to_path = build_filename_index(repo)
    manifest = json.load(open(args.manifest, encoding="utf-8"))

    rows = list(
        csv.DictReader(
            open(args.csv, newline="", encoding="utf-8"), skipinitialspace=True
        )
    )
    by_family: dict[str, dict] = defaultdict(
        lambda: {"fonts": [], "origin": None, "latin_family": ""}
    )
    for r in rows:
        origin = (r.get("Latin origin") or "").strip()
        if origin not in STRONG_ORIGINS:
            continue
        fam = (r.get("Family") or "").strip()
        by_family[fam]["fonts"].append((r.get("Font") or "").strip())
        by_family[fam]["origin"] = origin
        by_family[fam]["latin_family"] = (r.get("Latin family") or "").strip()

    recs = []
    for fam, info in by_family.items():
        canonical = pick_canonical(info["fonts"])
        path = filename_to_path.get(canonical)
        companion = (manifest.get(canonical) or {}).get("companion")
        if path is None or companion is None:
            continue
        font = StandaloneFont(path)
        code = detect_script(font)
        if code not in TARGET_SCRIPTS:
            continue
        letters = nonlatin_letters(font, code)
        naive = gram_texture(model, font, letters, cfg.glyph_size, device)
        latin = gram_texture(
            model,
            StandaloneFont(companion),
            cfg.input_codepoints,
            cfg.glyph_size,
            device,
        )
        if naive is None or latin is None:
            continue
        recs.append(
            {
                "origin": info["origin"],
                "script": TARGET_SCRIPTS[code],
                "companion": companion,
                "naive": naive,
                "latin": latin,
            }
        )

    if args.limit:
        recs = recs[: args.limit]

    # Mean-center the Gram embeddings: the flattened Gram is dominated by a
    # common diagonal-energy component, so raw cosines are ~0.99 everywhere.
    # Center against the distinct Latin companions and renormalize.
    seen_companions: set[str] = set()
    latin_pool: list[torch.Tensor] = []
    for r in recs:
        if r["companion"] not in seen_companions:
            seen_companions.add(r["companion"])
            latin_pool.append(r["latin"])
    mu = torch.stack(latin_pool).mean(dim=0)
    for r in recs:
        r["naive"] = F.normalize(r["naive"] - mu, p=2, dim=0)
        r["latin"] = F.normalize(r["latin"] - mu, p=2, dim=0)

    custom = [r for r in recs if r["origin"] == "Custom"]
    dm = [r for r in recs if r["origin"] == "Designed to match Latin"]

    def cos(a, b):
        return float((a * b).sum())

    print("=== Gram-texture domain-gap (cosine) ===")
    print(
        f"  Custom (non-Latin vs own Latin):        {stats([cos(r['naive'], r['latin']) for r in custom])}"
    )
    print(
        f"  Designed-to-match (non-Latin vs Latin): {stats([cos(r['naive'], r['latin']) for r in dm])}"
    )

    # Unrelated Latin-Latin baseline: distinct companions.
    seen, latins = {}, []
    for r in recs:
        if r["companion"] not in seen:
            seen[r["companion"]] = len(latins)
            latins.append(r["latin"])
    if len(latins) > 1:
        t = torch.stack(latins)
        sims = t @ t.T
        k = sims.shape[0]
        off = float(sims[~torch.eye(k, dtype=torch.bool)].mean())
        print(f"  Unrelated Latin-Latin (Gram texture):    mean={off:.3f}  (n={k})")

    # Per-script Custom breakdown.
    print("\n  Per-script Custom cos(non-Latin texture, own Latin texture):")
    per = defaultdict(list)
    for r in custom:
        per[r["script"]].append(cos(r["naive"], r["latin"]))
    for s, xs in sorted(per.items()):
        a = np.asarray(xs)
        print(
            f"    {s:12} n={len(a):2d}  mean={a.mean():.3f}  median={np.median(a):.3f}"
        )

    # ---- Retrieval (zero-shot, mean-centered Gram space) ----
    search_names: list[str] = []
    search_embs: list[torch.Tensor] = []
    idx: dict[str, int] = {}
    for r in recs:
        c = r["companion"]
        if c not in idx:
            idx[c] = len(search_names)
            search_names.append(c)
            search_embs.append(r["latin"])
    search = torch.stack(search_embs)  # (K, D)
    ks = (1, 5, 10)

    for label, origin in [
        ("Custom (sanity)", "Custom"),
        ("Designed to match Latin", "Designed to match Latin"),
    ]:
        test = [r for r in recs if r["origin"] == origin]
        ranks = []
        for r in test:
            sims = search @ r["naive"]
            ci = idx[r["companion"]]
            ranks.append(
                int((sims.argsort(descending=True) == ci).nonzero()[0].item()) + 1
            )
        print(f"\n=== Retrieval: {label} (zero-shot Gram, K={len(search_names)}) ===")
        print(f"  MRR: {np.mean([1 / r for r in ranks]):.3f}")
        print(
            f"  Recall: { {k: round(sum(1 for r in ranks if r <= k) / len(ranks), 3) for k in ks} }"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
