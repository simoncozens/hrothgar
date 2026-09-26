#!/usr/bin/env python3
"""Tier-1 baseline: map non-Latin glyphs into the Latin style space.

Training signal: ``Custom`` families — a non-Latin font whose companion Latin
lives in the *same* file.  We render the font's non-Latin base letters, encode
them with the frozen ``FontStyleEmbedder`` (a "naive" 256-d embedding), and
regress that against the cached Latin embedding of the same file.

Evaluation: ``Designed to match Latin`` families — a non-Latin font explicitly
designed to match a *separate* Latin family.  For each, we predict the Latin
embedding (with and without the ridge map) and retrieve the nearest companion
from the set of distinct Latin companion families, reporting Recall@k and MRR.

Only the six scripts of interest are used (Devanagari, Arabic, Thai, Tamil,
Gurmukhi, Telugu).  Script detection is done per-codepoint via
``fontTools.unicodedata.script``, and the non-Latin glyph set is the font's own
base letters (category ``L``) in that script — so no external glyphset data is
needed.
"""

from __future__ import annotations

import argparse
import csv
import os
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fontTools.unicodedata import script as codepoint_script
from sklearn.linear_model import Ridge

from hrothgar.googlefonts import StandaloneFont
from hrothgar.style_embedding.config import FontStyleEmbedderConfig
from hrothgar.style_embedding.model import FontStyleEmbedder
from hrothgar.style_embedding.render_utils import render_input_set

STRONG_ORIGINS = {"Custom", "Designed to match Latin"}

# ISO 15924 script codes -> human name, for the six scripts we care about.
TARGET_SCRIPTS = {
    "Deva": "Devanagari",
    "Arab": "Arabic",
    "Thai": "Thai",
    "Taml": "Tamil",
    "Guru": "Gurmukhi",
    "Telu": "Telugu",
}

# Script codes that are not writing systems (shared/inherited/etc.).
_NON_SCRIPT = {"Latn", "Zyyy", "Zinh", "Zzzz", "Zsym", "Zpun"}

MIN_GLYPHS = 8


def build_filename_index(repo: Path) -> dict[str, str]:
    """Map a ``.ttf`` basename to its full path (for rendering anchors)."""
    base = repo / "ofl" if (repo / "ofl").is_dir() else repo
    return {p.name: str(p) for p in base.glob("*/*.ttf")}


def detect_script(font: StandaloneFont) -> str | None:
    """Return the dominant non-Latin script code (e.g. ``"Deva"``), or None."""
    counts: Counter[str] = Counter()
    for cp in font.codepoints:
        counts[codepoint_script(chr(cp))] += 1
    for code in _NON_SCRIPT:
        counts.pop(code, None)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def nonlatin_letters(font: StandaloneFont, script_code: str) -> list[int]:
    out = []
    for cp in sorted(font.codepoints):
        if codepoint_script(chr(cp)) == script_code and unicodedata.category(
            chr(cp)
        ).startswith("L"):
            out.append(cp)
    return out


def naive_embed(
    model: FontStyleEmbedder,
    font: StandaloneFont,
    script_code: str,
    glyph_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    letters = nonlatin_letters(font, script_code)
    if len(letters) < MIN_GLYPHS:
        return None

    imgs = render_input_set(font, letters, glyph_size)  # (G, 1, H, W)
    blank = imgs.amin(dim=(-2, -1)) > 0.995  # (G, 1)
    imgs = imgs[~blank.squeeze(1)]  # drop blank (empty-outline) glyphs
    if imgs.shape[0] < MIN_GLYPHS:
        return None

    with torch.no_grad():
        return model.encode(imgs.unsqueeze(0).to(device)).squeeze(0).cpu()


def pick_canonical(fonts: list[str]) -> str:
    fonts = sorted(fonts)
    for f in fonts:
        if "regular" in f.lower():
            return f
    return fonts[0]


def rank_of_correct(sims_row: np.ndarray, correct_idx: int) -> int:
    order = np.argsort(-sims_row)
    return int(np.where(order == correct_idx)[0][0]) + 1


def recall_at_k(ranks: list[int], ks: tuple[int, ...]) -> dict[int, float]:
    return {k: sum(1 for r in ranks if r <= k) / len(ranks) for k in ks}


def mrr(ranks: list[int]) -> float:
    return float(np.mean([1.0 / r for r in ranks]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts"),
    )
    parser.add_argument("--csv", default="non-latins.resolved.csv")
    parser.add_argument("--embeddings", default="latin_companion_embeddings.pt")
    parser.add_argument("--model", default="models/style_embedding_finetune.pth")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alpha", type=float, default=1.0, help="ridge alpha")
    parser.add_argument(
        "--space",
        choices=["summary", "projection"],
        default="projection",
        help="embedding space to regress/retrieve in (default: projection)",
    )
    parser.add_argument(
        "--center",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="mean-center (subtract population mean) before cosine similarity",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    repo = Path(args.repo)

    cfg = FontStyleEmbedderConfig.from_sidecar(args.model)
    model = FontStyleEmbedder(cfg)
    model.load(args.model, device)
    model.to(device)
    model.eval()

    cache = torch.load(args.embeddings, map_location="cpu", weights_only=True)
    cache = {k: {kk: vv.detach() for kk, vv in v.items()} for k, v in cache.items()}
    filename_to_path = build_filename_index(repo)

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, skipinitialspace=True))

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

    families = []
    for fam, info in by_family.items():
        canonical = pick_canonical(info["fonts"])
        path = filename_to_path.get(canonical)
        target = cache.get(canonical)
        if path is None or target is None:
            continue
        companion = fam if info["origin"] == "Custom" else info["latin_family"]
        families.append(
            {
                "family": fam,
                "origin": info["origin"],
                "companion": companion,
                "font": canonical,
                "path": path,
                "target": target["embedding"],
                "target_proj": target["projection"],
            }
        )

    print("Computing naive non-Latin embeddings …")
    n_excluded_script = 0
    n_dropped = 0
    for rec in families:
        font = StandaloneFont(rec["path"])
        code = detect_script(font)
        if code not in TARGET_SCRIPTS:
            rec["script"] = None
            rec["naive"] = None
            n_excluded_script += 1
            continue
        rec["script"] = TARGET_SCRIPTS[code]
        rec["naive"] = naive_embed(model, font, code, cfg.glyph_size, device)
        if rec["naive"] is None:
            n_dropped += 1
            continue
        with torch.no_grad():
            p = model.projection(rec["naive"].unsqueeze(0).to(device)).squeeze(0)
            rec["naive_proj"] = torch.nn.functional.normalize(p, p=2, dim=-1).cpu()

    if args.space == "projection":
        for rec in families:
            rec["target"] = rec["target_proj"]
            if rec["naive"] is not None:
                rec["naive"] = rec["naive_proj"]

    if args.center:
        mu = torch.stack([rec["target"] for rec in families]).mean(dim=0)
        for rec in families:
            rec["target"] = F.normalize(rec["target"] - mu, p=2, dim=-1)
            if rec["naive"] is not None:
                rec["naive"] = F.normalize(rec["naive"] - mu, p=2, dim=-1)

    train = [r for r in families if r["origin"] == "Custom" and r["naive"] is not None]
    test = [
        r
        for r in families
        if r["origin"] == "Designed to match Latin" and r["naive"] is not None
    ]
    if args.limit:
        train = train[: args.limit]
        test = test[: args.limit]

    print(
        f"families: {len(families)} total, {len(train)} train (Custom), "
        f"{len(test)} test (Designed to match Latin); "
        f"{n_excluded_script} excluded (non-target script), {n_dropped} dropped "
        f"(no target / too few glyphs)"
    )

    search_names: list[str] = []
    search_embs: list[torch.Tensor] = []
    seen: dict[str, int] = {}
    for rec in families:
        if rec["target"] is None:
            continue
        c = rec["companion"]
        if c and c not in seen:
            seen[c] = len(search_names)
            search_names.append(c)
            search_embs.append(rec["target"])
    search = torch.stack(search_embs).numpy().astype(np.float32)
    search = search / np.linalg.norm(search, axis=1, keepdims=True)
    print(f"search set: {len(search_names)} distinct companion families")

    # ---- Domain-gap diagnostic (cosine similarity, higher = closer) ----
    def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a * b).sum() / (a.norm() * b.norm()).clamp(min=1e-8))

    custom_cos: list[float] = []
    dm_cos: list[float] = []
    for rec in families:
        if rec["naive"] is None:
            continue
        c = _cos(rec["naive"], rec["target"])
        (custom_cos if rec["origin"] == "Custom" else dm_cos).append(c)

    search_t = torch.stack(search_embs)  # (K, 256)
    search_t = search_t / search_t.norm(dim=1, keepdim=True).clamp(min=1e-8)
    sims_ll = search_t @ search_t.T
    k = sims_ll.shape[0]
    unrelated_ll = float(sims_ll[~torch.eye(k, dtype=torch.bool)].mean())

    cross_unrelated: list[float] = []
    for rec in families:
        if rec["naive"] is None:
            continue
        ci = seen.get(rec["companion"], -1)
        naive_n = rec["naive"] / rec["naive"].norm().clamp(min=1e-8)
        sims = search_t @ naive_n  # (K,)
        if ci >= 0:
            sims = torch.cat([sims[:ci], sims[ci + 1 :]])
        cross_unrelated.append(float(sims.mean()))

    def _stats(xs: list[float]) -> str:
        a = np.asarray(xs)
        return f"mean={a.mean():.3f}  median={np.median(a):.3f}  n={len(a)}"

    print("\n=== Domain-gap diagnostic (cosine similarity) ===")
    print(f"  Custom (non-Latin vs its own Latin):        {_stats(custom_cos)}")
    print(f"  Designed-to-match (non-Latin vs target Latin): {_stats(dm_cos)}")
    print(f"  Unrelated Latin-Latin (different families):  mean={unrelated_ll:.3f}")
    print(
        f"  Unrelated cross-script (non-Latin vs other Latin): {_stats(cross_unrelated)}"
    )

    test_correct_idx = [seen.get(rec["companion"], -1) for rec in test]

    def retrieve(embs: np.ndarray) -> list[int]:
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        sims = embs @ search.T
        ranks = []
        for i, ci in enumerate(test_correct_idx):
            ranks.append(
                len(search_names) + 1 if ci < 0 else rank_of_correct(sims[i], ci)
            )
        return ranks

    ks = (1, 5, 10)

    zs_embs = np.stack([r["naive"].numpy() for r in test]).astype(np.float32)
    zs_ranks = retrieve(zs_embs)
    print("\n=== Zero-shot (naive embedding, no mapping) ===")
    print(f"  MRR: {mrr(zs_ranks):.3f}")
    print(
        f"  Recall: { {k: round(v, 3) for k, v in recall_at_k(zs_ranks, ks).items()} }"
    )

    X_tr = np.stack([r["naive"].numpy() for r in train]).astype(np.float32)
    Y_tr = np.stack([r["target"].numpy() for r in train]).astype(np.float32)
    reg = Ridge(alpha=args.alpha).fit(X_tr, Y_tr)
    X_te = np.stack([r["naive"].numpy() for r in test]).astype(np.float32)
    pred = reg.predict(X_te)
    rg_ranks = retrieve(pred.astype(np.float32))
    print("\n=== Ridge map (naive -> Latin embedding) ===")
    print(f"  MRR: {mrr(rg_ranks):.3f}")
    print(
        f"  Recall: { {k: round(v, 3) for k, v in recall_at_k(rg_ranks, ks).items()} }"
    )

    print("\n=== Ridge Recall@1/@5 by script ===")
    by_script = defaultdict(list)
    for i, rec in enumerate(test):
        by_script[rec["script"] or "?"].append(rg_ranks[i])
    for script, ranks in sorted(by_script.items()):
        r = recall_at_k(ranks, (1, 5))
        print(f"  {script:12} n={len(ranks):2d}  R@1={r[1]:.3f}  R@5={r[5]:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
