#!/usr/bin/env python3
"""Fused structure+texture encoder for cross-script font pairing.

Two branches, both fed the same rendered glyph set:

  - texture:  frozen GlyphEncoder -> trainable 1x1 conv (256->K) -> Gram
  - structure: frozen FontStyleEmbedder summary (256-d)  [attention-pooled]

Each is projected to a common dimension, then combined with a *learned convex
combination* (softmax gating) and L2-normalized.  This is deliberately NOT a
naive concatenation: the two modalities are projected to the same dimension
first, and the gating is two parameters, so it cannot overfit the way a big
concat+ridge did.  We report the learned gating weights so we can see whether
structure actually helps cross-script.
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
import torch.nn as nn
import torch.nn.functional as F
from fontTools.unicodedata import script as codepoint_script

from hrothgar.googlefonts import StandaloneFont
from hrothgar.style_embedding.config import FontStyleEmbedderConfig
from hrothgar.style_embedding.model import FontStyleEmbedder
from hrothgar.style_embedding.render_utils import render_input_set

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


class GramTextureEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, gram_channels: int = 32):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.channel_proj = nn.Conv2d(256, gram_channels, 1, bias=False)
        self.gram_channels = gram_channels
        self.gram_dim = gram_channels * (gram_channels + 1) // 2
        self._idx = torch.triu_indices(gram_channels, gram_channels)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(imgs)  # (G, 256, h, w)
        feat = self.channel_proj(feat)  # (G, K, h, w)
        g, k, h, w = feat.shape
        feat = feat.reshape(g, k, h * w)
        gram = torch.bmm(feat, feat.transpose(1, 2)) / (h * w)
        gram = gram.mean(dim=0)
        tri = gram[self._idx[0], self._idx[1]]
        return F.normalize(tri, p=2, dim=0)


class FusedGramEncoder(nn.Module):
    def __init__(
        self, base: FontStyleEmbedder, texture: GramTextureEncoder, fusion_dim: int = 64
    ):
        super().__init__()
        self.base = base
        self.texture = texture
        for p in self.base.parameters():
            p.requires_grad = False
        self.proj_summary = nn.Linear(256, fusion_dim, bias=False)
        self.proj_gram = nn.Linear(texture.gram_dim, fusion_dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(2))

    def forward(self, imgs: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        gram = self.texture(imgs)  # (gram_dim,)
        zs = F.normalize(self.proj_summary(summary), p=2, dim=-1)
        zg = F.normalize(self.proj_gram(gram), p=2, dim=-1)
        w = torch.softmax(self.gate, dim=0)
        return F.normalize(w[0] * zs + w[1] * zg, p=2, dim=-1)

    def gate_weights(self) -> tuple[float, float]:
        w = torch.softmax(self.gate.detach(), dim=0)
        return float(w[0]), float(w[1])


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


def render_set(font: StandaloneFont, cps: list[int], size: int) -> torch.Tensor | None:
    imgs = render_input_set(font, cps, size)
    imgs = imgs[~(imgs.amin(dim=(-2, -1)) > 0.995).squeeze(1)]
    if imgs.shape[0] < MIN_GLYPHS:
        return None
    return imgs


def pick_canonical(fonts: list[str]) -> str:
    fonts = sorted(fonts)
    for f in fonts:
        if "regular" in f.lower():
            return f
    return fonts[0]


def info_nce(a: torch.Tensor, p: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = a @ p.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def report(ranks: list[int], label: str, k_search: int) -> None:
    ks = (1, 5, 10)
    rec = {k: sum(1 for r in ranks if r <= k) / len(ranks) for k in ks}
    mrr = float(np.mean([1.0 / r for r in ranks]))
    print(
        f"{label:28} n={len(ranks):3d}  MRR={mrr:.3f}  "
        f"R@1={rec[1]:.3f} @5={rec[5]:.3f} @10={rec[10]:.3f}  (K={k_search})"
    )


def build_latin_catalog(repo: Path, latin_cps: set[int]) -> list[tuple[str, str]]:
    from fontTools.ttLib import TTFont
    from gftools.util.google_fonts import Metadata

    base = repo / "ofl" if (repo / "ofl").is_dir() else repo
    out = []
    for pb in sorted(base.glob("*/METADATA.pb")):
        try:
            m = Metadata(str(pb))
        except Exception:
            continue
        if not m.fonts:
            continue
        entry = min(m.fonts, key=lambda f: (abs(int(f.weight) - 400), int(f.weight)))
        path = pb.parent / entry.filename
        if not path.exists():
            continue
        try:
            cmap = TTFont(str(path), lazy=True).getBestCmap()
            if not all(cp in cmap for cp in latin_cps):
                continue
        except Exception:
            continue
        out.append((m.name, str(path)))
    return out


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
    parser.add_argument("--gram-channels", type=int, default=32)
    parser.add_argument("--fusion-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save", default="models/gram_fused_encoder.pt")
    parser.add_argument("--csv-out", default="gram_fused_predictions.csv")
    args = parser.parse_args()

    device = torch.device(args.device)
    repo = Path(args.repo)

    cfg = FontStyleEmbedderConfig.from_sidecar(args.model)
    base = FontStyleEmbedder(cfg)
    base.load(args.model, device)
    base.to(device)
    base.eval()

    texture = GramTextureEncoder(base.encoder, gram_channels=args.gram_channels).to(
        device
    )
    model = FusedGramEncoder(base, texture, fusion_dim=args.fusion_dim).to(device)

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
        if origin not in ("Custom", "Designed to match Latin"):
            continue
        fam = (r.get("Family") or "").strip()
        by_family[fam]["fonts"].append((r.get("Font") or "").strip())
        by_family[fam]["origin"] = origin
        by_family[fam]["latin_family"] = (r.get("Latin family") or "").strip()

    train_fams, test_fams = [], []
    for fam, info in by_family.items():
        canon = pick_canonical(info["fonts"])
        path = filename_to_path.get(canon)
        companion = (manifest.get(canon) or {}).get("companion")
        if path is None or companion is None:
            continue
        font = StandaloneFont(path)
        code = detect_script(font)
        if code not in TARGET_SCRIPTS:
            continue
        rec = {
            "family": fam,
            "font_name": canon,
            "path": path,
            "script": TARGET_SCRIPTS[code],
            "letters": nonlatin_letters(font, code),
            "companion": companion,
            "companion_name": (
                fam if info["origin"] == "Custom" else info["latin_family"]
            ),
        }
        (train_fams if info["origin"] == "Custom" else test_fams).append(rec)

    if args.limit:
        train_fams = train_fams[: args.limit]
        test_fams = test_fams[: args.limit]

    def pre_render(recs):
        data = {}
        for rec in recs:
            nl = render_set(StandaloneFont(rec["path"]), rec["letters"], cfg.glyph_size)
            lat = render_set(
                StandaloneFont(rec["companion"]), cfg.input_codepoints, cfg.glyph_size
            )
            if nl is None or lat is None:
                continue
            with torch.no_grad():
                nl_sum = base.encode(nl.unsqueeze(0).to(device)).squeeze(0)
                lat_sum = base.encode(lat.unsqueeze(0).to(device)).squeeze(0)
            data[rec["family"]] = {
                "nonlatin": nl.to(device),
                "latin": lat.to(device),
                "nonlatin_sum": nl_sum,
                "latin_sum": lat_sum,
                "companion_name": rec["companion_name"],
            }
        return data

    print(f"Pre-rendering {len(train_fams)} train + {len(test_fams)} test families …")
    train_data = pre_render(train_fams)
    test_data = pre_render(test_fams)
    print(f"  {len(train_data)} train, {len(test_data)} test rendered")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    fam_keys = list(train_data.keys())
    model.train()
    for epoch in range(args.epochs):
        perm = np.random.permutation(len(fam_keys))
        total, nb = 0.0, 0
        for start in range(0, len(fam_keys), args.batch_size):
            batch = [fam_keys[i] for i in perm[start : start + args.batch_size]]
            a = torch.stack(
                [
                    model(train_data[f]["nonlatin"], train_data[f]["nonlatin_sum"])
                    for f in batch
                ]
            )
            p = torch.stack(
                [
                    model(train_data[f]["latin"], train_data[f]["latin_sum"])
                    for f in batch
                ]
            )
            loss = info_nce(a, p, args.temperature)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item()
            nb += 1
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}  loss={total / max(nb,1):.4f}")

    ws, wg = model.gate_weights()
    print(f"Learned gate weights: structure={ws:.3f}  texture={wg:.3f}")

    # ---- Eval ----
    model.eval()
    with torch.no_grad():
        search_names, search_embs, seen = [], [], {}
        for fam, d in {**train_data, **test_data}.items():
            c = d["companion_name"]
            if c and c not in seen:
                seen[c] = len(search_names)
                search_names.append(c)
                search_embs.append(model(d["latin"], d["latin_sum"]))
        search = torch.stack(search_embs)
        mu = search.mean(dim=0)
        search = F.normalize(search - mu, p=2, dim=-1)

        queries = {
            fam: F.normalize(model(d["nonlatin"], d["nonlatin_sum"]) - mu, p=2, dim=-1)
            for fam, d in {**train_data, **test_data}.items()
        }
        idx = {name: i for i, name in enumerate(search_names)}

        def ranks_for(data, order):
            out = []
            for f in order:
                sims = search @ queries[f]
                ci = idx.get(data[f]["companion_name"], -1)
                out.append(
                    int((sims.argsort(descending=True) == ci).nonzero()[0].item()) + 1
                    if ci >= 0
                    else len(search_names) + 1
                )
            return out

        print("\n=== Retrieval (fused, mean-centered) ===")
        report(
            ranks_for(train_data, list(train_data)),
            "Custom (sanity)",
            len(search_names),
        )
        report(
            ranks_for(test_data, list(test_data)),
            "Designed to match Latin",
            len(search_names),
        )

    # Full catalog
    torch.save(
        {
            "gram_channels": args.gram_channels,
            "fusion_dim": args.fusion_dim,
            "texture": texture.channel_proj.state_dict(),
            "proj_summary": model.proj_summary.state_dict(),
            "proj_gram": model.proj_gram.state_dict(),
            "gate": model.gate.detach(),
        },
        args.save,
    )
    print(f"Saved to {args.save}")

    print("Building full Latin catalog …")
    catalog = build_latin_catalog(repo, set(cfg.input_codepoints))
    print(f"  {len(catalog)} Latin families")
    cat_names, cat_embs = [], []
    for name, path in catalog:
        imgs = render_set(StandaloneFont(path), cfg.input_codepoints, cfg.glyph_size)
        if imgs is None:
            continue
        with torch.no_grad():
            summ = base.encode(imgs.unsqueeze(0).to(device)).squeeze(0)
            cat_embs.append(model(imgs.to(device), summ))
        cat_names.append(name)
    cat = torch.stack(cat_embs)
    mu_cat = cat.mean(dim=0)
    cat = F.normalize(cat - mu_cat, p=2, dim=-1)
    cat_idx = {n: i for i, n in enumerate(cat_names)}

    csv_rows, ranks = [], []
    for fam in test_data:
        with torch.no_grad():
            q = F.normalize(
                model(test_data[fam]["nonlatin"], test_data[fam]["nonlatin_sum"])
                - mu_cat,
                p=2,
                dim=-1,
            )
        sims = cat @ q
        order = sims.argsort(descending=True)
        gt = test_data[fam]["companion_name"]
        gt_rank = (
            int((order == cat_idx[gt]).nonzero()[0].item()) + 1 if gt in cat_idx else -1
        )
        top5 = [cat_names[i] for i in order[:5].tolist()]
        ranks.append(gt_rank if gt_rank > 0 else len(cat_names) + 1)
        csv_rows.append(
            {
                "Non-Latin": fam,
                "Designed-to-match GT": gt,
                "Designed-to-match Predicted": top5[0],
                "GT rank": gt_rank,
                "Top-5": "; ".join(top5),
            }
        )

    report(ranks, "Full catalog (Designed to match)", len(cat_names))
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "Non-Latin",
                "Designed-to-match GT",
                "Designed-to-match Predicted",
                "GT rank",
                "Top-5",
            ],
        )
        w.writeheader()
        w.writerows(csv_rows)
    print(f"Wrote {args.csv_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
