#!/usr/bin/env python3
"""Trainable compact Gram-style encoder for cross-script font pairing.

Architecture (unchanged):
    frozen GlyphEncoder -> trainable 1x1 conv (256 -> K) -> Gram pooling
        -> flatten upper-triangle -> L2-normalize -> compact embedding

Training now uses *file-level* Custom data (every weight/style of every
co-designed family is a training sample), with **family-aware InfoNCE**: the
negative set masks out other files from the *same* family, so different weights
of one family are never treated as negatives (matching the repo's own
contrastive loss convention).

Evaluation stays family-level (canonical Regular file per family).
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


class GramStyleEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, gram_channels: int = 32):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.channel_proj = nn.Conv2d(256, gram_channels, 1, bias=False)
        self.gram_channels = gram_channels
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


def info_nce_family(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    families: list[str],
    temperature: float,
) -> torch.Tensor:
    """Symmetric InfoNCE with same-family negative masking."""
    logits = anchors @ positives.T / temperature  # (B, B)
    fam = np.asarray(families)
    same = torch.tensor(fam[:, None] == fam[None, :], device=logits.device)
    mask = same & ~torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(mask, -1e9)
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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save", default="models/gram_style_encoder.pt")
    parser.add_argument("--csv-out", default="gram_pairing_predictions.csv")
    args = parser.parse_args()

    device = torch.device(args.device)
    repo = Path(args.repo)

    cfg = FontStyleEmbedderConfig.from_sidecar(args.model)
    base = FontStyleEmbedder(cfg)
    base.load(args.model, device)
    base.to(device)
    base.eval()

    model = GramStyleEncoder(base.encoder, gram_channels=args.gram_channels).to(device)

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

    # file-level training items (Custom) + family-level eval records.
    train_items: list[dict] = []
    eval_recs: list[dict] = []
    for fam, info in by_family.items():
        if info["origin"] == "Custom":
            for fname in info["fonts"]:
                path = filename_to_path.get(fname)
                if path is None:
                    continue
                font = StandaloneFont(path)
                code = detect_script(font)
                if code not in TARGET_SCRIPTS:
                    continue
                train_items.append(
                    {
                        "family": fam,
                        "font_name": fname,
                        "path": path,
                        "letters": nonlatin_letters(font, code),
                    }
                )
        else:  # Designed to match Latin -> family-level canonical
            canon = pick_canonical(info["fonts"])
            path = filename_to_path.get(canon)
            companion = (manifest.get(canon) or {}).get("companion")
            if path is None or companion is None:
                continue
            font = StandaloneFont(path)
            code = detect_script(font)
            if code not in TARGET_SCRIPTS:
                continue
            eval_recs.append(
                {
                    "family": fam,
                    "path": path,
                    "companion": companion,
                    "companion_name": info["latin_family"],
                    "letters": nonlatin_letters(font, code),
                    "origin": "Designed to match Latin",
                }
            )

    # Custom canonical records for the eval search set.
    custom_canonical_seen = set()
    for fam, info in by_family.items():
        if info["origin"] != "Custom":
            continue
        canon = pick_canonical(info["fonts"])
        path = filename_to_path.get(canon)
        if path is None:
            continue
        font = StandaloneFont(path)
        code = detect_script(font)
        if code not in TARGET_SCRIPTS:
            continue
        eval_recs.append(
            {
                "family": fam,
                "path": path,
                "companion": path,
                "companion_name": fam,
                "letters": nonlatin_letters(font, code),
                "origin": "Custom",
            }
        )

    if args.limit:
        train_items = train_items[: args.limit]

    print(
        f"Pre-rendering {len(train_items)} train files + {len(eval_recs)} eval families …"
    )

    # --- train data (file-level) ---
    train_data: dict[str, dict] = {}
    for item in train_items:
        font = StandaloneFont(item["path"])
        nl = render_set(font, item["letters"], cfg.glyph_size)
        lat = render_set(font, cfg.input_codepoints, cfg.glyph_size)
        if nl is not None and lat is not None:
            train_data[item["font_name"]] = {
                "nonlatin": nl.to(device),
                "latin": lat.to(device),
                "family": item["family"],
            }
    print(f"  {len(train_data)} train files")

    # --- eval data (family-level) ---
    eval_data: dict[str, dict] = {}
    for rec in eval_recs:
        nl = render_set(StandaloneFont(rec["path"]), rec["letters"], cfg.glyph_size)
        lat = render_set(
            StandaloneFont(rec["companion"]), cfg.input_codepoints, cfg.glyph_size
        )
        if nl is not None and lat is not None:
            eval_data[rec["family"]] = {
                "nonlatin": nl.to(device),
                "latin": lat.to(device),
                "companion_name": rec["companion_name"],
                "origin": rec["origin"],
            }
    print(f"  {len(eval_data)} eval families")

    # --- Train ---
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    keys = list(train_data.keys())
    model.train()
    for epoch in range(args.epochs):
        perm = np.random.permutation(len(keys))
        total, nb = 0.0, 0
        for start in range(0, len(keys), args.batch_size):
            batch = [keys[i] for i in perm[start : start + args.batch_size]]
            a = torch.stack([model(train_data[k]["nonlatin"]) for k in batch])
            p = torch.stack([model(train_data[k]["latin"]) for k in batch])
            fams = [train_data[k]["family"] for k in batch]
            loss = info_nce_family(a, p, fams, args.temperature)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item()
            nb += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}  loss={total / max(nb,1):.4f}")

    # --- Eval ---
    model.eval()
    with torch.no_grad():
        search_names, search_embs, seen = [], [], {}
        for fam, d in eval_data.items():
            c = d["companion_name"]
            if c and c not in seen:
                seen[c] = len(search_names)
                search_names.append(c)
                search_embs.append(model(d["latin"]))
        search = torch.stack(search_embs)
        mu = search.mean(dim=0)
        search = F.normalize(search - mu, p=2, dim=-1)

        queries = {
            fam: F.normalize(model(d["nonlatin"]) - mu, p=2, dim=-1)
            for fam, d in eval_data.items()
        }
        idx = {name: i for i, name in enumerate(search_names)}

        def ranks_for(origin):
            order = [f for f, d in eval_data.items() if d["origin"] == origin]
            out = []
            for f in order:
                sims = search @ queries[f]
                ci = idx.get(eval_data[f]["companion_name"], -1)
                out.append(
                    int((sims.argsort(descending=True) == ci).nonzero()[0].item()) + 1
                    if ci >= 0
                    else len(search_names) + 1
                )
            return out

        print("\n=== Retrieval (file-level trained Gram, mean-centered) ===")
        report(ranks_for("Custom"), "Custom (sanity)", len(search_names))
        report(
            ranks_for("Designed to match Latin"),
            "Designed to match Latin",
            len(search_names),
        )

    # --- Save ---
    torch.save(
        {
            "gram_channels": args.gram_channels,
            "channel_proj": model.channel_proj.state_dict(),
        },
        args.save,
    )
    print(f"Saved to {args.save}")

    # --- Full catalog ---
    print("Building full Latin catalog …")
    catalog = build_latin_catalog(repo, set(cfg.input_codepoints))
    print(f"  {len(catalog)} Latin families")
    cat_names, cat_embs = [], []
    for name, path in catalog:
        imgs = render_set(StandaloneFont(path), cfg.input_codepoints, cfg.glyph_size)
        if imgs is None:
            continue
        with torch.no_grad():
            cat_embs.append(model(imgs.to(device)))
        cat_names.append(name)
    cat = torch.stack(cat_embs)
    mu_cat = cat.mean(dim=0)
    cat = F.normalize(cat - mu_cat, p=2, dim=-1)
    cat_idx = {n: i for i, n in enumerate(cat_names)}

    csv_rows, ranks = [], []
    test_order = [
        f for f, d in eval_data.items() if d["origin"] == "Designed to match Latin"
    ]
    for fam in test_order:
        with torch.no_grad():
            q = F.normalize(model(eval_data[fam]["nonlatin"]) - mu_cat, p=2, dim=-1)
        sims = cat @ q
        order = sims.argsort(descending=True)
        gt = eval_data[fam]["companion_name"]
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
