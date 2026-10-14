#!/usr/bin/env python3
"""Compare summary vs Gram-texture features for predicting /Expressive/* tags.

For each Latin family with /Expressive/* coverage we compute two frozen
features from the same rendered Latin glyph set:

  - summary:  FontStyleEmbedder.encode() -> 256-d (attention-pooled)
  - gram:     Gram matrix of the frozen GlyphEncoder features -> 32896-d
              (spatial order destroyed)

Then, per tag, we fit a ridge regression (linear probe) from each feature and
report cross-validated R^2.  Only families that *have* a given tag are used
for that tag (missing values are excluded).  For the Gram, we reduce to 256-d
via PCA (fit on the training fold only) so the two probes have matched
capacity.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hrothgar.googlefonts import GoogleFonts
from hrothgar.style_embedding.config import FontStyleEmbedderConfig
from hrothgar.style_embedding.model import FontStyleEmbedder
from hrothgar.style_embedding.render_utils import render_input_set

EXPRESSIVE_PREFIX = "/Expressive"
MIN_GLYPHS = 8
N_FOLDS = 5
N_COMPONENTS = 256


def gram_features(
    model: FontStyleEmbedder, imgs: torch.Tensor, device: torch.device
) -> np.ndarray:
    with torch.no_grad():
        feat = model.encoder(imgs.to(device))  # (G, F, h, w)
        g, fd, h, w = feat.shape
        feat = feat.reshape(g, fd, h * w)
        gram = torch.bmm(feat, feat.transpose(1, 2)) / (h * w)  # (G, F, F)
        gram = gram.mean(dim=0)
        idx = torch.triu_indices(fd, fd)
        return gram[idx[0], idx[1]].cpu().numpy().astype(np.float32)


def pick_regular(fonts) -> str:
    names = sorted(f.path.name for f in fonts)
    for n in names:
        if "regular" in n.lower():
            return n
    return names[0]


def cv_r2(features: dict[str, np.ndarray], fams: list[str], y: np.ndarray) -> float:
    preds = np.zeros_like(y)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    for tr, te in kf.split(fams):
        X_tr = np.stack([features[fams[i]] for i in tr])
        X_te = np.stack([features[fams[i]] for i in te])
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(X_tr, y[tr])
        preds[te] = model.predict(X_te)
    return float(r2_score(y, preds))


def cv_r2_pca(features: dict[str, np.ndarray], fams: list[str], y: np.ndarray) -> float:
    preds = np.zeros_like(y)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    for tr, te in kf.split(fams):
        X_tr = np.stack([features[fams[i]] for i in tr])
        X_te = np.stack([features[fams[i]] for i in te])
        nc = min(N_COMPONENTS, X_tr.shape[0] - 1, X_tr.shape[1])
        model = make_pipeline(StandardScaler(), PCA(n_components=nc), Ridge(alpha=1.0))
        model.fit(X_tr, y[tr])
        preds[te] = model.predict(X_te)
    return float(r2_score(y, preds))


def cv_r2_combined(summary_feats, gram_feats, fams, y) -> float:
    preds = np.zeros_like(y)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    for tr, te in kf.split(fams):
        S_tr = np.stack([summary_feats[fams[i]] for i in tr])
        S_te = np.stack([summary_feats[fams[i]] for i in te])
        G_tr = np.stack([gram_feats[fams[i]] for i in tr])
        G_te = np.stack([gram_feats[fams[i]] for i in te])
        # Standardize each part independently, then concatenate (no second scaler).
        ss = StandardScaler().fit(S_tr)
        S_tr, S_te = ss.transform(S_tr), ss.transform(S_te)
        nc = min(N_COMPONENTS, G_tr.shape[0] - 1, G_tr.shape[1])
        sc = StandardScaler().fit(G_tr)
        pca = PCA(n_components=nc).fit(sc.transform(G_tr))
        G_tr_pca = pca.transform(sc.transform(G_tr))
        G_te_pca = pca.transform(sc.transform(G_te))
        X_tr = np.concatenate([S_tr, G_tr_pca], axis=1)
        X_te = np.concatenate([S_te, G_te_pca], axis=1)
        model = Ridge(alpha=1.0)
        model.fit(X_tr, y[tr])
        preds[te] = model.predict(X_te)
    return float(r2_score(y, preds))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts"),
    )
    parser.add_argument("--model", default="models/style_embedding_finetune.pth")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--limit", type=int, default=None, help="limit number of families"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    repo = Path(args.repo)

    cfg = FontStyleEmbedderConfig.from_sidecar(args.model)
    model = FontStyleEmbedder(cfg)
    model.load(args.model, device)
    model.to(device)
    model.eval()

    print("Loading Google Fonts …")
    gf = GoogleFonts(repo)
    input_cps = set(cfg.input_codepoints)

    # Group fonts by family, keep families that have Latin + /Expressive/* tags.
    by_family: dict[str, list] = defaultdict(list)
    for font in gf.fonts:
        if not input_cps <= set(font.codepoints):
            continue
        by_family[font.family].append(font)

    reps = []
    for fam, fonts in by_family.items():
        tags = gf.tags.get(fam, {})
        if not any(t.startswith(EXPRESSIVE_PREFIX) for t in tags):
            continue
        reps.append((fam, fonts, tags))

    if args.limit:
        reps = reps[: args.limit]
    print(f"{len(reps)} families with Latin + /Expressive/* tags")

    # Compute features (summary + gram) once.
    summary_feats: dict[str, np.ndarray] = {}
    gram_feats: dict[str, np.ndarray] = {}
    tag_vals: dict[str, dict[str, float]] = {}
    for i, (fam, fonts, tags) in enumerate(reps):
        if i % 100 == 0:
            print(f"  computing features {i}/{len(reps)} …")
        target_name = pick_regular(fonts)
        font = next(f for f in fonts if f.path.name == target_name)
        imgs = render_input_set(font, cfg.input_codepoints, cfg.glyph_size)
        imgs = imgs[~(imgs.amin(dim=(-2, -1)) > 0.995).squeeze(1)]
        if imgs.shape[0] < MIN_GLYPHS:
            continue
        with torch.no_grad():
            summ = (
                model.encode(imgs.unsqueeze(0).to(device))
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        gram = gram_features(model, imgs, device)
        summary_feats[fam] = summ
        gram_feats[fam] = gram
        tag_vals[fam] = {
            t: float(v) for t, v in tags.items() if t.startswith(EXPRESSIVE_PREFIX)
        }

    fams = list(tag_vals.keys())
    print(f"{len(fams)} families with features computed")

    # Distinct tags, ordered by family count (descending).
    tag_fam_count = defaultdict(int)
    for fam in fams:
        for t in tag_vals[fam]:
            tag_fam_count[t] += 1
    tags = sorted(tag_fam_count, key=lambda t: -tag_fam_count[t])

    print(f"\n{'tag':28} {'n':>5} {'R² sum':>8} {'R² gram':>8} {'R² both':>8}")
    print("-" * 60)
    rows_out = []
    for tag in tags:
        use = [f for f in fams if tag in tag_vals[f]]
        n = len(use)
        y = np.array([tag_vals[f][tag] for f in use], dtype=np.float32)
        r2_sum = cv_r2(summary_feats, use, y)
        r2_gram = cv_r2_pca(gram_feats, use, y)
        r2_both = cv_r2_combined(summary_feats, gram_feats, use, y)
        rows_out.append((tag, n, r2_sum, r2_gram, r2_both))
        print(f"{tag:28} {n:>5} {r2_sum:>8.3f} {r2_gram:>8.3f} {r2_both:>8.3f}")

    import numpy as _np

    print("-" * 60)
    print(
        f"{'mean R²':28} {'':>5} {_np.mean([r[2] for r in rows_out]):>8.3f} {_np.mean([r[3] for r in rows_out]):>8.3f} {_np.mean([r[4] for r in rows_out]):>8.3f}"
    )
    print(
        f"{'# best (highest R²)':28} {'':>5} {sum(1 for r in rows_out if r[2] >= r[3] and r[2] >= r[4]):>8} {sum(1 for r in rows_out if r[3] >= r[2] and r[3] >= r[4]):>8} {sum(1 for r in rows_out if r[4] >= r[2] and r[4] >= r[3]):>8}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
