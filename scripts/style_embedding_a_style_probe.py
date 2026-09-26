#!/usr/bin/env python
"""Frozen linear probe: does the style embedding retain the 'a' construction?

The single-story vs double-story 'a' is a discrete, font-wide design choice. If
the frozen style embedding still carries it, then a *linear* probe on the frozen
features should recover it with high AUC — and the blurry "vestigial traces" we
see in the shape head are a decoder artefact (continuous regression averaging the
two modes), not an encoder failure. If the probe AUC is low, the embedding has
genuinely dropped the mode.

Labels come from ``fontquant.csv`` (field ``appearance/lowercase_a_style``, values
``single_story`` / ``double_story`` / empty). Small-caps families ("SC" in the
filename) are ignored, per the data's own caveat.

For each labelled font we render the full glyph set and encode it with the frozen
``FontStyleEmbedder``, then:

* train a frozen linear probe on both the mean summary vector and the flattened
  style-token set, reporting cross-validated ROC-AUC + accuracy; and
* produce a t-SNE scatter of the summary embeddings coloured by construction.

Example::

    PYTHONPATH=Lib python scripts/style_embedding_a_style_probe.py \
        --repo ~/google/fonts_checkout \
        --embedder-path /path/to/style_embedding.pth
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from hrothgar.googlefonts import GoogleFonts
from hrothgar.style_embedding import FontStyleEmbedder, FontStyleEmbedderConfig

SINGLE_STORY = "single_story"
DOUBLE_STORY = "double_story"

# Continuous fontquant appearance columns for the regression probe.  Excludes
# stroke_contrast/*: fontquant's contrast measurement is not trustworthy yet.
DEFAULT_CONTINUOUS_FIELDS = (
    "appearance/weight",
    "appearance/weight_perceptual",
    "appearance/width",
    "appearance/slant",
    "appearance/x_height",
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_embedder(path: str, device: torch.device) -> FontStyleEmbedder:
    config = FontStyleEmbedderConfig.from_sidecar(path)
    embedder = FontStyleEmbedder(config).to(device)
    embedder.load(path, device=device)
    embedder.eval()
    for p in embedder.parameters():
        p.requires_grad = False
    return embedder


def load_a_style_labels(fontquant_path: str) -> tuple[dict[str, str], dict[str, int]]:
    """Return ``{filename: single_story|double_story}`` plus skip counts."""
    labels: dict[str, str] = {}
    stats = {"total_rows": 0, "sc_skipped": 0, "empty": 0, "other": 0}
    with open(fontquant_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["total_rows"] += 1
            name = (row.get("Font") or "").strip()
            if not name:
                continue
            # Small-caps families: the lowercase_a_style value is not meaningful.
            if "SC" in name:
                stats["sc_skipped"] += 1
                continue
            value = (row.get("appearance/lowercase_a_style") or "").strip()
            if value in (SINGLE_STORY, DOUBLE_STORY):
                labels[Path(name).name] = value
            elif not value:
                stats["empty"] += 1
            else:
                stats["other"] += 1
    return labels, stats


def load_continuous_labels(
    fontquant_path: str, fields: list[str]
) -> dict[str, dict[str, float]]:
    """Return ``{filename: {field: float}}`` for the requested continuous fields.

    Empty / non-numeric values are dropped per field; "SC" (small caps) filenames
    are skipped.
    """
    data: dict[str, dict[str, float]] = {}
    with open(fontquant_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("Font") or "").strip()
            if not name or "SC" in name:
                continue
            vals: dict[str, float] = {}
            for field in fields:
                raw = (row.get(field) or "").strip()
                if not raw:
                    continue
                try:
                    vals[field] = float(raw)
                except ValueError:
                    continue
            if vals:
                data[Path(name).name] = vals
    return data


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def _render_full_set(
    font, embedder: FontStyleEmbedder
) -> tuple[torch.Tensor | None, str | None]:
    """Render the full input glyph set as ``(G, 1, H, W)``.

    Returns ``(image, reason)`` where ``image`` is ``None`` (with a human-readable
    ``reason``) if any glyph is blank or fails to render.
    """
    size = embedder.config.glyph_size
    glyphs = []
    for cp in embedder.config.input_codepoints:
        try:
            arr = font.render(cp, size=size)
        except Exception as exc:
            return None, f"render exception cp=U+{cp:04X}: {exc}"
        gray = arr[0].copy() if arr.ndim == 3 else np.asarray(arr, dtype=np.float32)
        if gray.ndim != 2 or gray.size == 0:
            return None, f"bad render shape {getattr(arr, 'shape', None)} cp=U+{cp:04X}"
        if gray.min() > 0.995:  # blank glyph
            return None, f"blank glyph cp={chr(cp)!r} (U+{cp:04X})"
        glyphs.append(torch.from_numpy(gray))
    return torch.stack(glyphs).unsqueeze(1), None  # (G, 1, H, W)


def encode_fonts(
    fonts,
    embedder: FontStyleEmbedder,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, list[int], list[tuple[int, str, str]]]:
    """Encode the full glyph set per font.

    Returns:
        summary: ``(N, F)``.
        latents_flat: ``(N, K*F)``.
        kept_idx: positions in ``fonts`` that rendered successfully.
        skipped: ``(index, filename, reason)`` for fonts dropped from the batch.
    """
    summaries: list[np.ndarray] = []
    latents_flat: list[np.ndarray] = []
    kept_idx: list[int] = []
    skipped: list[tuple[int, str, str]] = []

    for start in range(0, len(fonts), batch_size):
        chunk = fonts[start : start + batch_size]
        images = []
        idxs = []
        for k, font in enumerate(chunk):
            im, reason = _render_full_set(font, embedder)
            if im is None:
                skipped.append((start + k, font.path.name, reason or "unknown"))
                continue
            images.append(im)
            idxs.append(start + k)

        if not images:
            continue

        # (B, G, 1, H, W)
        batch = torch.stack(images).to(device, dtype=torch.float32)
        with torch.no_grad():
            summary, latents, _ = embedder.encode_with_tokens(batch)
        summaries.append(summary.cpu().numpy())
        # (B, K, F) → (B, K*F)
        latents_flat.append(latents.cpu().numpy().reshape(latents.shape[0], -1))
        kept_idx.extend(idxs)

    if not summaries:
        f_dim = embedder.config.encoder_feature_dim
        k_dim = embedder.config.style_latents * f_dim
        return np.empty((0, f_dim)), np.empty((0, k_dim)), [], skipped

    return (
        np.concatenate(summaries, axis=0).astype(np.float64),
        np.concatenate(latents_flat, axis=0).astype(np.float64),
        kept_idx,
        skipped,
    )


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


def linear_probe(
    X: np.ndarray, y: np.ndarray, n_splits: int, seed: int
) -> tuple[float, float, float, float]:
    """Stratified-K-fold frozen linear probe → (auc_mean, auc_std, acc_mean, acc_std)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    n_splits = min(n_splits, int(np.bincount(y).min()))
    if n_splits < 2:
        raise ValueError(
            f"Not enough samples in the minority class for CV (counts={np.bincount(y)})"
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs, accs = [], []
    for tr, te in skf.split(X, y):
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(scaler.transform(X[tr]), y[tr])
        Xte = scaler.transform(X[te])
        proba = clf.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(y[te], proba))
        accs.append(accuracy_score(y[te], clf.predict(Xte)))
    return (
        float(np.mean(aucs)),
        float(np.std(aucs)),
        float(np.mean(accs)),
        float(np.std(accs)),
    )


def continuous_probe(
    X: np.ndarray, y: np.ndarray, n_splits: int, seed: int
) -> tuple[float, float, float, float]:
    """K-fold frozen ridge regression → (r2_mean, r2_std, r_mean, r_std).

    Ridge (alpha selected per fold via GCV) rather than plain OLS: the flattened
    latents have more features (K*F ≈ 4096) than training rows, where OLS
    interpolates training noise and produces meaningless held-out R².
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    alphas = np.logspace(-2, 4, 25)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    r2s, rs = [], []
    for tr, te in kf.split(X):
        scaler = StandardScaler().fit(X[tr])
        reg = RidgeCV(alphas=alphas).fit(scaler.transform(X[tr]), y[tr])
        pred = reg.predict(scaler.transform(X[te]))
        r2s.append(r2_score(y[te], pred))
        rs.append(
            float(np.corrcoef(y[te], pred)[0, 1])
            if np.std(pred) > 0 and np.std(y[te]) > 0
            else 0.0
        )
    return (
        float(np.mean(r2s)),
        float(np.std(r2s)),
        float(np.mean(rs)),
        float(np.std(rs)),
    )


def tsne_plot(
    X: np.ndarray,
    y: np.ndarray,
    out_path: str,
    perplexity: int,
    seed: int,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)
    n = Xs.shape[0]
    perp = min(perplexity, max(5, n - 1))
    tsne = TSNE(n_components=2, perplexity=perp, init="pca", random_state=seed)
    Z = tsne.fit_transform(Xs)

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, name, color in [
        (0, "single_story", "#1f77b4"),
        (1, "double_story", "#d62728"),
    ]:
        mask = y == label
        ax.scatter(
            Z[mask, 0],
            Z[mask, 1],
            s=18,
            alpha=0.7,
            c=color,
            label=f"{name} (n={int(mask.sum())})",
        )
    ax.set_title(f"{title}\nN={n}")
    ax.legend(frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved t-SNE plot to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, help="Path to the Google Fonts checkout.")
    p.add_argument(
        "--embedder-path",
        required=True,
        help="Path to the frozen FontStyleEmbedder checkpoint (needs a .conf.json sidecar).",
    )
    p.add_argument(
        "--fontquant",
        default="fontquant.csv",
        help="Path to fontquant.csv (default: fontquant.csv in the CWD).",
    )
    p.add_argument(
        "--continuous-fields",
        default=",".join(DEFAULT_CONTINUOUS_FIELDS),
        help="Comma-separated fontquant columns for the continuous regression "
        "probe (empty string disables). Default: weight, weight_perceptual, "
        "width, slant, x_height.",
    )
    p.add_argument(
        "--max-fonts", type=int, default=None, help="Cap the number of fonts."
    )
    p.add_argument("--cv-folds", type=int, default=5, help="Stratified CV folds.")
    p.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device",
        default=None,
        help="Torch device (default: cuda if available, else cpu).",
    )
    p.add_argument(
        "--out-dir",
        default="outputs",
        help="Directory for the t-SNE plot(s).",
    )
    p.add_argument(
        "--tsne-latents",
        action="store_true",
        help="Also produce a t-SNE plot of the flattened style-token set.",
    )
    p.add_argument(
        "--save-features",
        default=None,
        help="Optional .npz path to dump features + labels.",
    )
    args = p.parse_args()

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("Loading labels from fontquant.csv …")
    labels, stats = load_a_style_labels(args.fontquant)
    print(
        f"  rows={stats['total_rows']}  sc_skipped={stats['sc_skipped']}  "
        f"empty={stats['empty']}  labelled={len(labels)}"
    )

    cont_fields = [
        f.strip() for f in (args.continuous_fields or "").split(",") if f.strip()
    ]
    cont_data = (
        load_continuous_labels(args.fontquant, cont_fields) if cont_fields else {}
    )

    print("Loading embedder …")
    embedder = load_embedder(args.embedder_path, device)

    print("Loading fonts …")
    gf = GoogleFonts(args.repo, having=set(embedder.config.input_codepoints))
    font_by_name = {font.path.name: font for font in gf.fonts}

    # Intersect labels with fonts present in the checkout, preserving order.
    matched = []
    for name, label in labels.items():
        font = font_by_name.get(name)
        if font is not None:
            matched.append((name, font, label))
    if args.max_fonts is not None:
        matched = matched[: args.max_fonts]
    print(
        f"  labelled fonts in repo: {len(matched)} / {len(labels)} "
        f"(repo has {len(gf.fonts)} usable fonts)"
    )
    if not matched:
        raise SystemExit(
            "No labelled fonts matched the checkout. Check --repo / --fontquant; "
            "filenames must match fontquant.csv's 'Font' column exactly."
        )

    print(f"Encoding {len(matched)} fonts …")
    fonts = [f for _, f, _ in matched]
    X_summary, X_latents, kept_idx, skipped = encode_fonts(fonts, embedder, device)

    if skipped:
        print(f"  skipped {len(skipped)} fonts that failed to render a full glyph set:")
        for _idx, name, reason in skipped[:20]:
            print(f"    - {name}: {reason}")
        if len(skipped) > 20:
            print(f"    … and {len(skipped) - 20} more")

    if not kept_idx:
        raise SystemExit("No fonts rendered successfully; cannot run the probe.")

    matched = [matched[i] for i in kept_idx]

    y = np.array(
        [0 if label == SINGLE_STORY else 1 for _, _, label in matched],
        dtype=np.int64,
    )
    counts = np.bincount(y)
    majority_acc = float(counts.max() / counts.sum())
    print(f"\nClass counts: single_story={counts[0]}  double_story={counts[1]}")
    print(f"Majority-class baseline accuracy: {majority_acc:.3f}\n")

    print("Linear probe (frozen features, stratified CV):")
    for name, X in [("summary", X_summary), ("latents (flattened)", X_latents)]:
        auc_m, auc_s, acc_m, acc_s = linear_probe(X, y, args.cv_folds, args.seed)
        print(
            f"  {name:<22} AUC={auc_m:.3f}±{auc_s:.3f}   "
            f"acc={acc_m:.3f}±{acc_s:.3f}"
        )

    if cont_fields:
        print(
            f"\nContinuous linear probe (frozen features, {args.cv_folds}-fold KFold):"
        )
        names = [n for n, _, _ in matched]
        for field in cont_fields:
            idxs = []
            vals = []
            for i, n in enumerate(names):
                v = cont_data.get(n, {}).get(field)
                if v is not None:
                    idxs.append(i)
                    vals.append(v)
            if len(idxs) < 2 * args.cv_folds:
                print(f"  {field:<30} n={len(idxs)}  (too few values; skipped)")
                continue
            y_cont = np.asarray(vals, dtype=np.float64)
            parts = []
            for label, X in [("summary", X_summary), ("latents", X_latents)]:
                r2_m, r2_s, r_m, _ = continuous_probe(
                    X[idxs], y_cont, args.cv_folds, args.seed
                )
                parts.append(f"{label} R2={r2_m:.3f}±{r2_s:.3f} r={r_m:.3f}")
            print(f"  {field:<30} n={len(idxs):<5} " + "   ".join(parts))

    # t-SNE on the summary (the compact global vector of interest).
    tsne_plot(
        X_summary,
        y,
        out_path=str(Path(args.out_dir) / "a_style_tsne_summary.png"),
        perplexity=args.perplexity,
        seed=args.seed,
        title="t-SNE of frozen style summary, coloured by 'a' construction",
    )
    if args.tsne_latents:
        tsne_plot(
            X_latents,
            y,
            out_path=str(Path(args.out_dir) / "a_style_tsne_latents.png"),
            perplexity=args.perplexity,
            seed=args.seed,
            title="t-SNE of frozen style latents, coloured by 'a' construction",
        )

    if args.save_features:
        np.savez(
            args.save_features,
            summary=X_summary,
            latents=X_latents,
            y=y,
            fonts=np.array([n for n, _, _ in matched]),
        )
        print(f"\nSaved features + labels to {args.save_features}")


if __name__ == "__main__":
    main()
