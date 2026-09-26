#!/usr/bin/env python
"""Parametric-axis sensitivity probe for the style-extraction model.

A parametric variable font exposes *ground-truth, continuous* fine-detail labels:
its design-space axes.  By rendering the same font at swept axis positions and
passing each through the frozen ``StyleExtractionModelV2``, we can ask a much
cleaner question than the cross-font probes: does the model *track* a controlled
change in one fine axis (terminal rounding, inktrap size, straight↔rounded
counter) with everything else held fixed?

For each requested axis we sweep ``--samples`` points across its range (all
other axes at their default), then:

* encode the evidence glyphs at each point into the K style tokens,
* decode a few target codepoints from those tokens,
* run a frozen ridge regression to recover the axis value from (i) the
  *mean-pooled* style vector and (ii) the *full token set* (PCA-reduced), and
* quantify whether the *decoded glyphs* track the axis via an L1 distance matrix
  (``top1`` / ``diag/off``), plus a GT-vs-recon montage.

Interpretation
--------------
* ``summ R²`` ≈ 1 on a *global* axis (e.g. ``slnt``) is expected — a global
  summary should capture a global shear.
* On a *local fine-detail* axis (``ROND``, ``YTTL``+``XTTW``, ``ST**``) the
  token-set R² should beat the mean R² if the Perceiver is producing per-token
  structure; both ≈ 0 means the fine detail is not yet in the representation.
* ``trk@1`` well above ``1/N`` and ``d/o < 1`` mean the *decoder* is using that
  detail (reconstructions move with the axis); ≈ chance / ≈ 1 means it is not.
* ``smapR2`` / ``smapfR2`` (v4 only) are the axis R² of the *pre-SPADE* style
  map (mean-pooled / flattened+PCA).  These localise the bottleneck: if
  ``tok R2`` is high but ``smapfR2`` is low, the cross-attention read-out is the
  killer; if ``smapfR2`` is high but ``d/o`` is still ≈ 1, the SPADE/CNN (or the
  coarse skeleton) is the killer.  ``coarseR2`` is the axis R² of the stage-1
  coarse image; ``coarseD/O`` is its *visible* tracking (diag/off L1), which
  distinguishes "the coarse image visibly tracks the axis but the fine head
  destroys it" from "the axis signal is tiny everywhere and never becomes
  visible".

Example::

    PYTHONPATH=Lib python scripts/style_extraction_axis_probe.py \\
        --model-path models/style_extraction.pth \\
        --fonts "RobotoDelta-Roman-VF.ttf:STLI+STLO+STUI+STUO,YTTL+XTTW" \\
        --fonts "YouTubeMarquee[BASE,GRAD,ROND,SCAL,TANG,TRML,WDSP,XINK,XOFI,XOJN,XOLC,XOPQ,XORN,XOUC,XTFI,XTLC,XTLR,XTRA,XTSP,XTUC,YOFI,YOLC,YOPQ,YOUC,YTDE,YTLC,opsz,wdth,wght].ttf:ROND"

Axes joined with ``+`` are swept *together* (a shared normalised coordinate
mapped into each axis's own range); comma-separated axes are swept individually.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from hrothgar.googlefonts import StandaloneFont
from hrothgar.style_extraction import load_model
from hrothgar.style_extraction.render_utils import render_glyph

# Axes swept by default when a font is given without an explicit `:axes` list.
# Only axes actually present in the font's fvar table are kept.
DEFAULT_AXES = [
    "slnt",
    "YOPQ",
    "XOPQ",
    "ROND",
    "GRAD",
    "STLI",
    "STLO",
    "STUI",
    "STUO",
    "YTTL",
    "XTTW",
]


def read_fvar(
    path: str | Path,
) -> tuple[list[str], list[float], list[float], list[float]]:
    """Return ``(tags, defaults, mins, maxs)`` in fvar order."""
    from fontTools.ttLib import TTFont

    axes = TTFont(path)["fvar"].axes
    tags = [a.axisTag for a in axes]
    defaults = [float(a.defaultValue) for a in axes]
    mins = [float(a.minValue) for a in axes]
    maxs = [float(a.maxValue) for a in axes]
    return tags, defaults, mins, maxs


def _parse_font_spec(spec: str) -> tuple[str, list[list[str]] | None]:
    if ":" in spec:
        path, axes_str = spec.split(":", 1)
        groups = []
        for item in axes_str.split(","):
            item = item.strip()
            if not item:
                continue
            groups.append([t.strip() for t in item.split("+") if t.strip()])
        return path, groups
    return spec, None


def build_sweep(
    group: list[str],
    tags: list[str],
    mins: list[float],
    maxs: list[float],
    n: int,
) -> tuple[list[dict[str, float]] | None, np.ndarray | None, str]:
    valid = [t for t in group if t in tags]
    if len(group) == 1 and valid:
        tag = valid[0]
        idx = tags.index(tag)
        lo, hi = mins[idx], maxs[idx]
        if hi - lo < 1e-6:
            return None, None, tag
        values = np.linspace(lo, hi, n)
        return [{tag: float(v)} for v in values], values.astype(np.float64), tag
    if valid:
        ts = np.linspace(0.0, 1.0, n)
        overrides = []
        for t in ts:
            ov: dict[str, float] = {}
            for tag in valid:
                idx = tags.index(tag)
                ov[tag] = float(mins[idx] + t * (maxs[idx] - mins[idx]))
            overrides.append(ov)
        return overrides, ts.astype(np.float64), "+".join(valid)
    return None, None, "+".join(group)


def _coords_for(
    defaults: list[float], tags: list[str], overrides: dict[str, float]
) -> list[float]:
    coords = list(defaults)
    for tag, val in overrides.items():
        coords[tags.index(tag)] = float(val)
    return coords


def encode_decode_sweep(
    font: StandaloneFont,
    model: StyleExtractionModelV2,
    device: torch.device,
    char_set: list[int],
    cp_to_idx: dict[int, int],
    evidence_cps: list[int],
    target_cps: list[int],
    tags: list[str],
    defaults: list[float],
    overrides: list[dict[str, float]],
    with_intermediates: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
]:
    """Encode + decode across the sweep.

    Returns ``(X_summary, X_tokens, X_stylemap_mean, X_stylemap, X_coarse,
    coarse_recs, gts, recs)`` where ``X_summary`` is ``(N, D)`` mean-pooled
    style, ``X_tokens`` is ``(N, K*D)`` flattened tokens, and ``coarse_recs`` /
    ``gts`` / ``recs`` are ``(N, T, H, W)``.

    ``X_stylemap_mean`` / ``X_stylemap`` / ``X_coarse`` are the intermediate
    (pre-SPADE) style-map mean / flattened style-map / flattened coarse image;
    ``coarse_recs`` is the stage-1 coarse output image.  The intermediate
    returns are ``None`` for v2/v3, which expose no intermediate decode stages.
    """
    size = model.config.image_size
    ev_idx = torch.tensor([cp_to_idx[cp] for cp in evidence_cps], device=device)
    tgt_idx = torch.tensor([cp_to_idx[cp] for cp in target_cps], device=device)

    summaries, lat_flats, gt_list, rec_list = [], [], [], []
    sm_mean_list, sm_flat_list, coarse_flat_list, coarse_list = [], [], [], []
    with torch.no_grad():
        for ov in overrides:
            coords = _coords_for(defaults, tags, ov)

            ev = torch.stack(
                [
                    render_glyph(font, cp, size, axis_position=coords)
                    for cp in evidence_cps
                ]
            ).unsqueeze(
                1
            )  # (G, 1, H, W)

            gts = torch.stack(
                [
                    render_glyph(font, cp, size, axis_position=coords)
                    for cp in target_cps
                ]
            )  # (T, H, W)

            tokens = model.encode_style(
                ev.unsqueeze(0).to(device),
                style_codepoint_idx=ev_idx.unsqueeze(0),
            )  # (1, K, D)

            recs = []
            if with_intermediates:
                sm_means, sm_flats, coarse_flats, coarse_imgs = [], [], [], []
                for ti in tgt_idx:
                    coarse, style_map, fine, _attn = model.decode_with_intermediates(
                        ti.unsqueeze(0), tokens
                    )
                    recs.append(fine[0, 0].cpu())
                    coarse_imgs.append(coarse[0, 0].cpu())  # (H, W)
                    sm = style_map[0]  # (D, grid, grid)
                    sm_means.append(sm.mean(dim=(-1, -2)))  # (D,)
                    sm_flats.append(sm.reshape(-1))  # (D*grid*grid,)
                    coarse_flats.append(coarse[0, 0].reshape(-1))  # (H*W,)
                recs = torch.stack(recs)  # (T, H, W)
                coarse_list.append(torch.stack(coarse_imgs))  # (T, H, W)
                sm_mean_list.append(torch.stack(sm_means).mean(0).cpu())  # (D,)
                sm_flat_list.append(
                    torch.stack(sm_flats).mean(0).cpu()
                )  # (D*grid*grid,)
                coarse_flat_list.append(
                    torch.stack(coarse_flats).mean(0).cpu()
                )  # (H*W,)
            else:
                for ti in tgt_idx:
                    rec = model.decode(ti.unsqueeze(0), tokens)  # (1, 1, H, W)
                    recs.append(rec[0, 0].cpu())
                recs = torch.stack(recs)  # (T, H, W)

            summaries.append(tokens.mean(dim=1).squeeze(0).cpu())
            lat_flats.append(tokens.reshape(1, -1).squeeze(0).cpu())
            gt_list.append(gts)
            rec_list.append(recs)

    X_summary = torch.stack(summaries).numpy().astype(np.float64)
    X_tokens = torch.stack(lat_flats).numpy().astype(np.float64)
    gts = torch.stack(gt_list)  # (N, T, H, W)
    recs = torch.stack(rec_list)  # (N, T, H, W)

    X_stylemap_mean = X_stylemap = X_coarse = None
    coarse_recs = None
    if with_intermediates:
        X_stylemap_mean = torch.stack(sm_mean_list).numpy().astype(np.float64)
        X_stylemap = torch.stack(sm_flat_list).numpy().astype(np.float64)
        X_coarse = torch.stack(coarse_flat_list).numpy().astype(np.float64)
        coarse_recs = torch.stack(coarse_list)  # (N, T, H, W)
    return (
        X_summary,
        X_tokens,
        X_stylemap_mean,
        X_stylemap,
        X_coarse,
        coarse_recs,
        gts,
        recs,
    )


def ridge_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int,
    n_pca: int = 0,
) -> tuple[float, float, float, float]:
    """K-fold frozen ridge regression → (r2_mean, r2_std, r_mean, r_std).

    ``n_pca > 0`` reduces the features with PCA (fitted per training fold) before
    the ridge — needed for the flattened token set, which has far more features
    than samples.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    alphas = np.logspace(-2, 4, 25)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    r2s, rs = [], []
    for tr, te in kf.split(X):
        Xtr, Xte = X[tr], X[te]
        if n_pca and n_pca < Xtr.shape[1]:
            pca = PCA(n_components=min(n_pca, Xtr.shape[0] - 1), random_state=seed)
            Xtr = pca.fit_transform(Xtr)
            Xte = pca.transform(Xte)
        scaler = StandardScaler().fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xte = scaler.transform(Xte)
        reg = RidgeCV(alphas=alphas).fit(Xtr, y[tr])
        pred = reg.predict(Xte)
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


def reconstruction_tracking(
    gts: torch.Tensor, recs: torch.Tensor
) -> tuple[float, float]:
    """Whether decoded glyphs track the axis.

    Builds ``D[i, j] = L1(GT_i, rec_j)`` over sweep points per glyph, then:

    * ``top1`` — fraction of points where ``rec_i`` is the nearest reconstruction
      to ``GT_i`` (chance = ``1/N``).
    * ``diag_off`` — mean diagonal L1 / mean off-diagonal L1.  ``< 1`` = tracking.
    """
    n = gts.shape[0]
    t = gts.shape[1]
    if n < 2 or t == 0:
        return float("nan"), float("nan")

    top1s, diag_offs = [], []
    for gi in range(t):
        gt_g = gts[:, gi]  # (N, H, W)
        rec_g = recs[:, gi]  # (N, H, W)
        D = (gt_g[:, None] - rec_g[None, :]).abs().mean(dim=(-1, -2))  # (N, N)
        best = D.argmin(dim=-1)  # (N,)
        top1s.append(float((best == torch.arange(n, device=D.device)).float().mean()))
        diag = D.diagonal().mean()
        off = (D.sum() - D.diagonal().sum()) / (n * (n - 1))
        diag_offs.append(float(diag / (off + 1e-12)))
    return float(np.mean(top1s)), float(np.mean(diag_offs))


def visualize_axis(
    font: StandaloneFont,
    model: StyleExtractionModelV2,
    device: torch.device,
    cp_to_idx: dict[int, int],
    evidence_cps: list[int],
    target_cps: list[int],
    tags: list[str],
    defaults: list[float],
    overrides: list[dict[str, float]],
    col_labels: list[str],
    label: str,
    out_path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size = model.config.image_size
    ev_idx = torch.tensor([cp_to_idx[cp] for cp in evidence_cps], device=device)
    tgt_idx = torch.tensor([cp_to_idx[cp] for cp in target_cps], device=device)

    n_glyphs = len(target_cps)
    n_vals = len(overrides)
    fig, axes = plt.subplots(
        2 * n_glyphs, n_vals, figsize=(n_vals * 0.95, 2 * n_glyphs * 0.95)
    )
    if axes.ndim == 1:
        axes = axes[:, None]

    with torch.no_grad():
        for j, ov in enumerate(overrides):
            coords = _coords_for(defaults, tags, ov)
            ev = torch.stack(
                [
                    render_glyph(font, cp, size, axis_position=coords)
                    for cp in evidence_cps
                ]
            ).unsqueeze(1)
            tokens = model.encode_style(
                ev.unsqueeze(0).to(device), style_codepoint_idx=ev_idx.unsqueeze(0)
            )
            for i, ti in enumerate(tgt_idx):
                gt = render_glyph(font, target_cps[i], size, axis_position=coords)
                rec = model.decode(ti.unsqueeze(0), tokens)[0, 0].cpu()
                axes[2 * i, j].imshow(gt.numpy(), cmap="gray", vmin=0.0, vmax=1.0)
                axes[2 * i + 1, j].imshow(rec.numpy(), cmap="gray", vmin=0.0, vmax=1.0)

    for j, lbl in enumerate(col_labels):
        axes[0, j].set_title(lbl, fontsize=7)
    for i, ch in enumerate(target_cps):
        axes[2 * i, 0].set_ylabel(f"GT {chr(ch)}", fontsize=7)
        axes[2 * i + 1, 0].set_ylabel(f"rec {chr(ch)}", fontsize=7)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{Path(font.path).name} — {label}", fontsize=8)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  saved {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-path",
        required=True,
        help="Path to the frozen StyleExtractionModelV2 checkpoint (.conf.json sidecar).",
    )
    p.add_argument(
        "--fonts",
        required=True,
        action="append",
        help="A parametric font to probe: PATH or PATH:axis1,axis2,... "
        "(repeatable; omit :axes to sweep the default fine-detail axes). "
        "Join axes with '+' to sweep them together.",
    )
    p.add_argument("--samples", type=int, default=48, help="Points per axis sweep.")
    p.add_argument(
        "--cv-folds", type=int, default=5, help="KFold folds for ridge probe."
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default="outputs", help="Directory for montages.")
    p.add_argument(
        "--viz-glyphs",
        default="an",
        help="Glyphs to montage and track (must be in the model's character set).",
    )
    p.add_argument(
        "--viz-points",
        type=int,
        default=5,
        help="Axis positions to show in the montage (incl. endpoints).",
    )
    p.add_argument(
        "--token-pca",
        type=int,
        default=16,
        help="PCA components for the flattened token-set ridge.",
    )
    p.add_argument(
        "--no-viz", action="store_true", help="Skip reconstruction montages."
    )
    args = p.parse_args()

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("Loading model …")
    model, _config = load_model(args.model_path, device)
    char_set = sorted(set(model.config.character_set))
    cp_to_idx = {cp: i for i, cp in enumerate(char_set)}
    with_intermediates = hasattr(model, "decode_with_intermediates")

    print("\nAxis sensitivity (frozen ridge R² + reconstruction tracking):")
    header = (
        f"  {'font':<26} {'axis':<16} {'n':<4} "
        f"{'summ R2':>8} {'summ r':>7}  "
        f"{'tok R2':>8} {'tok r':>7}  "
        f"{'smapR2':>8} {'smapfR2':>8} {'coarseR2':>9} {'coarseD/O':>9}  "
        f"{'trk@1':>7} {'d/o':>7}"
    )
    print(header)

    for spec in args.fonts:
        font_path, groups = _parse_font_spec(spec)
        font_path = Path(font_path)
        if not font_path.exists():
            print(f"  [skip] {font_path} not found")
            continue

        tags, defaults, mins, maxs = read_fvar(font_path)
        if groups is None:
            groups = [[a] for a in DEFAULT_AXES if a in tags]
        for g in groups:
            for a in [t for t in g if t not in tags]:
                print(f"  [skip] {font_path.name}: axis '{a}' not in fvar")
        groups = [[t for t in g if t in tags] for g in groups]
        groups = [g for g in groups if g]

        font = StandaloneFont(font_path)
        stem = font_path.stem

        # Deterministic evidence set + target glyphs.
        avail = [cp for cp in char_set if cp in font.codepoints]
        viz_cps = [
            ord(ch)
            for ch in args.viz_glyphs
            if ord(ch) in cp_to_idx and ord(ch) in font.codepoints
        ]
        seen: set[int] = set()
        viz_cps = [c for c in viz_cps if not (c in seen or seen.add(c))]
        evidence_cps = [cp for cp in avail if cp not in set(viz_cps)][
            : model.config.num_evidence_glyphs
        ]
        if len(evidence_cps) < model.config.num_evidence_glyphs:
            evidence_cps = [cp for cp in avail if cp not in set(viz_cps)]
        if not viz_cps:
            print(
                f"  [skip] {font_path.name}: no viz glyphs from '{args.viz_glyphs}' in character set"
            )
            continue

        for group in groups:
            overrides, y, label = build_sweep(group, tags, mins, maxs, args.samples)
            if overrides is None:
                print(f"  [skip] {font_path.name}:{label} is degenerate")
                continue

            (
                X_summary,
                X_tokens,
                X_smap_mean,
                X_smap,
                X_coarse,
                coarse_recs,
                gts,
                recs,
            ) = encode_decode_sweep(
                font,
                model,
                device,
                char_set,
                cp_to_idx,
                evidence_cps,
                viz_cps,
                tags,
                defaults,
                overrides,
                with_intermediates=with_intermediates,
            )

            r2_sm, _, r_sm, _ = ridge_probe(X_summary, y, args.cv_folds, args.seed)
            r2_tk, _, r_tk, _ = ridge_probe(
                X_tokens, y, args.cv_folds, args.seed, n_pca=args.token_pca
            )
            top1, diag_off = reconstruction_tracking(gts, recs)

            parts = [f"  {stem:<26} {label:<16} {len(y):<4}"]
            parts.append(f"{r2_sm:>8.3f} {r_sm:>7.3f}")
            parts.append(f"{r2_tk:>8.3f} {r_tk:>7.3f}")
            if with_intermediates:
                r2_smap, _, _, _ = ridge_probe(X_smap_mean, y, args.cv_folds, args.seed)
                r2_smapf, _, _, _ = ridge_probe(
                    X_smap, y, args.cv_folds, args.seed, n_pca=args.token_pca
                )
                r2_coarse, _, _, _ = ridge_probe(
                    X_coarse, y, args.cv_folds, args.seed, n_pca=args.token_pca
                )
                _coarse_top1, coarse_dioff = reconstruction_tracking(gts, coarse_recs)
                parts.append(
                    f"{r2_smap:>8.3f} {r2_smapf:>8.3f} {r2_coarse:>9.3f} {coarse_dioff:>9.3f}"
                )
            else:
                parts.append(f"{'—':>8} {'—':>8} {'—':>9} {'—':>9}")
            parts.append(f"{top1:>7.3f} {diag_off:>7.3f}")
            print("  ".join(parts))

            if not args.no_viz:
                viz_ov, _, _ = build_sweep(
                    group, tags, mins, maxs, max(args.viz_points, 2)
                )
                if len(group) == 1:
                    col_labels = [
                        f"{group[0]}={list(ov.values())[0]:g}" for ov in viz_ov
                    ]
                else:
                    col_labels = [
                        f"t={i/(len(viz_ov)-1):.2f}" for i in range(len(viz_ov))
                    ]
                out_path = str(
                    Path(args.out_dir) / f"axis_{stem}_{label.replace('+', '_')}.png"
                )
                visualize_axis(
                    font,
                    model,
                    device,
                    cp_to_idx,
                    evidence_cps,
                    viz_cps,
                    tags,
                    defaults,
                    viz_ov,
                    col_labels,
                    label,
                    out_path,
                )


if __name__ == "__main__":
    main()
