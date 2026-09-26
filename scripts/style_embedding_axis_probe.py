#!/usr/bin/env python
"""Parametric-axis sensitivity probe for the frozen style embedding.

A parametric variable font exposes *ground-truth, continuous* fine-detail labels:
its design-space axes. By rendering the same font at swept axis positions and
encoding each with the frozen ``FontStyleEmbedder``, we can ask a much cleaner
question than the cross-font probes: does the embedding *track* a controlled
change in one fine axis (contrast, slant, curve squareness, terminal rounding)
with everything else held fixed?

For each requested axis we:

* sweep ``--samples`` points across the axis range (all other axes at default),
* encode the full glyph set at each point into the summary + style latents,
* run a frozen ridge regression to recover the axis value from each, and
* report R² (summary vs latents).

``slnt`` is a global shear (the summary should capture it); ``YOPQ``/``XOPQ``/
``ST**``/``ROND`` are *local* fine detail, where the token set should beat the
mean summary if it is doing its job.

Optionally we also montage the shape head's bbox-normalized reconstruction vs the
ground-truth glyph across the sweep, so you can *see* whether the reconstruction
tracks the axis (e.g. straight → slightly rounded → rounded terminals).

Example::

    PYTHONPATH=Lib python scripts/style_embedding_axis_probe.py \
        --embedder-path /path/to/style_embedding.pth \
        --fonts "RobotoDelta-Roman-VF.ttf:slnt,YOPQ,XOPQ,STLI+STLO+STUI+STUO" \
        --fonts "YouTubeMarquee[BASE,GRAD,ROND,SCAL,TANG,TRML,WDSP,XINK,XOFI,XOJN,XOLC,XOPQ,XORN,XOUC,XTFI,XTLC,XTLR,XTRA,XTSP,XTUC,YOFI,YOLC,YOPQ,YOUC,YTDE,YTLC,opsz,wdth,wght].ttf:ROND"

Axes joined with ``+`` are swept *together* (a shared normalised coordinate
mapped into each axis's own range), e.g. ``STLI+STLO+STUI+STUO`` for a coherent
"round → square" sweep. Comma-separated axes are swept individually.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hrothgar.googlefonts import StandaloneFont
from hrothgar.style_embedding import FontStyleEmbedder, FontStyleEmbedderConfig

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
]


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


# ---------------------------------------------------------------------------
# Rendering + encoding
# ---------------------------------------------------------------------------


def render_instance(
    font: StandaloneFont,
    embedder: FontStyleEmbedder,
    tags: list[str],
    defaults: list[float],
    overrides: dict[str, float],
) -> torch.Tensor:
    """Render the full input glyph set at ``overrides`` → ``(G, 1, H, W)``."""
    size = embedder.config.glyph_size
    coords = list(defaults)
    for tag, val in overrides.items():
        coords[tags.index(tag)] = float(val)

    glyphs = []
    for cp in embedder.config.input_codepoints:
        arr = font.render(cp, size=size, axis_position=coords)
        gray = arr[0].copy() if arr.ndim == 3 else np.asarray(arr, dtype=np.float32)
        glyphs.append(torch.from_numpy(gray))
    return torch.stack(glyphs).unsqueeze(1)  # (G, 1, H, W)


def encode_sweep(
    font: StandaloneFont,
    embedder: FontStyleEmbedder,
    device: torch.device,
    overrides: list[dict[str, float]],
    tags: list[str],
    defaults: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Encode the glyph set at each override dict → (summary, flattened latents)."""
    images = [render_instance(font, embedder, tags, defaults, ov) for ov in overrides]
    batch = torch.stack(images).to(device, dtype=torch.float32)  # (N, G, 1, H, W)
    with torch.no_grad():
        summary, latents, _ = embedder.encode_with_tokens(batch)
    return (
        summary.cpu().numpy().astype(np.float64),
        latents.cpu().numpy().reshape(latents.shape[0], -1).astype(np.float64),
    )


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


def ridge_probe(
    X: np.ndarray, y: np.ndarray, n_splits: int, seed: int
) -> tuple[float, float, float, float]:
    """K-fold frozen ridge regression → (r2_mean, r2_std, r_mean, r_std)."""
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


def _normalize_glyph(glyph: torch.Tensor, size: int) -> torch.Tensor:
    """Crop to ink bbox and stretch to ``(size, size)`` (matches the shape target)."""
    ink = glyph < 0.5
    if not ink.any():
        return torch.ones(size, size)
    ys, xs = ink.nonzero(as_tuple=True)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    crop = glyph[y0 : y1 + 1, x0 : x1 + 1]
    if crop.numel() == 0:
        return torch.ones(size, size)
    crop = crop[None, None]  # (1, 1, h, w)
    out = F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False)
    return out[0, 0]


def reconstruction_tracking(
    font: StandaloneFont,
    embedder: FontStyleEmbedder,
    device: torch.device,
    overrides: list[dict[str, float]],
    tags: list[str],
    defaults: list[float],
    viz_chars: str,
) -> tuple[float, float] | None:
    """Quantify whether *reconstructions* track the axis (not just the embedding).

    For each viz glyph, build the L1 distance matrix ``D[i,j] = L1(GT_i, rec_j)``
    over the sweep points, then report:

    * ``top1`` — fraction of points where ``rec_i`` is the nearest reconstruction
      to ``GT_i`` (chance baseline is ``1/N``).
    * ``diag_off`` — mean diagonal L1 / mean off-diagonal L1. ``< 1`` means the
      reconstruction sits closer to its own GT than to others (tracking); ``≈ 1``
      means no response.
    """
    size = embedder.config.glyph_size
    cp_idxs = [
        embedder.config.input_codepoints.index(ord(ch))
        for ch in viz_chars
        if ord(ch) in embedder.config.input_codepoints
    ]
    n = len(overrides)
    if not cp_idxs or n < 2:
        return None

    gts: list[torch.Tensor] = []
    recs: list[torch.Tensor] = []
    for ov in overrides:
        im = render_instance(font, embedder, tags, defaults, ov)  # (G,1,H,W)
        batch = im.unsqueeze(0).to(device, dtype=torch.float32)
        gt = torch.stack([_normalize_glyph(im[cp, 0].clone(), size) for cp in cp_idxs])
        cp_t = torch.tensor(cp_idxs, device=device)
        with torch.no_grad():
            _, latents, spatial_style = embedder.encode_with_tokens(batch)
            lat_ng = latents.expand(len(cp_idxs), -1, -1)
            spa_ng = spatial_style.expand(len(cp_idxs), -1, -1, -1)
            rec = embedder.reconstruct_shape(lat_ng, cp_t, spatial_style=spa_ng)[
                :, 0
            ].cpu()
        gts.append(gt)
        recs.append(rec)

    gts = torch.stack(gts)  # (N, ng, H, W)
    recs = torch.stack(recs)  # (N, ng, H, W)
    top1s: list[float] = []
    diag_offs: list[float] = []
    for gi in range(len(cp_idxs)):
        gt_g = gts[:, gi]  # (N, H, W)
        rec_g = recs[:, gi]  # (N, H, W)
        D = (gt_g[:, None] - rec_g[None, :]).abs().mean(dim=(-1, -2))  # (N, N)
        ranks = (D.argsort(dim=-1) == torch.arange(n)[:, None]).nonzero(as_tuple=True)[
            1
        ]
        top1s.append(float((ranks == 0).float().mean()))
        diag = D.diagonal().mean()
        off = (D.sum() - D.diagonal().sum()) / (n * (n - 1))
        diag_offs.append(float(diag / (off + 1e-12)))
    return float(np.mean(top1s)), float(np.mean(diag_offs))


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def visualize_axis(
    font: StandaloneFont,
    embedder: FontStyleEmbedder,
    device: torch.device,
    overrides: list[dict[str, float]],
    col_labels: list[str],
    tags: list[str],
    defaults: list[float],
    viz_chars: str,
    label: str,
    out_path: str,
) -> None:
    """Montage GT (bbox-normalised) vs shape-head reconstruction across the sweep."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size = embedder.config.glyph_size
    cp_idxs = [
        embedder.config.input_codepoints.index(ord(ch))
        for ch in viz_chars
        if ord(ch) in embedder.config.input_codepoints
    ]
    if not cp_idxs:
        return

    n_glyphs = len(cp_idxs)
    n_vals = len(overrides)
    fig, axes = plt.subplots(
        2 * n_glyphs, n_vals, figsize=(n_vals * 0.95, 2 * n_glyphs * 0.95)
    )
    if axes.ndim == 1:
        axes = axes[:, None]

    for j, ov in enumerate(overrides):
        im = render_instance(font, embedder, tags, defaults, ov)
        batch = im.unsqueeze(0).to(device, dtype=torch.float32)
        with torch.no_grad():
            _summary, latents, spatial_style = embedder.encode_with_tokens(batch)
        for i, cp_idx in enumerate(cp_idxs):
            gt_norm = _normalize_glyph(im[cp_idx, 0].clone(), size).numpy()
            cp_t = torch.tensor([cp_idx], device=device)
            rec = (
                embedder.reconstruct_shape(latents, cp_t, spatial_style=spatial_style)[
                    0, 0
                ]
                .cpu()
                .numpy()
            )
            axes[2 * i, j].imshow(gt_norm, cmap="gray", vmin=0.0, vmax=1.0)
            axes[2 * i + 1, j].imshow(rec, cmap="gray", vmin=0.0, vmax=1.0)

    for j, lbl in enumerate(col_labels):
        axes[0, j].set_title(lbl, fontsize=7)
    shown = [ch for ch in viz_chars if ord(ch) in embedder.config.input_codepoints]
    for i, ch in enumerate(shown):
        axes[2 * i, 0].set_ylabel(f"GT {ch}", fontsize=7)
        axes[2 * i + 1, 0].set_ylabel(f"rec {ch}", fontsize=7)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{Path(font.path).name} — {label}", fontsize=8)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_font_spec(spec: str) -> tuple[str, list[list[str]] | None]:
    """Return ``(path, groups)``; each group is a list of tags swept together.

    ``+`` joins tags into a group (e.g. ``STLI+STLO+STUI+STUO``); ``,`` separates
    groups (e.g. ``slnt,YOPQ,STLI+STLO+STUI+STUO``).
    """
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
    """Build override dicts + regression target for a single or grouped sweep.

    A single-axis group sweeps that axis over its own [min, max]. A multi-axis
    group sweeps a shared normalised coordinate t ∈ [0, 1] mapped into each axis's
    own range, so "round → square" moves all axes coherently together.
    """
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--embedder-path",
        required=True,
        help="Path to the frozen FontStyleEmbedder checkpoint (needs a .conf.json sidecar).",
    )
    p.add_argument(
        "--fonts",
        required=True,
        action="append",
        help="A parametric font to probe: PATH or PATH:axis1,axis2,... "
        "(repeatable; omit :axes to sweep the default fine-detail axes). "
        "Join axes with '+' to sweep them together, e.g. STLI+STLO+STUI+STUO.",
    )
    p.add_argument("--samples", type=int, default=48, help="Points per axis sweep.")
    p.add_argument(
        "--cv-folds", type=int, default=5, help="KFold folds for ridge probe."
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device",
        default=None,
        help="Torch device (default: cuda if available, else cpu).",
    )
    p.add_argument("--out-dir", default="outputs", help="Directory for montages.")
    p.add_argument(
        "--viz-glyphs",
        default="an",
        help="Glyphs to montage (default: 'an').",
    )
    p.add_argument(
        "--viz-points",
        type=int,
        default=5,
        help="Axis positions to show in the montage (evenly spaced, incl. endpoints).",
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

    print("Loading embedder …")
    embedder = load_embedder(args.embedder_path, device)

    print("\nAxis sensitivity (frozen ridge R² + reconstruction tracking):")
    print(
        f"  {'font':<28} {'axis':<16} {'n':<4} {'summ R2':>10} {'summ r':>7}  {'lat R2':>10} {'lat r':>7}  {'trk@1':>7} {'d/o':>7}"
    )

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

        for group in groups:
            overrides, y, label = build_sweep(group, tags, mins, maxs, args.samples)
            if overrides is None:
                print(f"  [skip] {font_path.name}:{label} is degenerate")
                continue

            X_summary, X_latents = encode_sweep(
                font, embedder, device, overrides, tags, defaults
            )

            parts = [f"  {stem:<28} {label:<16} {len(y):<4}"]
            for X in (X_summary, X_latents):
                r2_m, r2_s, r_m, _ = ridge_probe(X, y, args.cv_folds, args.seed)
                parts.append(f"{r2_m:>9.3f}±{r2_s:<4.3f} {r_m:>6.3f}")
            top1_s, diag_s = "—", "—"
            if embedder.config.use_shape:
                trk = reconstruction_tracking(
                    font, embedder, device, overrides, tags, defaults, args.viz_glyphs
                )
                if trk is not None:
                    top1, diag_off = trk
                    top1_s, diag_s = f"{top1:.3f}", f"{diag_off:.3f}"
            parts.append(f"{top1_s:>7} {diag_s:>7}")
            print("  ".join(parts))

            if not args.no_viz and embedder.config.use_shape:
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
                    embedder,
                    device,
                    viz_ov,
                    col_labels,
                    tags,
                    defaults,
                    args.viz_glyphs,
                    label,
                    out_path,
                )


if __name__ == "__main__":
    main()
