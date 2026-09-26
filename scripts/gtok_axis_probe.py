#!/usr/bin/env python
"""Parametric-axis sensitivity probe for the frozen GTok tokenizer.

This is the token-level analogue of ``style_embedding_axis_probe.py``.  The
question it answers is narrower and more fundamental: **can GTok even
reconstruct the fine-detail axis at all?**

A parametric variable font gives us ground-truth, continuous labels — its
design-space axes.  By rendering the same font at swept axis positions and
pushing each through the frozen GTok tokenizer, we can ask whether the
*quantized codes* and the *decoded reconstruction* track a controlled change in
one fine axis (contrast, curve squareness, terminal rounding) with everything
else held fixed.

For each requested axis we:

* sweep ``--samples`` points across the axis range (other axes at default),
* encode each instance's glyphs through GTok and report a frozen ridge probe of
  the mean-pooled quantized code vector against the axis value, and
* measure *reconstruction tracking*: for each glyph, whether the decoded image
  at axis point ``t`` is closer to its own ground truth than to the ground truth
  at other points (``trk@1`` = nearest-neighbour hit rate, ``d/o`` = diagonal /
  off-diagonal L1 ratio, where ``< 1`` means tracking).

If the code R² is high but ``trk@1`` ≈ 1/N and ``d/o`` ≈ 1, then the scalar is
in the codes but the decoder smears it away — the same "retained but not
decoded" split we saw in the style embedder.  If *both* are flat, the fine
detail never survived tokenization.

Example::

    PYTHONPATH=Lib python scripts/gtok_axis_probe.py \
        --model-path models/gtok_model.pth \
        --fonts "RobotoDelta-Roman-VF.ttf:YOPQ,XOPQ,STLI+STLO+STUI+STUO" \
        --fonts "YouTubeMarquee[BASE,GRAD,ROND,SCAL,TANG,TRML,WDSP,XINK,XOFI,XOJN,XOLC,XOPQ,XORN,XOUC,XTFI,XTLC,XTLR,XTRA,XTSP,XTUC,YOFI,YOLC,YOPQ,YOUC,YTDE,YTLC,opsz,wdth,wght].ttf:ROND"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hrothgar.glyph_rendering import crop_to_ink
from hrothgar.googlefonts import StandaloneFont
from hrothgar.gtok.model import load_model
from hrothgar.utils import pick_device

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
# Axis helpers (mirror style_embedding_axis_probe.py)
# ---------------------------------------------------------------------------


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
    """Return ``(path, groups)``; each group is a list of tags swept together."""
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
    """Build override dicts + regression target for a single or grouped sweep."""
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


# ---------------------------------------------------------------------------
# Rendering + tokenizing
# ---------------------------------------------------------------------------


def _render_viz_glyphs(
    font: StandaloneFont,
    cp_list: list[int],
    size: int,
    coords: list[float],
) -> torch.Tensor:
    """Render viz glyphs at ``coords`` → ``(G, 3, H, W)`` float32 [0, 1]."""
    ims = []
    for cp in cp_list:
        arr = font.render(cp, size=size, axis_position=coords)  # (3, H, W)
        ims.append(crop_to_ink(torch.from_numpy(arr.copy()), size))
    return torch.stack(ims)  # (G, 3, H, W)


def encode_sweep(
    font: StandaloneFont,
    model,
    device: torch.device,
    overrides: list[dict[str, float]],
    tags: list[str],
    defaults: list[float],
    viz_chars: str,
    image_size: int,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Encode/decode each override → ``(code_vecs, recon, gt)``.

    Returns:
        code_vecs: ``(N, code_dim)`` mean-pooled quantized codes per instance.
        recon: ``(N, G, H, W)`` reconstructed glyphs (channel 0).
        gt: ``(N, G, H, W)`` ground-truth glyphs (channel 0).
    """
    cp_list = [ord(ch) for ch in viz_chars]

    all_images: list[torch.Tensor] = []
    gt_list: list[torch.Tensor] = []
    for ov in overrides:
        coords = list(defaults)
        for tag, val in ov.items():
            coords[tags.index(tag)] = float(val)
        im = _render_viz_glyphs(font, cp_list, image_size, coords)  # (G,3,H,W)
        gt_list.append(im[:, 0])  # (G,H,W)
        all_images.append(im)

    images = torch.stack(all_images).to(device, dtype=torch.float32)  # (N,G,3,H,W)
    N, G, C, H, W = images.shape
    flat = images.reshape(N * G, C, H, W)

    with torch.no_grad():
        quantized, _ = model.encode(flat)  # (N*G, seq, code_dim)
        recon = model.decode(quantized)  # (N*G, 3, H, W)

    code_dim = quantized.shape[-1]
    # Mean-pool over the token sequence → per-glyph code vector, then pool over
    # glyphs → one font-level code vector per instance.  This is the GTok
    # analogue of the style embedder's "summary"; the full (seq * code_dim)
    # flattened codes are far too high-dimensional for a 48-point ridge probe.
    code_vecs = (
        quantized.reshape(N, G, -1, code_dim)
        .mean(dim=2)  # (N, G, code_dim)
        .mean(dim=1)  # (N, code_dim)
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    recon = recon.reshape(N, G, C, H, W)[:, :, 0].cpu()  # (N,G,H,W)
    gt = torch.stack(gt_list)  # (N,G,H,W)
    return code_vecs, recon, gt


def _normalize_glyph(glyph: torch.Tensor, size: int) -> torch.Tensor:
    """Crop to ink bbox and stretch to ``(size, size)`` (isolates shape)."""
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


def recon_shape_l1(gt: torch.Tensor, recon: torch.Tensor, image_size: int) -> float:
    """Mean L1 between bbox-normalized GT and reconstruction (shape fidelity)."""
    n, g = gt.shape[:2]
    errs = []
    for i in range(n):
        for gi in range(g):
            a = _normalize_glyph(gt[i, gi].clone(), image_size)
            b = _normalize_glyph(recon[i, gi].clone(), image_size)
            errs.append((a - b).abs().mean().item())
    return float(np.mean(errs))


def reconstruction_tracking(
    gt: torch.Tensor, recon: torch.Tensor, image_size: int
) -> tuple[float, float]:
    """Return ``(trk@1, d/o)`` over viz glyphs (bbox-normalized shapes)."""
    n, g = gt.shape[:2]
    top1s: list[float] = []
    diag_offs: list[float] = []
    for gi in range(g):
        gt_g = torch.stack(
            [_normalize_glyph(gt[i, gi].clone(), image_size) for i in range(n)]
        )
        rec_g = torch.stack(
            [_normalize_glyph(recon[i, gi].clone(), image_size) for i in range(n)]
        )
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
    model,
    device: torch.device,
    overrides: list[dict[str, float]],
    col_labels: list[str],
    tags: list[str],
    defaults: list[float],
    viz_chars: str,
    label: str,
    image_size: int,
    out_path: str,
) -> None:
    """Montage GT vs GTok reconstruction (bbox-normalized) across the sweep."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cp_list = [ord(ch) for ch in viz_chars]
    n_glyphs = len(cp_list)
    n_vals = len(overrides)
    fig, axes = plt.subplots(
        2 * n_glyphs, n_vals, figsize=(n_vals * 0.95, 2 * n_glyphs * 0.95)
    )
    if axes.ndim == 1:
        axes = axes[:, None]

    for j, ov in enumerate(overrides):
        coords = list(defaults)
        for tag, val in ov.items():
            coords[tags.index(tag)] = float(val)
        im = _render_viz_glyphs(font, cp_list, image_size, coords)  # (G,3,H,W)
        batch = im.to(device, dtype=torch.float32).unsqueeze(0)  # (1,G,3,H,W)
        with torch.no_grad():
            quantized, _ = model.encode(batch[0])  # (G,seq,code_dim)
            rec = model.decode(quantized)  # (G,3,H,W)
        for i in range(n_glyphs):
            gt_norm = _normalize_glyph(im[i, 0].clone(), image_size).numpy()
            rec_norm = _normalize_glyph(rec[i, 0].clone().cpu(), image_size).numpy()
            axes[2 * i, j].imshow(gt_norm, cmap="gray", vmin=0.0, vmax=1.0)
            axes[2 * i + 1, j].imshow(rec_norm, cmap="gray", vmin=0.0, vmax=1.0)

    for j, lbl in enumerate(col_labels):
        axes[0, j].set_title(lbl, fontsize=7)
    for i, ch in enumerate(viz_chars):
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-path",
        required=True,
        help="Path to the frozen GTok checkpoint (needs a .conf.json sidecar).",
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
        help="Torch device (default: mps/cuda if available, else cpu).",
    )
    p.add_argument("--out-dir", default="outputs", help="Directory for montages.")
    p.add_argument(
        "--viz-glyphs",
        default="aegs",
        help="Glyphs to tokenize, track, and montage (default: 'aegs').",
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

    device = torch.device(args.device) if args.device else pick_device()

    print("Loading GTok …")
    model, config = load_model(Path(args.model_path), device)
    image_size = config.image_size
    print(f"Loaded GTok model (image_size={image_size})")

    print("\nAxis sensitivity (frozen code ridge R² + reconstruction tracking):")
    print(
        f"  {'font':<28} {'axis':<16} {'n':<4} "
        f"{'code R2':>10} {'code r':>7}  {'shape L1':>9} {'trk@1':>7} {'d/o':>7}"
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
            assert overrides is not None and y is not None

            X_code, recon, gt = encode_sweep(
                font,
                model,
                device,
                overrides,
                tags,
                defaults,
                args.viz_glyphs,
                image_size,
            )

            r2_m, r2_s, r_m, _ = ridge_probe(X_code, y, args.cv_folds, args.seed)
            shape_l1 = recon_shape_l1(gt, recon, image_size)
            trk1, diag_off = reconstruction_tracking(gt, recon, image_size)

            print(
                f"  {stem:<28} {label:<16} {len(y):<4} "
                f"{r2_m:>9.3f}±{r2_s:<4.3f} {r_m:>6.3f}  "
                f"{shape_l1:>8.4f} {trk1:>6.3f} {diag_off:>6.3f}"
            )

            if not args.no_viz:
                viz_ov, _, _ = build_sweep(
                    group, tags, mins, maxs, max(args.viz_points, 2)
                )
                assert viz_ov is not None
                if len(group) == 1:
                    col_labels = [
                        f"{group[0]}={list(ov.values())[0]:g}" for ov in viz_ov
                    ]
                else:
                    col_labels = [
                        f"t={i/(len(viz_ov)-1):.2f}" for i in range(len(viz_ov))
                    ]
                out_path = str(
                    Path(args.out_dir)
                    / f"gtok_axis_{stem}_{label.replace('+', '_')}.png"
                )
                visualize_axis(
                    font,
                    model,
                    device,
                    viz_ov,
                    col_labels,
                    tags,
                    defaults,
                    args.viz_glyphs,
                    label,
                    image_size,
                    out_path,
                )


if __name__ == "__main__":
    main()
