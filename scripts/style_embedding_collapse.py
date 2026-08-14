#!/usr/bin/env python
"""Experiment A: does the frozen style embedding collapse fine style differences?

This measures, for a set of fonts, whether the embedding distance between two
fonts tracks the *perceptual* distance between a fixed letter rendered in those
fonts.  If fine style detail (terminal shape, stroke modulation, contrast) is
preserved by the embedding, then visually-different fonts should be far apart in
embedding space.  If the embedding is coarse, we expect the "collapse"
signature: many pairs that look very different yet sit close together in
embedding space.

The probe is fully self-supervised (no style labels).  It uses the same
glyph-set ``FontStyleEmbedder.encode`` path the generator uses to compute
embeddings, and the same LPIPS module the generator training uses for the
perceptual image distance.

Example::

    PYTHONPATH=Lib python scripts/style_embedding_collapse.py \
        --repo ~/google/fonts_checkout \
        --embedder-path models/style_embedding.pth \
        --char a --max-fonts 200 --plot outputs/collapse_a.png
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from hrothgar.googlefonts import GoogleFonts
from hrothgar.style_embedding import FontStyleEmbedder, FontStyleEmbedderConfig


# ---------------------------------------------------------------------------
# Small statistics helpers (no scipy dependency).
# ---------------------------------------------------------------------------

def _rankdata(x: np.ndarray) -> np.ndarray:
    """Return 1-based ranks.  Ties are negligible for continuous distances."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    return ranks


def pearson(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (r, r**2) for two equal-length vectors."""
    r = float(np.corrcoef(a, b)[0, 1])
    return r, r * r


def spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (rho, rho**2) via Pearson on ranks."""
    return pearson(_rankdata(a), _rankdata(b))


def _upper_triangle(m: np.ndarray) -> np.ndarray:
    return m[np.triu_indices(m.shape[0], k=1)]


# ---------------------------------------------------------------------------
# Image distances
# ---------------------------------------------------------------------------

def pairwise_l1(images: torch.Tensor) -> np.ndarray:
    """Pairwise mean-absolute distance on flattened grayscale images."""
    flat = images[:, 0].reshape(images.shape[0], -1).double()
    return torch.cdist(flat, flat, p=1).cpu().numpy()


def pairwise_ssim(images: torch.Tensor, window: int = 11) -> np.ndarray:
    """Pairwise 1 - SSIM on the grayscale channel.

    SSIM is computed with a uniform window; images are assumed to be
    single-channel-compatible grayscale glyphs in [0, 1].
    """
    gray = images[:, 0]  # (N, H, W)
    n = gray.shape[0]
    device = gray.device
    kernel = torch.ones(
        (1, 1, window, window), device=device, dtype=torch.float64
    ) / (window * window)
    pad = window // 2
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    # Per-image local statistics (vectorised).
    x = gray.double()  # (N, H, W)
    mu = F.conv2d(x[:, None], kernel, padding=pad)  # (N, 1, H, W)
    mu_sq = mu * mu
    sigma = F.conv2d((x * x)[:, None], kernel, padding=pad) - mu_sq

    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            mu_xy = mu[i] * mu[j]
            sigma_xy = (
                F.conv2d((x[i] * x[j])[None, None], kernel, padding=pad) - mu_xy
            )
            num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
            den = (mu[i] ** 2 + mu[j] ** 2 + c1) * (sigma[i] + sigma[j] + c2)
            ssim = (num / den).mean()
            out[i, j] = out[j, i] = float(1.0 - ssim.item())
    return out


def pairwise_lpips(images: torch.Tensor, lpips_model) -> np.ndarray:
    """Pairwise LPIPS distance.  ``images`` should be in [0, 1]."""
    n = images.shape[0]
    images = torch.clamp(images, 0.0, 1.0)
    d = torch.zeros((n, n), dtype=torch.float32, device=images.device)
    import tqdm

    with torch.no_grad():
        for i in tqdm.tqdm(range(n)):
            anchor = images[i : i + 1].expand(n, -1, -1, -1)
            d[i] = lpips_model(anchor, images).squeeze()
    return d.cpu().numpy()


# ---------------------------------------------------------------------------
# Collapse metric
# ---------------------------------------------------------------------------

def collapse_metrics(d_emb: np.ndarray, d_img: np.ndarray, frac: float = 0.1) -> dict:
    """Quantify how much visual difference survives in the nearest-embedding pairs."""
    de = _upper_triangle(d_emb)
    di = _upper_triangle(d_img)

    thresh_bottom = np.quantile(de, frac)
    bottom = di[de <= thresh_bottom]

    thresh_top = np.quantile(de, 1.0 - frac)
    top = di[de >= thresh_top]

    overall_p90 = float(np.percentile(di, 90))
    bottom_p90 = float(np.percentile(bottom, 90)) if bottom.size else float("nan")

    return {
        "frac": frac,
        "n_pairs": int(de.size),
        "n_bottom_pairs": int((de <= thresh_bottom).sum()),
        "d_emb_bottom_threshold": float(thresh_bottom),
        "d_img_bottom_max": float(bottom.max()) if bottom.size else float("nan"),
        "d_img_bottom_p90": bottom_p90,
        "d_img_top_p10": float(np.percentile(top, 10)) if top.size else float("nan"),
        "d_img_overall_p90": overall_p90,
        "d_img_overall_max": float(di.max()),
        # 1.0 => nearest-embedding pairs are as different as the typical pair
        # (severe collapse).  0.0 => nearest-embedding pairs are near-identical.
        "collapse_ratio": bottom_p90 / overall_p90 if overall_p90 > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Embedding + rendering
# ---------------------------------------------------------------------------

def load_embedder(path: str, device: torch.device) -> FontStyleEmbedder:
    config = FontStyleEmbedderConfig.from_sidecar(path)
    embedder = FontStyleEmbedder(config).to(device)
    embedder.load(path, device=device)
    embedder.eval()
    for p in embedder.parameters():
        p.requires_grad = False
    return embedder


def embed_font(font, embedder: FontStyleEmbedder, device: torch.device) -> torch.Tensor:
    codepoints = embedder.config.input_codepoints
    size = embedder.config.glyph_size

    glyphs = []
    for cp in codepoints:
        arr = font.render(cp, size=size)  # (3, size, size) float32 in [0, 1]
        gray = arr[0].copy() if arr.ndim == 3 else np.asarray(arr, dtype=np.float32)
        glyphs.append(torch.from_numpy(gray))

    # (1, G, 1, H, W) — the same shape FontStyleEmbedder.encode expects.
    images = torch.stack(glyphs).unsqueeze(1).unsqueeze(0)
    images = images.to(device, dtype=torch.float32)
    with torch.no_grad():
        return embedder.encode(images).squeeze(0).cpu()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, help="Path to Google Fonts checkout.")
    p.add_argument(
        "--embedder-path",
        default="models/style_embedding.pth",
        help="Path to the frozen FontStyleEmbedder checkpoint (needs a .conf.json sidecar).",
    )
    p.add_argument(
        "--char",
        default="a",
        help="Letter(s) to render for image distance. Pass several (e.g. 'aefg') "
             "for a less noisy multi-glyph distance (default: a).",
    )
    p.add_argument("--size", type=int, default=128, help="Glyph render size.")
    p.add_argument("--max-fonts", type=int, default=200, help="Number of fonts to sample.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--metrics",
        default="lpips,l1",
        help="Comma-separated image distances: lpips, l1, ssim.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Torch device (default: cuda if available, else cpu).",
    )
    p.add_argument("--plot", default=None, help="Optional path to save a scatter PNG.")
    p.add_argument("--save-dists", default=None, help="Optional .npz path for distance matrices.")
    args = p.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    print("Loading embedder…")
    embedder = load_embedder(args.embedder_path, device)

    print("Loading fonts…")
    chars = list(args.char)
    having = {ord(c) for c in chars}
    gf = GoogleFonts(args.repo, having=having)
    fonts = gf.fonts
    rng = random.Random(args.seed)
    rng.shuffle(fonts)
    fonts = fonts[: args.max_fonts]
    print(f"Sampled {len(fonts)} fonts.")

    # Build embeddings and per-char renderings, dropping blank glyphs.
    embeddings = []
    renderings_by_char: list[list[torch.Tensor]] = [[] for _ in chars]
    kept = []
    for font in fonts:
        char_imgs = []
        ok = True
        for c in chars:
            try:
                glyph = font.render(ord(c), size=args.size)
            except Exception:
                ok = False
                break
            glyph = np.asarray(glyph, dtype=np.float32)
            if glyph.ndim != 3 or glyph.shape[0] != 3:
                ok = False
                break
            if glyph[0].min() > 0.995:  # blank glyph
                ok = False
                break
            char_imgs.append(torch.from_numpy(glyph))
        if not ok:
            continue
        try:
            emb = embed_font(font, embedder, device)
        except Exception as exc:
            print(f"  skipped {font.path.name}: {exc}")
            continue
        embeddings.append(emb)
        for ci in range(len(chars)):
            renderings_by_char[ci].append(char_imgs[ci])
        kept.append(font)

    n = len(kept)
    if n < 2:
        raise SystemExit("Not enough renderable fonts; try a different --char or repo.")

    emb = torch.stack(embeddings).numpy()  # (N, D)
    imgs_by_char = [
        torch.stack(renderings_by_char[ci]) for ci in range(len(chars))
    ]  # list of (N, 3, H, W)

    print(f"\n{'-' * 60}")
    print(f"Fonts: {n}   Glyph(s): {args.char!r}   Embedding dim: {emb.shape[1]}")
    print(f"Device: {device}")
    print(f"{'-' * 60}")

    # Embedding distances: L2 (what the additive fusion actually sees) and cosine.
    from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

    d_emb_l2 = euclidean_distances(emb)
    d_emb_cos = cosine_distances(emb)

    # Image distances (averaged across the requested glyphs when multi-char).
    image_dists: dict[str, np.ndarray] = {}
    if "l1" in metrics:
        print(f"Computing pairwise L1 across {len(chars)} glyph(s)…")
        image_dists["l1"] = np.mean(
            [pairwise_l1(imgs) for imgs in imgs_by_char], axis=0
        )
    if "ssim" in metrics:
        print(f"Computing pairwise (1-SSIM) across {len(chars)} glyph(s)…")
        image_dists["ssim"] = np.mean(
            [pairwise_ssim(imgs) for imgs in imgs_by_char], axis=0
        )
    if "lpips" in metrics:
        print(f"Computing pairwise LPIPS across {len(chars)} glyph(s)…")
        from hrothgar.gtok.llamagen_lpips import LPIPS

        lpips = LPIPS().to(device).eval()
        image_dists["lpips"] = np.mean(
            [pairwise_lpips(imgs.to(device), lpips) for imgs in imgs_by_char], axis=0
        )

    # Report correlation + collapse for each combination.
    for img_name, d_img in image_dists.items():
        print(f"\n=== image distance: {img_name} ===")
        for emb_name, d_emb in [("L2", d_emb_l2), ("cosine", d_emb_cos)]:
            de = _upper_triangle(d_emb)
            di = _upper_triangle(d_img)
            pr, pr2 = pearson(de, di)
            sr, sr2 = spearman(de, di)
            print(
                f"  vs {emb_name:>6} embedding: "
                f"Pearson r={pr:.3f}  R2={pr2:.3f}  "
                f"Spearman rho={sr:.3f}"
            )
            if emb_name == "L2":
                c = collapse_metrics(d_emb, d_img)
                print(
                    "    collapse (bottom 10% d_emb): "
                    f"n_pairs={c['n_pairs']}  "
                    f"d_img_bottom_p90={c['d_img_bottom_p90']:.4f}  "
                    f"d_img_overall_p90={c['d_img_overall_p90']:.4f}  "
                    f"collapse_ratio={c['collapse_ratio']:.3f}"
                )

    # Optional plot of the primary relationship (LPIPS vs L2 embedding).
    if args.plot and "lpips" in image_dists:
        _plot(
            _upper_triangle(d_emb_l2),
            _upper_triangle(image_dists["lpips"]),
            args.plot,
            title=f"style collapse, char={args.char!r}",
        )

    # Optional dump of the matrices.
    if args.save_dists:
        np.savez(
            args.save_dists,
            d_emb_l2=d_emb_l2,
            d_emb_cos=d_emb_cos,
            **{f"d_img_{k}": v for k, v in image_dists.items()},
        )
        print(f"\nSaved distance matrices to {args.save_dists}")


def _plot(x: np.ndarray, y: np.ndarray, path: str, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=6, alpha=0.35)
    pr, _ = pearson(x, y)
    plt.xlabel("L2 embedding distance")
    plt.ylabel("LPIPS image distance")
    plt.title(f"{title}\nPearson r = {pr:.3f}")
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    print(f"Saved plot to {path}")


if __name__ == "__main__":
    main()
