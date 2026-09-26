#!/usr/bin/env python
"""CNN vs ViT entanglement probe for the G-Tok encoder.

The code-entropy health check found that the tokenizer's codes are *not*
specialised by patch type: essentially every codebook entry is used on blank
(white), solid-ink (black), and edge (mix) patches alike, so the code for a
patch is a function of its position / global context rather than its local
content.

This probe asks whether that entanglement is introduced by the ViT's global
self-attention or is already present in the CNN features.  For the same glyph
corpus it extracts two feature maps:

* ``pre_vit``  — ``cnn_encoder`` → ``proj_patch`` (local, translation-equivariant)
* ``post_vit`` — ``pre_vit`` → ``vit_encoder`` (global self-attention)

and measures how well each can linearly separate the patch types
(white / black / mix).  If ``pre_vit`` separates them cleanly but ``post_vit``
does not, the ViT is the entangler.

Example::

    PYTHONPATH=Lib python scripts/gtok_cnn_vit_probe.py \
        --gtok-model-path models/gtok.pth \
        --dataset-path "$GOOGLE_FONTS_REPO"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from hrothgar.glyph_rendering import crop_to_ink
from hrothgar.googlefonts import GoogleFonts
from hrothgar.gtok.health import _classify_patches
from hrothgar.gtok.model import load_model
from hrothgar.utils import pick_device

_PROBE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
_PATCH_NAMES = ["white", "black", "mix"]


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def build_samples(gf: GoogleFonts, num_fonts: int, seed: int) -> list[tuple]:
    """Sample ``num_fonts`` fonts × a fixed character set into (font, cp) pairs."""
    rng = np.random.RandomState(seed)
    fonts = list(gf.fonts)
    rng.shuffle(fonts)
    samples: list[tuple] = []
    for font in fonts[:num_fonts]:
        for c in _PROBE_CHARS:
            cp = ord(c)
            if font.has_codepoint(cp):
                samples.append((font, cp))
    return samples


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def within_type_cosine(feats: np.ndarray, labels: np.ndarray) -> float:
    """Mean cosine similarity of each feature to its own class mean.

    Higher => tighter per-type clusters => more content-specialised.
    """
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    sims: list[np.ndarray] = []
    for c in np.unique(labels):
        mask = labels == c
        mean = feats[mask].mean(axis=0, keepdims=True)
        mean = mean / (np.linalg.norm(mean) + 1e-12)
        sims.append((feats[mask] @ mean.T).ravel())
    return float(np.concatenate(sims).mean()) if sims else float("nan")


def linear_probe_accuracy(feats: np.ndarray, labels: np.ndarray) -> float:
    """Held-out accuracy of a linear probe predicting patch type from features."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        feats, labels, test_size=0.2, random_state=0, stratify=labels
    )
    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(scaler.transform(X_tr), y_tr)
    return float(clf.score(scaler.transform(X_te), y_te))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gtok-model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument(
        "--num-fonts",
        type=int,
        default=100,
        help="Number of fonts to sample (default: 100).",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=200_000,
        help="Cap on total token patches collected (default: 200000).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--patch-margin",
        type=float,
        default=1.0,
        help="8-bit pixel margin for white/black/mix classification (default: 1).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    print(f"Using device: {device}")

    gtok, gtok_config = load_model(Path(args.gtok_model_path), device)
    image_size = gtok_config.image_size
    gtok.eval()
    for p in gtok.parameters():
        p.requires_grad = False
    print(f"Loaded G-Tok (image_size={image_size})")

    gf = GoogleFonts(args.dataset_path)
    samples = build_samples(gf, args.num_fonts, args.seed)
    print(f"Sampled {len(samples)} glyphs from {args.num_fonts} fonts")

    margin = args.patch_margin / 255.0
    grid_h, grid_w = gtok.token_grid_height, gtok.token_grid_width
    seq_len = grid_h * grid_w

    rng = np.random.RandomState(args.seed)
    pre_list: list[torch.Tensor] = []
    post_list: list[torch.Tensor] = []
    label_list: list[torch.Tensor] = []
    total = 0

    def _collate(batch: list[tuple]) -> torch.Tensor:
        images = [
            crop_to_ink(
                torch.tensor(font.render(cp, size=image_size), dtype=torch.float32),
                image_size,
            )
            for font, cp in batch
        ]
        return torch.stack(images)

    with torch.no_grad():
        it = range(0, len(samples), args.batch_size)
        for i in tqdm.tqdm(it, desc="Encoding patches"):
            images = _collate(samples[i : i + args.batch_size]).to(device)
            B = images.shape[0]

            # Same white/black/mix classification as the code-entropy check.
            category = _classify_patches(images, grid_h, grid_w, margin)  # (B, N)

            cnn_out = gtok.cnn_encoder(images)
            pre = gtok.proj_patch(cnn_out).flatten(2).transpose(1, 2)  # (B, N, D)
            post = gtok.vit_encoder(pre)  # (B, N, D)

            pre = pre.reshape(B * seq_len, -1)
            post = post.reshape(B * seq_len, -1)
            cat = category.reshape(B * seq_len)

            n = pre.shape[0]
            if total + n <= args.max_patches:
                keep = np.arange(n)
            else:
                keep = rng.choice(n, size=args.max_patches - total, replace=False)
            if len(keep) == 0:
                break

            pre_list.append(pre[keep].cpu())
            post_list.append(post[keep].cpu())
            label_list.append(cat[keep].cpu())
            total += len(keep)
            if total >= args.max_patches:
                break

    pre = torch.cat(pre_list).numpy()
    post = torch.cat(post_list).numpy()
    labels = torch.cat(label_list).numpy()
    print(f"\nTotal patches: {len(labels)}")

    counts = np.bincount(labels, minlength=3)
    print("Patch-type balance:")
    for c, name in enumerate(_PATCH_NAMES):
        print(f"  {name:<6}: {counts[c]:>7} ({100.0 * counts[c] / len(labels):.1f}%)")
    print(f"  majority-class baseline: {counts.max() / counts.sum():.3f}")

    pre_acc = linear_probe_accuracy(pre, labels)
    post_acc = linear_probe_accuracy(post, labels)
    print("\nLinear separability of patch type (white/black/mix):")
    print(f"  pre-ViT  (CNN) : acc = {pre_acc:.3f}")
    print(f"  post-ViT (ViT) : acc = {post_acc:.3f}")

    pre_sim = within_type_cosine(pre, labels)
    post_sim = within_type_cosine(post, labels)
    print("\nMean within-type cosine similarity (higher = tighter clusters):")
    print(f"  pre-ViT  (CNN) : {pre_sim:.3f}")
    print(f"  post-ViT (ViT) : {post_sim:.3f}")

    print("\nInterpretation:")
    if pre_acc > post_acc + 0.1 and pre_sim > post_sim + 0.05:
        print("  The ViT's global self-attention is the entangler: the CNN features")
        print("  separate patch types cleanly, but the ViT mixes in global context")
        print("  and destroys that local-content signal.")
    elif post_acc > pre_acc + 0.1:
        print("  The ViT IMPROVES patch-type separability (unexpected).")
    else:
        print("  The entanglement is already present in the CNN features (or the")
        print("  difference is small) — the ViT is not the (main) cause.")


if __name__ == "__main__":
    main()
