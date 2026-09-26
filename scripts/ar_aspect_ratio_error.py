#!/usr/bin/env python
"""Break down AR generator glyphloss by the target glyph's aspect ratio.

Under the crop-to-ink policy every glyph is stretched anisotropically to a
square, so a font's *same* stroke width / terminal size / corner radius lands
at very different normalised sizes depending on the glyph's ink-bbox aspect
ratio (width / height): an "l" (tall, narrow) becomes a near-full-block stem
while an "m" (wide) becomes hairline stems.

If that per-glyph scale mismatch makes style transfer harder, we expect the
generator's error to be *worse at the extremes* (very narrow and very wide
glyphs) and best for roughly-square glyphs ("o", "n", "e").  This script
loads the trained generator and a slice of the validation set, computes
per-sample glyphloss for both

* full-context (bidirectional / teacher-forced) reconstruction, and
* iterative MaskGIT generation,

and reports each binned by aspect ratio, plus the correlation between
``|log2(width/height)|`` (distance from square) and glyphloss.

Example::

    PYTHONPATH=Lib python scripts/ar_aspect_ratio_error.py \
        --gtok-model-path models/gtok.pth \
        --ar-model-path models/maskgit_glyph_gen.pth \
        --style-embedder-path models/style_embedding.pth \
        --dataset-path "$GOOGLE_FONTS_REPO" \
        --num-batches 100
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import torch
import tqdm

from glyphloss import glyph_reconstruction_loss
from hrothgar.ar.dataset import ARPhase1DatasetMaker
from hrothgar.ar.model import ARModel, ARModelConfig
from hrothgar.gtok.model import load_model as load_gtok_model
from hrothgar.style_embedding import FontStyleEmbedder, FontStyleEmbedderConfig
from hrothgar.utils import pick_device

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_stack(args, device):
    """Load GTok + AR generator + font style embedder from checkpoints."""
    gtok, gtok_config = load_gtok_model(Path(args.gtok_model_path), device)
    image_size = gtok_config.image_size

    ar_config = ARModelConfig.from_sidecar(args.ar_model_path)
    if ar_config.image_size != image_size:
        raise ValueError(
            f"AR model image_size {ar_config.image_size} != GTok image_size "
            f"{image_size}"
        )
    model = ARModel(ar_config, gtok_model=gtok).to(device)
    model.load(args.ar_model_path, device)
    model.eval()

    style_config = FontStyleEmbedderConfig.from_sidecar(args.style_embedder_path)
    embedder = FontStyleEmbedder(style_config).to(device)
    embedder.load(args.style_embedder_path, device=device)
    embedder.eval()
    for p in embedder.parameters():
        p.requires_grad = False

    return gtok, model, embedder, image_size


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def per_sample_glyphloss(recon: torch.Tensor, target: torch.Tensor) -> list[float]:
    """Compute glyphloss per sample.

    ``glyph_reconstruction_loss`` returns a batch mean, so we call it one
    sample at a time to recover per-glyph values for bucketing.
    """
    return [
        glyph_reconstruction_loss(recon[i : i + 1], target[i : i + 1]).item()
        for i in range(recon.shape[0])
    ]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = a.std() * b.std() + 1e-12
    return float((a * b).mean() / denom)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report(ars: np.ndarray, bidir: np.ndarray, iter_loss: np.ndarray, args) -> None:
    has_iter = not args.skip_iterative

    print(f"\nSamples evaluated: {len(ars)}")
    print("\nAspect ratio (width / height) distribution:")
    for p in (0, 5, 25, 50, 75, 95, 100):
        print(f"  p{p:>3}: {np.percentile(ars, p):.3f}")
    print(f"  mean: {ars.mean():.3f}")

    # Five equal-count buckets (robust to the skewed aspect-ratio distribution).
    order = np.argsort(ars)
    bounds = np.array_split(order, 5)

    header = f"\n{'bucket':<7}{'n':>6}{'AR range':>20}"
    header += f"{'bidir mean':>12}{'bidir med':>11}"
    if has_iter:
        header += f"{'iter mean':>12}{'iter med':>11}"
    print("\nGlyphloss by aspect-ratio bucket (sorted low→high AR):")
    print(header)

    for b, idx in enumerate(bounds):
        if len(idx) == 0:
            continue
        lo, hi = ars[idx].min(), ars[idx].max()
        row = f"{b:<7}{len(idx):>6}{f'[{lo:.3f}, {hi:.3f}]':>20}"
        row += f"{bidir[idx].mean():>12.4f}{np.median(bidir[idx]):>11.4f}"
        if has_iter:
            row += f"{iter_loss[idx].mean():>12.4f}{np.median(iter_loss[idx]):>11.4f}"
        print(row)

    # Distance from square, so a U-shaped "worse at both extremes" effect shows
    # up as a positive correlation.
    distortion = np.abs(np.log2(np.maximum(ars, 1e-6)))
    print("\nCorrelation |log2(width/height)| vs glyphloss:")
    print(f"  full-context : r = {_pearson(distortion, bidir):+.3f}")
    if has_iter:
        print(f"  iterative    : r = {_pearson(distortion, iter_loss):+.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gtok-model-path", required=True)
    parser.add_argument("--ar-model-path", required=True)
    parser.add_argument("--style-embedder-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument(
        "--num-batches",
        type=int,
        default=100,
        help="Number of validation batches to evaluate (default: 100).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for the test loader (default: 32).",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=1234,
        help="Font/codepoint split seed, must match training (default: 1234).",
    )
    parser.add_argument(
        "--canary-size",
        type=int,
        default=None,
        help="Limit to this many fonts for a fast run (limits the style-embedding precompute).",
    )
    parser.add_argument(
        "--skip-iterative",
        action="store_true",
        help="Skip iterative generation (keep only full-context reconstruction).",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    print(f"Using device: {device}")

    gtok, model, embedder, image_size = load_stack(args, device)
    print(f"Loaded GTok + AR generator + style embedder (image_size={image_size})")

    # Build the test dataset.  This precomputes the font style embedding for
    # every font (the slow part, same as training setup).
    maker = ARPhase1DatasetMaker(
        args.dataset_path,
        batch_size=args.batch_size,
        font_style_embedder=embedder,
        embedder_device=device,
        image_size=image_size,
        split_seed=args.split_seed,
        canary_size=args.canary_size,
    )
    test_loader = maker.test_loader()

    ars: list[float] = []
    bidir_losses: list[float] = []
    iter_losses: list[float] = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm.tqdm(
            itertools.islice(test_loader, args.num_batches),
            total=args.num_batches,
            desc="Evaluating batches",
        ):
            target = batch["target_rendering"].to(device)
            content = batch["content_rendering"].to(device)
            style = batch["style_renderings"].to(device)
            emb = batch["font_style_embedding"].to(device)
            cp = batch["char"].to(device)
            metrics = batch.get("metrics")
            if metrics is not None:
                metrics = metrics.to(device)
            bbox = batch["bbox_size"].to(device)  # (B, 2) = (width, height)

            aspect = (bbox[:, 0] / bbox[:, 1].clamp(min=1e-6)).cpu().numpy()

            # Full-context (bidirectional) reconstruction.
            out = model(
                content,
                style,
                font_style_embedding=emb,
                target_images=target,
                target_codepoints=cp,
                metrics=metrics,
            )
            recon = out.reconstructed_images.clamp(0.0, 1.0)
            bidir = per_sample_glyphloss(recon, target)

            # Iterative MaskGIT generation.
            if args.skip_iterative:
                iter_loss = [float("nan")] * target.shape[0]
            else:
                gen = model.generate(
                    content,
                    style,
                    cp,
                    font_style_embedding=emb,
                    metrics=metrics,
                )
                gen_recon = gen.reconstructed_images.clamp(0.0, 1.0)
                iter_loss = per_sample_glyphloss(gen_recon, target)

            ars.extend(aspect.tolist())
            bidir_losses.extend(bidir)
            iter_losses.extend(iter_loss)

    report(
        np.asarray(ars),
        np.asarray(bidir_losses),
        np.asarray(iter_losses),
        args,
    )


if __name__ == "__main__":
    main()
