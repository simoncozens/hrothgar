#!/usr/bin/env python
"""Break down AR generator reconstruction quality by codepoint and font category.

Loads a trained AR generator, samples held-out glyphs, and reports one-shot
reconstruction SSIM / LPIPS / glyphloss broken down by (a) codepoint and
(b) font category (sans / serif / display / handwriting / ...).

The point is to reveal whether the model is *uniformly* limited (a global
cold-start / conditioning bottleneck) or does much better on common text faces
than on rare display styles.

It also reports a "content-copy" baseline for each group: the SSIM / LPIPS /
glyphloss you would get by simply outputting the reference-font (content)
rendering instead of the model's reconstruction.  If the model's numbers track
this baseline, the model is mostly copying the content glyph (the reference-font
rendering of the target codepoint) and ignoring the style references.

Example::

    PYTHONPATH=Lib python scripts/ar_style_breakdown.py \
        --gtok-model-path models/gtok.pth \
        --ar-model-path models/maskgit_glyph_gen.pth \
        --dataset-path ~/google/fonts_checkout \
        --max-batches 100 \
        --steps 1
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

from glyphloss import glyph_reconstruction_loss
from hrothgar.ar.dataset import ARPhase1DatasetMaker
from hrothgar.ar.model import ARModel, ARModelConfig
from hrothgar.gtok.llamagen_lpips import LPIPS
from hrothgar.gtok.model import load_model as load_gtok_model
from hrothgar.utils import pick_device


def load_stack(args, device):
    """Load GTok and the AR generator."""
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

    return gtok, model, ar_config, image_size


def build_maker(args, ar_config, image_size):
    """Build the dataset maker (test split), matching the training split."""
    return ARPhase1DatasetMaker(
        args.dataset_path,
        batch_size=args.batch_size,
        image_size=image_size,
        style_glyph_count=args.style_glyph_count,
        common_style_codepoints=ar_config.style_codepoints,
        target_codepoints=ar_config.target_codepoints,
        target_only=ar_config.target_only,
        class_balanced=False,
        split_seed=args.split_seed,
        canary_size=args.limit_dataset_size,
    )


def collect_breakdown(
    model, maker, max_batches, batch_size, steps, ssim, lpips, device
):
    """Evaluate the model and return per-codepoint / per-category metric lists."""
    model.maskgit_decoder.config.num_inference_steps = steps

    # group -> list of (ssim, lpips, glyphloss)
    by_codepoint: dict[int, list] = defaultdict(list)
    by_category: dict[str, list] = defaultdict(list)
    content_by_codepoint: dict[int, list] = defaultdict(list)
    content_by_category: dict[str, list] = defaultdict(list)

    test_set = maker.test_set()
    batch = []
    n_batches = 0
    for item in test_set:
        batch.append(item)
        if len(batch) < batch_size:
            continue

        collated = maker.collate_fn(batch)
        val_target = collated["target_rendering"].to(device)
        val_content = collated["content_rendering"].to(device)
        val_style = collated["style_renderings"].to(device)
        val_cp = collated["char"].to(device)
        batch_metrics = collated.get("metrics")
        if batch_metrics is not None:
            batch_metrics = batch_metrics.to(device)

        gen_output = model.generate(
            content_images=val_content,
            style_reference_images=val_style,
            target_codepoints=val_cp,
            metrics=batch_metrics,
        )
        gen_recon = torch.clamp(gen_output.reconstructed_images, 0.0, 1.0).float()
        gen_target = torch.clamp(val_target, 0.0, 1.0).float()

        with torch.autocast(device_type=device.type, enabled=False):
            for i, item_ in enumerate(batch):
                recon_i = gen_recon[i : i + 1]
                target_i = gen_target[i : i + 1]
                s = float(ssim(recon_i, target_i))
                l = float(lpips(recon_i, target_i))
                g = float(glyph_reconstruction_loss(recon_i, target_i))

                cp = item_["char"]
                cat = item_["font"].category()
                by_codepoint[cp].append((s, l, g))
                by_category[cat].append((s, l, g))

                # "Copy the content glyph" baseline: how close is the
                # reference-font rendering to the target?  If the model's
                # output tracks this, the style path is being ignored.
                content_i = torch.clamp(val_content[i : i + 1], 0.0, 1.0)
                cs = float(ssim(content_i, target_i))
                cl = float(lpips(content_i, target_i))
                cg = float(glyph_reconstruction_loss(content_i, target_i))
                content_by_codepoint[cp].append((cs, cl, cg))
                content_by_category[cat].append((cs, cl, cg))

        batch = []
        n_batches += 1
        if n_batches >= max_batches:
            break

    if n_batches == 0:
        raise RuntimeError("No batches evaluated; is the test set empty?")

    return (
        by_codepoint,
        by_category,
        content_by_codepoint,
        content_by_category,
    )


def _mean_metrics(vals):
    n = len(vals)
    return (
        sum(v[0] for v in vals) / n,
        sum(v[1] for v in vals) / n,
        sum(v[2] for v in vals) / n,
        n,
    )


def _print_table(title, rows):
    """rows: list of (label, (ssim, lpips, glyphloss, n))."""
    print(f"\n{title}")
    print(f"  {'group':<14} {'n':>6} {'SSIM':>7} {'LPIPS':>7} {'glyphloss':>10}")
    for label, (s, l, g, n) in rows:
        print(f"  {label:<14} {n:>6} {s:>7.4f} {l:>7.4f} {g:>10.4f}")


def _print_category_comparison(model_by_category, content_by_category) -> None:
    """Print model output and content-copy baseline side by side."""
    print("\nPer-category: model output vs content-copy baseline")
    print(
        f"  {'category':<12} {'n':>6} {'SSIM':>7} {'cSSIM':>7} "
        f"{'LPIPS':>7} {'cLPIPS':>7} {'glyphloss':>10} {'cglyphloss':>11}"
    )
    for cat, vals in sorted(model_by_category.items()):
        ms, ml, mg, n = _mean_metrics(vals)
        cs, cl, cg, _ = _mean_metrics(content_by_category[cat])
        print(
            f"  {cat:<12} {n:>6} {ms:>7.4f} {cs:>7.4f} "
            f"{ml:>7.4f} {cl:>7.4f} {mg:>10.4f} {cg:>11.4f}"
        )


def report(
    by_codepoint, by_category, content_by_codepoint, content_by_category
) -> None:
    codepoint_rows = [
        (repr(chr(cp)), _mean_metrics(vals))
        for cp, vals in sorted(by_codepoint.items())
    ]
    _print_table("Per-codepoint (model output)", codepoint_rows)

    content_codepoint_rows = [
        (repr(chr(cp)), _mean_metrics(vals))
        for cp, vals in sorted(content_by_codepoint.items())
    ]
    _print_table("Per-codepoint (content-copy baseline)", content_codepoint_rows)

    _print_category_comparison(by_category, content_by_category)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gtok-model-path", required=True)
    p.add_argument("--ar-model-path", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Inference steps (default 1 = one-shot cold-start).",
    )
    p.add_argument(
        "--max-batches",
        type=int,
        default=100,
        help="Number of test batches to evaluate.",
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--style-glyph-count", type=int, default=8)
    p.add_argument(
        "--split-seed",
        type=int,
        default=1234,
        help="Seed for the train/test split (must match training).",
    )
    p.add_argument(
        "--limit-dataset-size",
        type=int,
        default=None,
        help="Limit the font count for a quick smoke test.",
    )
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    print(f"Using device: {device}")

    gtok, model, ar_config, image_size = load_stack(args, device)
    print(f"Loaded AR generator (image_size={image_size})")

    maker = build_maker(args, ar_config, image_size)

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LPIPS().to(device)

    (
        by_codepoint,
        by_category,
        content_by_codepoint,
        content_by_category,
    ) = collect_breakdown(
        model,
        maker,
        args.max_batches,
        args.batch_size,
        args.steps,
        ssim,
        lpips,
        device,
    )

    report(by_codepoint, by_category, content_by_codepoint, content_by_category)


if __name__ == "__main__":
    main()
