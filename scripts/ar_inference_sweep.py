#!/usr/bin/env python
"""Sweep MaskGIT inference hyperparameters (steps, temperature, scheduler).

Loads a trained AR generator (plus GTok), builds the held-out test split, and
evaluates iterative decode across a grid of (num_inference_steps, temperature,
scheduler), reporting SSIM / LPIPS / glyphloss / token accuracy at each point.

This is the evaluation harness for tuning exposure-bias fixes and for
comparing the confidence scheduler against the Halton scheduler.

With ``--oracle``, each sample's style references are replaced with the target
glyph itself — a "perfect oracle" that tells us whether the model can use a
directly relevant style example, or whether the bottleneck is the
style-reference path / cold-start generation itself.

Example::

    PYTHONPATH=Lib python scripts/ar_inference_sweep.py \
        --gtok-model-path models/gtok.pth \
        --ar-model-path models/maskgit_glyph_gen.pth \
        --dataset-path ~/google/fonts_checkout \
        --schedulers confidence \
        --steps 1,4,8,16,32,64 \
        --temperatures 1.0 \
        --max-batches 50
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

from glyphloss import glyph_reconstruction_loss
from hrothgar.ar.dataset import ARPhase1DatasetMaker
from hrothgar.ar.maskgit import ConfidenceSampler, HaltonSampler
from hrothgar.ar.model import ARModel, ARModelConfig
from hrothgar.gtok.llamagen_lpips import LPIPS
from hrothgar.gtok.model import load_model as load_gtok_model
from hrothgar.utils import pick_device


def _parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_strings(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


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


def build_test_loader(args, ar_config, image_size, device):
    """Build the held-out test loader, matching the training split."""
    maker = ARPhase1DatasetMaker(
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
    return maker.test_loader()


def set_scheduler(model: ARModel, name: str) -> None:
    """Swap the MaskGIT decoder's inference sampler in place."""
    config = model.maskgit_decoder.config
    config.scheduler = name
    if name == "confidence":
        model.maskgit_decoder.sampler = ConfidenceSampler(config)
    elif name == "halton":
        model.maskgit_decoder.sampler = HaltonSampler(config)
    else:
        raise ValueError(f"Unknown scheduler: {name!r}")


def evaluate(
    model, test_loader, max_batches, ssim, lpips, device, oracle=False
) -> dict:
    """Run iterative decode over the test set and return aggregated metrics.

    When ``oracle`` is set, the style references are replaced with the target
    glyph itself (the model sees the exact answer in its style path), isolating
    whether the style-reference conditioning can transfer a directly relevant
    example.
    """
    ssim_vals, lpips_vals, glyph_vals, tok_vals = [], [], [], []
    n_batches = 0
    for batch in itertools.islice(test_loader, max_batches):
        val_target = batch["target_rendering"].to(device)
        val_content = batch["content_rendering"].to(device)
        val_style = batch["style_renderings"].to(device)
        val_cp = batch["char"].to(device)
        batch_metrics = batch.get("metrics")
        if batch_metrics is not None:
            batch_metrics = batch_metrics.to(device)

        if oracle:
            val_style = val_target.unsqueeze(1).expand_as(val_style)

        gen_output = model.generate(
            content_images=val_content,
            style_reference_images=val_style,
            target_codepoints=val_cp,
            metrics=batch_metrics,
        )

        gt_tokens = model.target_token_indices_from_images(val_target)
        gen_recon = torch.clamp(gen_output.reconstructed_images, 0.0, 1.0).float()
        gen_target = torch.clamp(val_target, 0.0, 1.0).float()

        with torch.autocast(device_type=device.type, enabled=False):
            ssim_vals.append(ssim(gen_recon, gen_target))
            lpips_vals.append(lpips(gen_recon, gen_target))
            glyph_vals.append(glyph_reconstruction_loss(gen_recon, gen_target))
        tok_vals.append((gen_output.target_token_indices == gt_tokens).float().mean())

        n_batches += 1

    if n_batches == 0:
        raise RuntimeError("No test batches evaluated; is the test loader empty?")

    return {
        "ssim": float(torch.stack(ssim_vals).mean()),
        "lpips": float(torch.stack(lpips_vals).mean()),
        "glyphloss": float(torch.stack(glyph_vals).mean()),
        "token_acc": float(torch.stack(tok_vals).mean()),
        "n_batches": n_batches,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gtok-model-path", required=True)
    p.add_argument("--ar-model-path", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument(
        "--schedulers",
        default="confidence",
        help="Comma-separated schedulers to sweep (confidence, halton).",
    )
    p.add_argument(
        "--steps",
        default="4,8,16,32",
        help="Comma-separated inference step counts to sweep.",
    )
    p.add_argument(
        "--temperatures",
        default="1.0",
        help="Comma-separated softmax temperatures to sweep.",
    )
    p.add_argument(
        "--max-batches",
        type=int,
        default=50,
        help="Number of test batches per sweep point.",
    )
    p.add_argument(
        "--oracle",
        action="store_true",
        help="Replace style refs with the target glyph (perfect oracle).",
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

    schedulers = _parse_strings(args.schedulers)
    steps_list = _parse_ints(args.steps)
    temperatures = _parse_floats(args.temperatures)

    gtok, model, ar_config, image_size = load_stack(args, device)
    print(
        f"Loaded AR generator (image_size={image_size}, "
        f"default scheduler={ar_config.maskgit_scheduler})"
    )
    if args.oracle:
        print("ORACLE MODE: style references = target glyph")

    test_loader = build_test_loader(args, ar_config, image_size, device)

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LPIPS().to(device)

    for scheduler in schedulers:
        set_scheduler(model, scheduler)
        print(f"\n=== scheduler: {scheduler} ===")
        for temperature in temperatures:
            model.maskgit_decoder.config.temperature = temperature
            print(f"  temperature={temperature:g}")
            print(
                f"  {'steps':>6} {'SSIM':>7} {'LPIPS':>7} "
                f"{'glyphloss':>10} {'tok_acc':>8}"
            )
            for steps in steps_list:
                model.maskgit_decoder.config.num_inference_steps = steps
                metrics = evaluate(
                    model,
                    test_loader,
                    args.max_batches,
                    ssim,
                    lpips,
                    device,
                    oracle=args.oracle,
                )
                print(
                    f"  {steps:>6} {metrics['ssim']:>7.4f} "
                    f"{metrics['lpips']:>7.4f} {metrics['glyphloss']:>10.4f} "
                    f"{metrics['token_acc']:>8.4f}"
                )

    print(
        f"\nDone ({len(schedulers)} scheduler(s), "
        f"{len(temperatures)} temperature(s), {len(steps_list)} step count(s))."
    )


if __name__ == "__main__":
    main()
