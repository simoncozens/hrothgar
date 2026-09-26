#!/usr/bin/env python
"""Probe MaskGIT iterative decoding: per-step commit accuracy and calibration.

Loads a trained AR generator, runs iterative decode over the held-out test
split, and logs, at *each* inference step:

- ``commit_acc``  — fraction of this step's newly-revealed positions where the
                    model's own prediction matches the ground-truth token.
- ``mean_conf``   — mean softmax max-probability (confidence) of those commits.
- ``conf@corr`` / ``conf@wrong`` — mean confidence of the commits that were
                    correct vs. wrong.  If these two are close, confidence is
                    *miscalibrated* (the model can't tell good from bad tokens).
- ``step_ssim``   — SSIM of the partially-decoded image at this step.

This is the diagnostic for the cold-start gap: if ``commit_acc`` is ~13% at
step 1 and stays flat, iterative decoding can never bootstrap because it keeps
committing wrong tokens as context.

With ``--oracle-reveal``, the loop commits the *ground-truth* token at each
reveal position instead of the model's argmax.  This shows the ceiling: how
accurately the model *would* predict when it is handed correct context.  If
``commit_acc`` jumps up in oracle mode (while staying flat in normal mode), the
model is fine given context and the bottleneck is purely the empty-context
start; if it stays flat even in oracle mode, the reveal order / schedule is the
problem.

Note: in oracle mode the final step feeds the *full* ground-truth sequence back
to the transformer (the "bidirectional context" regime, which is not the
teacher-forced ceiling).  Interpret the final ``step_ssim`` in oracle mode via
the per-step trajectory, not the last row.

Example::

    PYTHONPATH=Lib python scripts/ar_iterative_probe.py \
        --gtok-model-path models/gtok.pth \
        --ar-model-path models/maskgit_glyph_gen.pth \
        --dataset-path ~/google/fonts_checkout \
        --scheduler confidence --steps 8 --max-batches 50
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure

from glyphloss import glyph_reconstruction_loss
from hrothgar.ar.dataset import ARPhase1DatasetMaker
from hrothgar.ar.maskgit import (
    ConfidenceSampler,
    HaltonSampler,
    _cosine_unmask_schedule,
)
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


@torch.no_grad()
def probe_batch(
    model: ARModel,
    batch: dict,
    *,
    oracle_reveal: bool,
    ssim,
    lpips,
    device: torch.device,
) -> tuple[list[dict], dict]:
    """Run one batch through the iterative loop, returning per-step logs.

    Mirrors ``MaskGITSampler.generate`` (see ``hrothgar/ar/maskgit.py``) so the
    diagnostics reflect the exact production decode path, with the one addition
    of logging each step's commits before applying them.
    """
    val_target = batch["target_rendering"].to(device)
    val_content = batch["content_rendering"].to(device)
    val_style = batch["style_renderings"].to(device)
    val_cp = batch["char"].to(device)
    batch_metrics = batch.get("metrics")
    if batch_metrics is not None:
        batch_metrics = batch_metrics.to(device)

    gt_tokens = model.target_token_indices_from_images(val_target)

    latincore_idx = model._unicode_to_latincore(val_cp)
    conditioning_map = model.build_conditioning_map(
        content_images=val_content,
        style_reference_images=val_style,
        latincore_idx=latincore_idx,
        metrics=batch_metrics,
    )

    decoder = model.maskgit_decoder
    transformer = decoder.transformer
    mask_token_id = decoder.mask_token_id
    N = decoder.sequence_length
    T = decoder.config.num_inference_steps
    temperature = decoder.config.temperature
    sampler = decoder.sampler

    batch_size = conditioning_map.shape[0]
    predicted = torch.full(
        (batch_size, N), mask_token_id, dtype=torch.long, device=device
    )
    unmasked = torch.zeros(batch_size, N, dtype=torch.bool, device=device)

    target = torch.clamp(val_target, 0.0, 1.0).float()

    step_logs: list[dict] = []
    for step in range(T):
        logits = transformer(idx=predicted, imgs_feature_map=conditioning_map)
        probs = F.softmax(logits / temperature, dim=-1)
        pred_tokens = probs.max(dim=-1).indices  # (B, N)
        conf = probs.max(dim=-1).values  # (B, N)

        target_keep = _cosine_unmask_schedule(step + 1, T, N)
        reveal = sampler._reveal(probs, unmasked, target_keep)  # (B, N) bool

        # Diagnostics reflect the model's *prediction* at this step's reveal,
        # regardless of whether we actually commit it or (in oracle mode) the
        # ground truth.
        correct = pred_tokens == gt_tokens
        n_commit = int(reveal.sum().item())
        commit_acc = correct[reveal].float().mean().item() if n_commit else float("nan")
        mean_conf = conf[reveal].mean().item() if n_commit else float("nan")
        conf_correct = (
            conf[reveal & correct].mean().item()
            if int((reveal & correct).sum().item())
            else float("nan")
        )
        conf_wrong = (
            conf[reveal & ~correct].mean().item()
            if int((reveal & ~correct).sum().item())
            else float("nan")
        )

        # Commit (prediction, or ground-truth for the oracle ceiling).
        if oracle_reveal:
            predicted[reveal] = gt_tokens[reveal]
        else:
            predicted[reveal] = pred_tokens[reveal]
        unmasked[reveal] = True

        # Partially-decoded image quality at this step.
        step_logits = transformer(idx=predicted, imgs_feature_map=conditioning_map)
        _, step_images = model.hard_decode(step_logits, temperature=1.0)
        step_recon = torch.clamp(step_images, 0.0, 1.0).float()
        with torch.autocast(device_type=device.type, enabled=False):
            step_ssim = float(ssim(step_recon, target))

        step_logs.append(
            {
                "n_commit": n_commit,
                "commit_acc": commit_acc,
                "mean_conf": mean_conf,
                "conf_correct": conf_correct,
                "conf_wrong": conf_wrong,
                "step_ssim": step_ssim,
            }
        )

    # Safety net: fill any still-masked positions (matches MaskGITSampler).
    remaining = ~unmasked
    if remaining.any():
        logits = transformer(idx=predicted, imgs_feature_map=conditioning_map)
        predicted[remaining] = torch.argmax(logits, dim=-1)[remaining]

    # Final reconstruction (matches model.generate).
    final_logits = transformer(idx=predicted, imgs_feature_map=conditioning_map)
    _, recon_images = model.hard_decode(final_logits, temperature=1.0)
    recon = torch.clamp(recon_images, 0.0, 1.0).float()
    with torch.autocast(device_type=device.type, enabled=False):
        s = float(ssim(recon, target))
        l = float(lpips(recon, target).mean())  # LPIPS is per-image (B,1,1,1)
        g = float(glyph_reconstruction_loss(recon, target))

    final = {
        "ssim": s,
        "lpips": l,
        "glyphloss": g,
        "tok_acc": float((predicted == gt_tokens).float().mean()),
    }
    return step_logs, final


def aggregate(all_step_logs, all_final, n_batches, batch_size, T):
    """Average per-step diagnostics and final metrics across batches."""
    agg = []
    for step in range(T):
        row = {
            "n_commit": sum(b[step]["n_commit"] for b in all_step_logs)
            / (n_batches * batch_size),
        }
        for key in (
            "commit_acc",
            "mean_conf",
            "conf_correct",
            "conf_wrong",
            "step_ssim",
        ):
            vals = [b[step][key] for b in all_step_logs if not math.isnan(b[step][key])]
            row[key] = sum(vals) / len(vals) if vals else float("nan")
        agg.append(row)

    final = {
        key: sum(f[key] for f in all_final) / n_batches
        for key in ("ssim", "lpips", "glyphloss", "tok_acc")
    }
    return agg, final


def print_report(agg, final, *, oracle_reveal, scheduler, steps, temperature):
    mode = "oracle-reveal" if oracle_reveal else "normal"
    print(
        f"\n=== scheduler={scheduler}  mode={mode}  "
        f"steps={steps}  temperature={temperature:g} ==="
    )
    print(
        f"  {'step':>4} {'n_commit':>9} {'commit_acc':>10} {'mean_conf':>9} "
        f"{'conf@corr':>9} {'conf@wrong':>10} {'step_ssim':>9}"
    )
    for i, row in enumerate(agg):
        print(
            f"  {i + 1:>4} {row['n_commit']:>9.1f} {row['commit_acc']:>10.4f} "
            f"{row['mean_conf']:>9.4f} {row['conf_correct']:>9.4f} "
            f"{row['conf_wrong']:>10.4f} {row['step_ssim']:>9.4f}"
        )
    print(
        f"  final: SSIM={final['ssim']:.4f}  LPIPS={final['lpips']:.4f}  "
        f"glyphloss={final['glyphloss']:.4f}  tok_acc={final['tok_acc']:.4f}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gtok-model-path", required=True)
    p.add_argument("--ar-model-path", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument(
        "--scheduler",
        default="confidence",
        choices=["confidence", "halton"],
        help="Inference reveal scheduler.",
    )
    p.add_argument("--steps", type=int, default=8, help="Number of inference steps.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--oracle-reveal",
        action="store_true",
        help="Commit ground-truth tokens instead of argmax.",
    )
    p.add_argument(
        "--max-batches",
        type=int,
        default=50,
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
    set_scheduler(model, args.scheduler)
    model.maskgit_decoder.config.num_inference_steps = args.steps
    model.maskgit_decoder.config.temperature = args.temperature
    print(
        f"Loaded AR generator (image_size={image_size}, " f"scheduler={args.scheduler})"
    )
    if args.oracle_reveal:
        print("ORACLE-REVEAL MODE: committing ground-truth tokens at each step")

    test_loader = build_test_loader(args, ar_config, image_size, device)

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LPIPS().to(device)

    all_step_logs, all_final = [], []
    n_batches = 0
    for batch in itertools.islice(test_loader, args.max_batches):
        step_logs, final = probe_batch(
            model,
            batch,
            oracle_reveal=args.oracle_reveal,
            ssim=ssim,
            lpips=lpips,
            device=device,
        )
        all_step_logs.append(step_logs)
        all_final.append(final)
        n_batches += 1

    if n_batches == 0:
        raise RuntimeError("No test batches evaluated; is the test loader empty?")

    agg, final = aggregate(
        all_step_logs, all_final, n_batches, args.batch_size, args.steps
    )
    print_report(
        agg,
        final,
        oracle_reveal=args.oracle_reveal,
        scheduler=args.scheduler,
        steps=args.steps,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
