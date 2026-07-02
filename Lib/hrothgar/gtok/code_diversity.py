"""Quantify the many-to-one mapping in the G-Tok codebook.

Diagnostic for whether multiple distinct code sequences can decode to
perceptually equivalent glyphs — a phenomenon that makes token-level
cross-entropy a misleading objective for AR model training.

Two experiments:

1. **Intra-image code diversity** — encode the same glyph N times with
   small Gaussian noise injected before quantization.  Measure (a) what
   fraction of token positions are invariant, (b) pairwise SSIM/LPIPS
   between all decoded variants.

2. **Code distance vs visual distance** — for pairs of *different* glyphs,
   measure Hamming distance between code sequences and SSIM between
   decoded images.  A weak correlation implies that similar codes can
   produce visually similar results and that dissimilar codes can
   sometimes still produce similar results.

Usage::

    python -m hrothgar.gtok.code_diversity \\
        --gtok-model-path models/gtok_model.pth \\
        --dataset-path $GOOGLE_FONTS_REPO
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import tqdm
from piqa import SSIM as PIQA_SSIM

from hrothgar.googlefonts import GoogleFonts
from hrothgar.gtok.llamagen_lpips import LPIPS
from hrothgar.gtok.model import GtokModel, load_model
from hrothgar.utils import torch_setup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CodeDiversityConfig:
    gtok_model_path: str = "models/gtok_model.pth"
    dataset_path: str = os.environ.get("GOOGLE_FONTS_REPO", "")
    num_samples: int = 64  # glyphs to test per experiment
    num_perturbations: int = 10  # N encodings per glyph (experiment 1)
    noise_std: float = 0.05  # std of Gaussian noise added pre-quantization
    device: str = ""

    # Derived — set after model load.
    codebook_size: int = 0
    code_dim: int = 0
    sequence_length: int = 0
    image_size: int = 0


# ---------------------------------------------------------------------------
# Helper: encode with optional pre-quantization noise
# ---------------------------------------------------------------------------


@torch.no_grad()
def _encode_noisy(
    gtok: GtokModel,
    image: torch.Tensor,  # (1, 3, H, W) on correct device
    noise_std: float,
) -> torch.Tensor:
    """Encode one glyph, optionally adding Gaussian noise pre-quantization.

    Returns code indices of shape ``(sequence_length,)``.
    """
    cnn_out = gtok.cnn_encoder(image)
    tokens = gtok.proj_patch(cnn_out).flatten(2).transpose(1, 2)
    vit_out = gtok.vit_encoder(tokens)
    pre_quant = gtok.vit_encoder_to_quantizer(vit_out)

    # Same reshape as GtokModel.encode.
    batch_size, _seq_length, _code_dim = pre_quant.shape
    pre_quant_4d = pre_quant.reshape(
        batch_size,
        gtok.token_grid_height,
        gtok.token_grid_width,
        gtok.config.quantizer_code_dim,
    ).permute(0, 3, 1, 2)

    # Inject noise before quantization.
    if noise_std > 0:
        pre_quant_4d = pre_quant_4d + torch.randn_like(pre_quant_4d) * noise_std

    # Quantize; indices_info[2] is min_encoding_indices.
    _quantized, _loss_info, indices_info = gtok.quantizer(pre_quant_4d)
    return indices_info[2].squeeze(0)  # (sequence_length,)


# ---------------------------------------------------------------------------
# Experiment 1: Intra-image code diversity
# ---------------------------------------------------------------------------


def _run_intra_image_diversity(
    gtok: GtokModel,
    config: CodeDiversityConfig,
    device: torch.device,
) -> dict:
    """Encode each glyph N times with noise and measure code + visual diversity."""
    gf = GoogleFonts(config.dataset_path)

    # Collect (font, codepoint) samples.
    samples: list[tuple] = []
    fonts = sorted(gf.fonts, key=lambda f: f.family)
    for font in fonts:
        if len(samples) >= config.num_samples:
            break
        for cp in sorted(font.codepoints)[:10]:  # first 10 chars per font
            if len(samples) >= config.num_samples:
                break
            samples.append((font, cp))
    samples = samples[: config.num_samples]

    # Metrics.
    invariant_fractions: list[float] = []
    all_n_unique: list[torch.Tensor] = []  # per-glyph (seq_len,) tensors
    all_pairwise_ssim: list[float] = []
    all_pairwise_lpips: list[float] = []

    ssim_metric = PIQA_SSIM(n_channels=3, value_range=1.0).to(device)
    lpips_metric = LPIPS().to(device)

    for font, cp in tqdm.tqdm(samples, desc="Intra-image diversity"):
        image = torch.tensor(
            font.render(cp, size=config.image_size), dtype=torch.float32
        )
        if float(image.max()) == float(image.min()):
            continue
        image = image.unsqueeze(0).to(device)  # (1, 3, H, W)

        # Encode N times with noise, decode each variant.
        all_codes: list[torch.Tensor] = []
        all_images: list[torch.Tensor] = []
        for _ in range(config.num_perturbations):
            codes = _encode_noisy(gtok, image, config.noise_std)  # (seq_len,)
            all_codes.append(codes)

            # Decode this code sequence.
            # Must use L2-normalized embeddings to match what the decoder expects.
            if gtok.quantizer.l2_norm:
                emb = torch.nn.functional.normalize(
                    gtok.quantizer.embedding.weight, p=2, dim=-1
                )
            else:
                emb = gtok.quantizer.embedding.weight
            quant_from_codes = emb[codes].unsqueeze(0)  # (1, seq_len, code_dim)
            recon = gtok.decode(quant_from_codes)
            recon = torch.clamp(recon, 0.0, 1.0)
            all_images.append(recon.squeeze(0))  # (3, H, W)

        # --- Code-level invariance ---
        code_stack = torch.stack(all_codes, dim=0)  # (K, seq_len)
        n_unique = torch.tensor(
            [len(set(code_stack[:, i].tolist())) for i in range(code_stack.shape[1])]
        )
        invariant_frac = (n_unique == 1).float().mean().item()
        invariant_fractions.append(invariant_frac)

        # Accumulate per-position uniqueness for spatial analysis.
        all_n_unique.append(n_unique)  # list of (seq_len,) tensors

        # --- Visual pairwise consistency ---
        img_stack = torch.stack(all_images, dim=0)  # (K, 3, H, W)
        for i in range(config.num_perturbations):
            for j in range(i + 1, config.num_perturbations):
                a = img_stack[i].unsqueeze(0)
                b = img_stack[j].unsqueeze(0)
                all_pairwise_ssim.append(ssim_metric(a, b).item())
                all_pairwise_lpips.append(lpips_metric(a, b).item())

    n_valid = len(invariant_fractions)
    avg_invariant = sum(invariant_fractions) / max(n_valid, 1)
    avg_ssim = sum(all_pairwise_ssim) / max(len(all_pairwise_ssim), 1)
    avg_lpips = sum(all_pairwise_lpips) / max(len(all_pairwise_lpips), 1)

    # Spatial invariance map: average n_unique per position, reshaped to grid.
    spatial_map: list[list[float]] = []
    if all_n_unique:
        mean_n_unique = torch.stack(all_n_unique).float().mean(dim=0)  # (seq_len,)
        # sequence_length should be a perfect square for the token grid.
        grid_size = int(config.sequence_length**0.5)
        if grid_size * grid_size == config.sequence_length:
            spatial_map = mean_n_unique.reshape(grid_size, grid_size).tolist()

    return {
        "n_samples": n_valid,
        "n_perturbations": config.num_perturbations,
        "noise_std": config.noise_std,
        "avg_invariant_fraction": avg_invariant,
        "avg_pairwise_ssim": avg_ssim,
        "avg_pairwise_lpips": avg_lpips,
        "spatial_n_unique": spatial_map,
        "all_invariant_fractions": invariant_fractions,
    }


# ---------------------------------------------------------------------------
# Experiment 2: Code distance vs visual distance
# ---------------------------------------------------------------------------


@torch.no_grad()
def _run_code_vs_visual_distance(
    gtok: GtokModel,
    config: CodeDiversityConfig,
    device: torch.device,
) -> dict:
    """For pairs of different glyphs, measure code Hamming distance vs visual SSIM."""
    gf = GoogleFonts(config.dataset_path)

    # Build a pool of (font, cp) samples.
    samples: list[tuple] = []
    fonts = sorted(gf.fonts, key=lambda f: f.family)
    for font in fonts:
        if len(samples) >= 256:
            break
        for cp in sorted(font.codepoints)[:5]:
            if len(samples) >= 256:
                break
            samples.append((font, cp))

    # Pre-tokenize all samples (deterministic — no noise).
    all_codes: list[torch.Tensor] = []
    all_images: list[torch.Tensor] = []
    for font, cp in tqdm.tqdm(samples, desc="Tokenizing for distance experiment"):
        image = torch.tensor(
            font.render(cp, size=config.image_size), dtype=torch.float32
        )
        if float(image.max()) == float(image.min()):
            continue
        image_gpu = image.unsqueeze(0).to(device)
        codes = _encode_noisy(gtok, image_gpu, noise_std=0.0)
        all_codes.append(codes.cpu())
        all_images.append(image)
        # (image stays on CPU for SSIM computation)

    n = len(all_codes)
    if n < 2:
        return {"error": "Not enough valid samples for distance experiment"}

    ssim_metric = PIQA_SSIM(n_channels=3, value_range=1.0).to(device)

    # Sample random pairs.
    num_pairs = min(1000, n * (n - 1) // 2)
    rng = torch.Generator().manual_seed(42)
    pair_i = torch.randint(0, n, (num_pairs,), generator=rng)
    pair_j = torch.randint(0, n - 1, (num_pairs,), generator=rng)
    # Ensure j != i: if j >= i, shift by 1.
    pair_j = torch.where(pair_j >= pair_i, pair_j + 1, pair_j)

    code_dists: list[float] = []
    visual_dists: list[float] = []  # 1 - SSIM

    for i, j in tqdm.tqdm(
        zip(pair_i.tolist(), pair_j.tolist()),
        total=num_pairs,
        desc="Computing distances",
    ):
        cd = (all_codes[i] != all_codes[j]).float().mean().item()
        code_dists.append(cd)

        a = all_images[i].unsqueeze(0).to(device)
        b = all_images[j].unsqueeze(0).to(device)
        s = ssim_metric(a, b).item()
        visual_dists.append(1.0 - s)

    code_tensor = torch.tensor(code_dists)
    visual_tensor = torch.tensor(visual_dists)

    # Pearson correlation.
    mean_cd = code_tensor.mean()
    mean_vd = visual_tensor.mean()
    cov = ((code_tensor - mean_cd) * (visual_tensor - mean_vd)).mean()
    std_cd = code_tensor.std()
    std_vd = visual_tensor.std()
    pearson = (cov / (std_cd * std_vd + 1e-8)).item()

    # Bucketed averages (10 equal-width buckets by code distance).
    num_buckets = 10
    bucket_edges = torch.linspace(0, 1, num_buckets + 1)
    bucket_stats: list[dict] = []
    for k in range(num_buckets):
        lo, hi = bucket_edges[k].item(), bucket_edges[k + 1].item()
        mask = (code_tensor >= lo) & (code_tensor < hi)
        if k == num_buckets - 1:
            mask = mask | (code_tensor >= hi)
        n_in = int(mask.sum().item())
        if n_in == 0:
            continue
        bucket_stats.append(
            {
                "code_dist_min": lo,
                "code_dist_max": hi,
                "n_pairs": n_in,
                "mean_code_dist": code_tensor[mask].mean().item(),
                "mean_visual_dist": visual_tensor[mask].mean().item(),
                "mean_ssim": (1.0 - visual_tensor[mask]).mean().item(),
            }
        )

    return {
        "n_samples": n,
        "n_pairs": num_pairs,
        "pearson_r": pearson,
        "mean_code_dist": code_tensor.mean().item(),
        "mean_ssim": (1.0 - visual_tensor.mean()).item(),
        "buckets": bucket_stats,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_results_1(results: dict) -> None:
    n_s = results["n_samples"]
    inv_frac = results["avg_invariant_fraction"]
    ssim = results["avg_pairwise_ssim"]
    lpips = results["avg_pairwise_lpips"]

    print(f"Valid samples:              {n_s}")
    print(f"Avg invariant fraction:     {inv_frac:.4f}  ({inv_frac * 100:.1f}%)")
    print(f"  → {100 - inv_frac * 100:.1f}% of token positions vary across encodings")
    print(f"Avg pairwise SSIM:          {ssim:.4f}")
    print(f"Avg pairwise LPIPS:         {lpips:.4f}")
    print()

    if inv_frac < 0.6:
        print("⚠️  LOW invariance: fewer than 60% of token positions are stable")
        print("    across encodings of the same glyph. The many-to-one mapping is")
        print("    SIGNIFICANT — many tokens have multiple valid choices.")
    elif inv_frac < 0.85:
        print(f"⚡ MODERATE invariance: {inv_frac * 100:.0f}% stable token positions.")
        print("    Some many-to-one ambiguity exists. Token accuracy is")
        print("    moderately misleading as a primary metric.")
    else:
        print(f"✓ HIGH invariance: {inv_frac * 100:.0f}% stable token positions.")
        print("    The many-to-one mapping is minimal — token accuracy is a")
        print("    reasonable primary metric.")

    if ssim > 0.95 and inv_frac < 0.6:
        print()
        print("    🔴 CRITICAL: SSIM is very high despite low code invariance.")
        print("    → Many distinct code sequences decode to perceptually")
        print("    identical images. Token CE is punishing the AR model for")
        print("    picking tokens that are actually fine.")

    # Spatial heatmap: mean unique codes per position (1.0 = invariant).
    spatial = results.get("spatial_n_unique", [])
    if spatial:
        print()
        print("  Spatial invariance map (mean unique codes per position):")
        print("  Lower = more ambiguous.  1.0 = always same code.")
        grid_h = len(spatial)
        grid_w = len(spatial[0]) if spatial else 0
        # Compact ASCII heatmap.
        # Shade: ░ = high variance (low invariance), █ = fully invariant.
        shades = " ░▒▓█"
        for row in spatial:
            row_str = "  "
            for val in row:
                # val ranges from 1.0 (invariant) to num_perturbations (max variance).
                # Map to 0–1 where 0 = invariant, 1 = high variance.
                norm = min((val - 1.0) / (results["n_perturbations"] - 1.0), 1.0)
                idx = int(norm * (len(shades) - 1))
                row_str += shades[idx]
                row_str += f"{val:.2f} " if grid_w <= 16 else ""
            print(row_str)
        print()


def _print_results_2(results: dict) -> None:
    if "error" in results:
        print(f"  Error: {results['error']}")
        return

    pearson = results["pearson_r"]
    print(f"Samples in pool:       {results['n_samples']}")
    print(f"Pairs evaluated:       {results['n_pairs']}")
    print(f"Pearson r (code↔visual): {pearson:.4f}")
    print(f"Mean code distance:    {results['mean_code_dist']:.4f}")
    print(f"Mean SSIM:             {results['mean_ssim']:.4f}")
    print()

    print("Code-distance buckets → visual similarity:")
    header = (
        f"  {'Code range':>12s}  {'N':>5s}  {'Mean SSIM':>10s}  {'Mean vis dist':>14s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for b in results["buckets"]:
        print(
            f"  [{b['code_dist_min']:.3f}, {b['code_dist_max']:.3f})  "
            f"{b['n_pairs']:>5d}  {b['mean_ssim']:>10.4f}  {b['mean_visual_dist']:>14.4f}"
        )
    print()

    if abs(pearson) < 0.3:
        print("⚠️  WEAK correlation: code distance and visual distance are nearly")
        print("    independent. Large code differences can produce visually similar")
        print("    images. Token accuracy is a POOR proxy for visual quality.")
    elif pearson > 0.7:
        print("✓ STRONG correlation: code distance tracks visual distance well.")
        print("    Token accuracy is a reasonable proxy for visual quality.")
    else:
        print(f"⚡ MODERATE correlation (r={pearson:.3f}): some relationship but not")
        print("    tight. Token accuracy has limited value as a proxy metric.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantify many-to-one mapping in G-Tok codebook"
    )
    parser.add_argument(
        "--gtok-model-path",
        type=str,
        default="models/gtok_model.pth",
        help="Path to trained G-Tok weights (.pth)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=os.environ.get("GOOGLE_FONTS_REPO", ""),
        help="Path to the Google Fonts repository",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="Number of glyphs to test in experiment 1",
    )
    parser.add_argument(
        "--num-perturbations",
        type=int,
        default=10,
        help="Number of noisy encodings per glyph",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.05,
        help="Std of Gaussian noise added pre-quantization",
    )
    args = parser.parse_args()

    if not args.dataset_path:
        parser.error(
            "--dataset-path is required (or set GOOGLE_FONTS_REPO environment variable)"
        )

    device = torch_setup()
    print(f"Device: {device}")

    # Load model.
    gtok, gtok_config = load_model(Path(args.gtok_model_path), device=device)
    gtok.eval()
    for p in gtok.parameters():
        p.requires_grad = False

    config = CodeDiversityConfig(
        gtok_model_path=args.gtok_model_path,
        dataset_path=args.dataset_path,
        num_samples=args.num_samples,
        num_perturbations=args.num_perturbations,
        noise_std=args.noise_std,
        codebook_size=gtok_config.quantizer_codebook_size,
        code_dim=gtok_config.quantizer_code_dim,
        sequence_length=gtok.sequence_length,
        image_size=gtok_config.image_size,
    )

    print("=" * 72)
    print("G-Tok Code Diversity Diagnostic")
    print("=" * 72)
    print(f"Model:           {args.gtok_model_path}")
    print(f"Codebook size:   {config.codebook_size}")
    print(f"Code dimension:  {config.code_dim}")
    print(f"Sequence length: {config.sequence_length}")
    print(f"Image size:      {config.image_size}")
    print()

    # --- Experiment 1 ---
    print("─" * 72)
    print("EXPERIMENT 1: Intra-image code diversity")
    print("─" * 72)
    print(
        f"Encoding {config.num_samples} glyphs {config.num_perturbations} times "
        f"each (noise std={config.noise_std})"
    )
    print()
    results_1 = _run_intra_image_diversity(gtok, config, device)
    _print_results_1(results_1)

    # --- Experiment 2 ---
    print("─" * 72)
    print("EXPERIMENT 2: Code distance vs visual distance")
    print("─" * 72)
    print()
    results_2 = _run_code_vs_visual_distance(gtok, config, device)
    _print_results_2(results_2)

    print("Done.")


if __name__ == "__main__":
    main()
