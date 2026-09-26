"""Diagnose whether the decoder is using per-token style structure.

Two things can make the style-swap test show only coarse changes while the
mean-pooled linear probe still passes:

1. The Perceiver collapses its K style tokens to near-identical copies — the
   *mean* is informative (so the probe passes), but there is no per-token
   structure for cross-attention to exploit.  This is a representation problem
   at the token-set level (fix the Perceiver / add diversity pressure).
2. The tokens are diverse but the decoder's cross-attention effectively pools
   them (only reads the mean).  This is a usage problem (fix the decoder's
   conditioning).

This script measures both:

* token diversity — off-diagonal cosine similarity of the K tokens, plus the
  singular-value spectrum (effective rank) of the centered (K, D) token matrix.
* mean-ablation — decode with the full token set vs. K copies of the mean
  token.  If the outputs are ~identical, the decoder only uses the mean.

Interpretation
--------------
- off-diag cosine ≈ 1, effective rank ≈ 1  →  Perceiver collapse (fix encoder).
- off-diag cosine low, effective rank high, but mean-ablation L1 ≈ 0
  →  tokens are rich but the decoder ignores them (fix decoder).
- both healthy and mean-ablation L1 large  →  per-token style is used; the
  coarse-only swap is then a matter of *what* the tokens encode, not whether
  they are used.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from hrothgar.style_extraction import load_model
from hrothgar.style_extraction.dataset import StyleExtractionDatasetMaker
from hrothgar.utils import torch_setup


def _effective_rank(singular_values: torch.Tensor) -> float:
    s = singular_values + 1e-8
    return float((s.sum() ** 2) / (s**2).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = torch_setup()
    model, config = load_model(args.model_path, device)

    maker = StyleExtractionDatasetMaker(
        repo_url=args.dataset_path,
        batch_size=args.batch_size,
        image_size=config.image_size,
        character_set=config.character_set,
        num_evidence_glyphs=config.num_evidence_glyphs,
    )

    batch = next(iter(maker.test_loader()))
    style_images = batch["style_images"].to(device)
    style_cp = batch["style_codepoint_idx"].to(device)
    target_idx = batch["target_codepoint_idx"].to(device)

    with torch.no_grad():
        tokens = model.encode_style(style_images, style_codepoint_idx=style_cp)

    k = tokens.shape[1]
    eye = torch.eye(k, dtype=torch.bool, device=device)

    print(f"Style tokens: {k} (dim {tokens.shape[-1]})")
    print("Per-font token diversity:")
    for i in range(tokens.shape[0]):
        t = tokens[i]  # (K, D)
        tn = F.normalize(t, dim=-1)
        sim = tn @ tn.T  # (K, K)
        off = sim[~eye]
        t_centered = t - t.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(t_centered)
        er = _effective_rank(s)
        # Fraction of variance in the top singular direction.
        top_share = float((s[0] ** 2) / (s**2).sum())
        print(
            f"  font {i:2d}: off-diag cos mean={off.mean():.3f} std={off.std():.3f} | "
            f"effective_rank={er:5.1f}/{k} | top-dir variance={top_share:.2f}"
        )

    # Mean ablation: replace the token set with K copies of its mean.
    with torch.no_grad():
        full = model.decode(target_idx, tokens)
        mean_tokens = tokens.mean(dim=1, keepdim=True).expand(-1, k, -1)
        mean_out = model.decode(target_idx, mean_tokens)
    l1 = torch.abs(full - mean_out).mean().item()
    print(f"\nMean-ablation L1 (full tokens vs. K copies of mean): {l1:.6f}")
    if l1 < 1e-3:
        print(
            "  → decoder only uses the mean of the tokens (per-token structure unused)."
        )
    else:
        print("  → decoder uses per-token structure (difference is meaningful).")


if __name__ == "__main__":
    main()
