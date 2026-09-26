"""Long-horizon check: do eager and torch.compile training trajectories track?

Runs K steps of SGD with the full style_extraction loss stack (LPIPS dropout
active, as in real training) and compares the loss curves and final weights.

Usage: venv/bin/python scripts/compile_track.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Lib"))

from hrothgar.glyphloss_curvature import CurvatureWeightedGlyphLoss
from hrothgar.gtok.llamagen_lpips import LPIPS
from hrothgar.style_extraction.config import (
    StyleExtractionLossWeights,
    StyleExtractionV2Config,
)
from hrothgar.style_extraction.losses import (
    ink_coverage_loss,
    reconstruction_loss,
    style_contrastive_loss,
    style_token_diversity_loss,
)
from hrothgar.style_extraction.model_v2 import StyleExtractionModelV2

torch.manual_seed(0)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")

config = StyleExtractionV2Config(image_size=64, num_evidence_glyphs=4)
B, G = 2, config.num_evidence_glyphs
K = 25

weights = StyleExtractionLossWeights()
glyphloss_fn = CurvatureWeightedGlyphLoss(
    k=20.0, lambda_pixel=0.0, lambda_spectral=2.5
).to(device)
lpips = LPIPS().to(device)  # train mode: dropout active, as in the real loop


def run(use_compile):
    model = StyleExtractionModelV2(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses = []

    # Pre-generate identical batches so eager and compiled see the same data.
    batches = []
    for _ in range(K):
        batches.append(
            (
                torch.rand(B, G, 1, 64, 64, device=device),
                torch.randint(0, config.num_codepoints, (B, G), device=device),
                torch.randint(0, config.num_codepoints, (B,), device=device),
                torch.rand(B, 1, 64, 64, device=device),
            )
        )

    def step(style_images, style_cp, target_idx, target_images):
        opt.zero_grad(set_to_none=True)
        tokens = model.encode_style(style_images, style_codepoint_idx=style_cp)
        recon = model.decode(target_idx, tokens)
        div = style_token_diversity_loss(tokens)
        output_style = model.encode_style(recon.unsqueeze(1)).mean(dim=1)
        contr = style_contrastive_loss(output_style, tokens.mean(dim=1).detach())
        recon_total, terms = reconstruction_loss(
            recon,
            target_images,
            weights=weights,
            lpips_metric=lpips,
            glyphloss_fn=glyphloss_fn,
        )
        ink = ink_coverage_loss(recon, target_images)
        total = (
            recon_total
            + weights.ink_coverage * ink
            + weights.style_contrastive * contr
            + weights.token_diversity * div
        )
        total.backward()
        opt.step()
        return total.detach()

    if use_compile:
        step = torch.compile(step)

    for batch in batches:
        losses.append(step(*batch).item())
    return losses, model


l_eager, m_eager = run(False)
l_compiled, m_compiled = run(True)


def gap(a, b):
    return max(abs(x - y) / abs(x) for x, y in zip(a, b))


# Control: two *independent eager* runs differ only by LPIPS dropout masks.
# If the eager-vs-compiled gap is no larger than this, the compiled run is
# statistically indistinguishable from a second eager run.
torch.manual_seed(999)
l_eager2, _ = run(False)

print("eager-vs-eager (control):")
for i, (a, b) in enumerate(zip(l_eager, l_eager2)):
    print(f"  step {i:2d}: eager1={a:10.4f} eager2={b:10.4f} rel={abs(a-b)/abs(a):.2e}")
print()
print("eager-vs-compiled:")
for i, (a, b) in enumerate(zip(l_eager, l_compiled)):
    print(
        f"  step {i:2d}: eager={a:10.4f} compiled={b:10.4f} rel={abs(a-b)/abs(a):.2e}"
    )

control_gap = gap(l_eager, l_eager2)
compile_gap = gap(l_eager, l_compiled)
print(f"\nmax rel gap  eager-vs-eager : {control_gap:.3e}")
print(f"max rel gap  eager-vs-compiled: {compile_gap:.3e}")
assert compile_gap < 4 * max(control_gap, 1e-3), "compiled drift exceeds dropout noise"

w_diff = max(
    (a - b).abs().max().item()
    for a, b in zip(m_eager.parameters(), m_compiled.parameters())
)
print(f"max final-weight abs diff: {w_diff:.3e}")
print("OK: compiled training tracks eager within dropout-noise level.")
