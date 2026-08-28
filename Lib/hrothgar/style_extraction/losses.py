"""Losses for the style autoencoder.

Reconstruction (L1 + glyphloss + LPIPS) is the primary objective — it is also
the acceptance signal, because an embedding is "rich" iff it can be decoded
back to high fidelity.  A PatchGAN adversarial term is available but disabled
by default (``adversarial`` weight 0) to keep the first version stable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hrothgar.style_extraction.config import StyleExtractionLossWeights


def reconstruction_loss(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    *,
    weights: StyleExtractionLossWeights,
    lpips_metric=None,
    glyphloss_fn=None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute reconstruction losses and loggable terms.

    Returns ``(total_loss, terms)`` where ``total_loss`` is the weighted scalar
    for backpropagation and ``terms`` holds the unweighted components.
    """
    l1 = F.l1_loss(reconstructed, target)

    glyphloss = torch.tensor(0.0, device=reconstructed.device)
    if glyphloss_fn is not None and weights.glyphloss > 0:
        glyphloss = glyphloss_fn(reconstructed, target)

    lpips = torch.tensor(0.0, device=reconstructed.device)
    if lpips_metric is not None and weights.perceptual_lpips > 0:
        recon_clamped = reconstructed.clamp(0.0, 1.0)
        target_clamped = target.clamp(0.0, 1.0)
        lpips = lpips_metric(recon_clamped, target_clamped).mean()

    total = (
        weights.l1 * l1
        + weights.glyphloss * glyphloss
        + weights.perceptual_lpips * lpips
    )

    terms: dict[str, torch.Tensor] = {
        "l1": l1.detach(),
        "glyphloss": glyphloss.detach(),
        "lpips": lpips.detach(),
    }
    return total, terms


def ink_coverage_loss(
    reconstructed: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Penalise outputs with substantially less ink than the target.

    Prevents the generator from collapsing to empty / near-empty glyphs when
    uncertain.  Only penalises a *deficit* of ink (not excess), so it doesn't
    fight the edge-focused glyphloss.
    """
    pred_ink = reconstructed.mean(dim=(1, 2, 3))
    target_ink = target.mean(dim=(1, 2, 3))
    deficit = torch.clamp(target_ink - pred_ink, min=0.0)
    return deficit.mean()


def adversarial_generator_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """LSGAN generator loss: push fake images toward the real manifold."""
    return F.mse_loss(fake_scores, torch.ones_like(fake_scores))


def adversarial_discriminator_loss(
    real_scores: torch.Tensor, fake_scores: torch.Tensor, real_label: float = 0.9
) -> torch.Tensor:
    """LSGAN discriminator loss: real -> ``real_label``, fake -> 0.

    ``real_label`` defaults to 0.9 (one-sided label smoothing) so the
    discriminator can't become overconfident and saturate its gradient back
    to the generator.
    """
    real_loss = F.mse_loss(
        real_scores, torch.full_like(real_scores, real_label)
    )
    fake_loss = F.mse_loss(fake_scores, torch.zeros_like(fake_scores))
    return 0.5 * (real_loss + fake_loss)


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator (pix2pix-style)."""

    def __init__(self, input_nc: int = 1, ndf: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(input_nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        nf = ndf
        for _ in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers += [
                nn.Conv2d(nf_prev, nf, 4, 2, 1),
                nn.InstanceNorm2d(nf),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        nf_prev = nf
        nf = min(nf * 2, 512)
        layers += [
            nn.Conv2d(nf_prev, nf, 4, 1, 1),
            nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, 1, 4, 1, 1),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
