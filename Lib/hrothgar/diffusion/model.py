"""Class-conditional diffusion glyph generator (Phase 1).

Wraps ``denoising_diffusion_pytorch.classifier_free_guidance``'s class-conditional
``Unet`` and ``GaussianDiffusion``.  We deliberately do **not** reimplement the
diffusion process: the library's ``GaussianDiffusion`` already handles noise
schedules, loss weighting, and (D)DIM sampling, and its ``Unet`` already threads a
class embedding through every ResNet block.

The only thing we add is a thin ``nn.Module`` facade so the training loop and the
canary can treat the whole model as:

    loss = model(images, classes)          # training
    glyphs = model.sample(classes)          # inference -> [0, 1]

Phase 2 will replace the class-embedding ``Unet`` with a custom exemplar-conditional
UNet (cross-attention onto a style feature map); the ``GaussianDiffusion`` wrapper
and this facade stay put.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from denoising_diffusion_pytorch.classifier_free_guidance import (
    GaussianDiffusion,
    Unet,
)

from hrothgar.diffusion.config import DiffusionConfig


class DiffusionGlyphModel(nn.Module):
    """Class-conditional diffusion model for grayscale glyphs."""

    def __init__(
        self, config: DiffusionConfig, glyphloss_fn: nn.Module | None = None
    ) -> None:
        super().__init__()
        if config.num_classes <= 0:
            raise ValueError(
                "DiffusionConfig.num_classes must be set (codepoint set x axis "
                "positions) before building the model."
            )
        if config.cond_scale == 1.0:
            # The library's forward_with_cond_scale returns a bare tensor when
            # cond_scale == 1, but model_predictions unpacks a two-tuple.
            raise ValueError("cond_scale must be != 1.0 (e.g. 3.0).")

        self.config = config
        # Auxiliary reconstruction loss (e.g. CurvatureWeightedGlyphLoss).
        # ``None`` or ``glyphloss_weight == 0`` disables it.
        self.glyphloss_fn = glyphloss_fn

        unet = Unet(
            dim=config.dim,
            num_classes=config.num_classes,
            cond_drop_prob=config.cond_drop_prob,
            dim_mults=config.dim_mults,
            channels=config.channels,
            attn_dim_head=config.attn_dim_head,
            attn_heads=config.attn_heads,
        )

        self.diffusion = GaussianDiffusion(
            unet,
            image_size=config.image_size,
            timesteps=config.timesteps,
            sampling_timesteps=config.sampling_timesteps,
            objective=config.objective,
            beta_schedule=config.beta_schedule,
            ddim_sampling_eta=config.ddim_sampling_eta,
        )

    def forward(self, images: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        """Return the diffusion training loss (noise-prediction MSE)."""
        return self.diffusion(images, classes=classes)

    def forward_with_aux(
        self,
        images: torch.Tensor,
        classes: torch.Tensor,
        apply_aux: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(total, diffusion_loss, aux_loss)`` for one training batch.

        ``total`` backpropagates the standard noise-prediction MSE plus, when
        ``apply_aux`` and enabled, a curvature-weighted glyph reconstruction
        loss on a *sampled* glyph (differentiable DDIM from pure noise).  The
        latter is what reaches the class conditioning — a single-timestep
        ``x_0`` recovery cannot, because at low noise ``x_0`` is already in the
        input.
        """
        diff_loss = self.diffusion(images, classes=classes)

        aux = torch.zeros((), device=images.device)
        if apply_aux and self.glyphloss_fn is not None and self.config.glyphloss_weight > 0:
            samples = self.sample_with_grad(
                classes, steps=self.config.glyphloss_sample_steps
            )
            aux = self.glyphloss_fn(
                samples.clamp(0.0, 1.0), images.clamp(0.0, 1.0)
            )
            total = diff_loss + self.config.glyphloss_weight * aux
        else:
            total = diff_loss

        return total, diff_loss.detach(), aux.detach()

    def sample_with_grad(
        self, classes: torch.Tensor, steps: int | None = None
    ) -> torch.Tensor:
        """Differentiable DDIM sampling (gradient flows back to the model).

        Mirrors the library's ``ddim_sample`` but keeps the computation graph,
        so an auxiliary reconstruction loss on the returned glyph trains the
        class conditioning.  ``steps`` defaults to
        ``config.glyphloss_sample_steps`` (a short chain keeps the backward
        graph affordable).

        Returns:
            ``(B, channels, H, W)`` in ``[0, 1]``.
        """
        diffusion = self.diffusion
        device = classes.device
        batch = classes.shape[0]
        shape = (
            batch,
            self.config.channels,
            self.config.image_size,
            self.config.image_size,
        )

        total = diffusion.num_timesteps
        steps = steps or self.config.glyphloss_sample_steps
        steps = min(max(int(steps), 1), total)
        eta = diffusion.ddim_sampling_eta

        times = torch.linspace(-1, total - 1, steps=steps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        # A single denoising step (CFG: two forward passes).  We checkpoint it
        # so the backward pass recomputes activations instead of storing a
        # whole chain of them — peak memory becomes ~one step, not ``steps``.
        def _denoise(img, t, cls):
            preds = diffusion.model_predictions(
                img,
                t,
                cls,
                cond_scale=self.config.cond_scale,
                rescaled_phi=0.7,
                clip_x_start=True,
            )
            return preds.pred_noise, preds.pred_x_start

        img = torch.randn(shape, device=device)
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start = torch.utils.checkpoint.checkpoint(
                _denoise, img, time_cond, classes, use_reentrant=False
            )

            if time_next < 0:
                img = x_start
                continue

            alpha = diffusion.alphas_cumprod[time]
            alpha_next = diffusion.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        # [-1, 1] -> [0, 1]
        return (img + 1.0) * 0.5

    @torch.no_grad()
    def sample(self, classes: torch.Tensor) -> torch.Tensor:
        """Sample glyphs conditioned on class ids.

        Args:
            classes: ``(B,)`` long tensor of class ids.

        Returns:
            ``(B, channels, H, W)`` float tensor in ``[0, 1]``.
        """
        return self.diffusion.sample(classes, cond_scale=self.config.cond_scale)


def build_diffusion_model(
    config: DiffusionConfig, glyphloss_fn: nn.Module | None = None
) -> DiffusionGlyphModel:
    """Construct a ``DiffusionGlyphModel`` from a config."""
    return DiffusionGlyphModel(config, glyphloss_fn=glyphloss_fn)
