"""Training loop for the class-conditional diffusion glyph generator.

Phase 1 is class-conditional diffusion: the primary objective is the
noise-prediction MSE produced by ``DiffusionGlyphModel.forward``.  An optional
auxiliary glyph reconstruction loss (``glyphloss_weight > 0``) can be added on
the model's predicted ``x_0`` at low-noise timesteps to give the fine-detail
(edge/terminal) pixels an explicit gradient.

The trainer keeps just enough state (optimizer, model, device) for the canary and
future ``train.py`` entry points to share.
"""

from __future__ import annotations

import torch

from hrothgar.diffusion.config import DiffusionConfig
from hrothgar.diffusion.model import DiffusionGlyphModel


class DiffusionTrainer:
    """Optimizer + step + sampling for a ``DiffusionGlyphModel``."""

    def __init__(
        self,
        model: DiffusionGlyphModel,
        config: DiffusionConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.global_step = 0
        # Cache the most recent auxiliary loss so the canary's periodic report
        # (which need not land on an aux step) still shows a real value instead
        # of the 0 from a non-aux step.
        self._last_aux = torch.zeros(())

    def train_step(
        self, images: torch.Tensor, classes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One diffusion training step.

        The auxiliary glyph reconstruction loss is computed only every
        ``config.glyphloss_sample_every`` steps (backprop through the sample
        chain is expensive).  The returned ``aux_loss`` is the most recent
        auxiliary value (cached), so loggers see a real number even on
        non-aux steps.

        Returns:
            ``(total, diffusion_loss, aux_loss)``.  ``total`` is what was
            backpropagated; the other two are detached for logging.
        """
        apply_aux = (
            self.config.glyphloss_weight > 0
            and (self.global_step % self.config.glyphloss_sample_every) == 0
        )
        self.optimizer.zero_grad(set_to_none=True)
        total, diff_loss, aux_loss = self.model.forward_with_aux(
            images, classes, apply_aux=apply_aux
        )
        total.backward()
        self.optimizer.step()
        self.global_step += 1

        if apply_aux:
            self._last_aux = aux_loss.detach()
        return total.detach(), diff_loss.detach(), self._last_aux

    @torch.no_grad()
    def sample(self, classes: torch.Tensor) -> torch.Tensor:
        """Sample glyphs for the given class ids (``[0, 1]``)."""
        self.model.eval()
        out = self.model.sample(classes)
        self.model.train()
        return out
