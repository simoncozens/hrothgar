"""VQ loss module adapted from upstream GAR-Font.

Only the LPIPS import path has been changed — everything else is identical
to the upstream ``GAR-Font/model/tokenizer/vq_loss.py``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from hrothgar.upstream.lpips import LPIPS


class VQ_loss(nn.Module):
    def __init__(
        self,
        image_size=256,
        reconstruction_loss="l2",
        reconstruction_weight=1.0,
        codebook_weight=1.0,
        perceptual_weight=1.0,
    ):
        super().__init__()
        # reconstruction loss
        if reconstruction_loss == "l1":
            self.rec_loss = F.l1_loss
        elif reconstruction_loss == "l2":
            self.rec_loss = F.mse_loss
        else:
            raise ValueError(f"Unknown rec loss '{reconstruction_loss}'.")
        self.rec_weight = reconstruction_weight

        # codebook loss
        self.codebook_weight = codebook_weight

        # perceptual loss
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

    def forward(
        self,
        codebook_loss,
        inputs,
        reconstructions,
        global_step,
        logger=None,
        log_every=100,
    ):
        # reconstruction loss
        rec_loss = self.rec_loss(inputs.contiguous(), reconstructions.contiguous())

        # perceptual loss
        p_loss = torch.tensor(0.0, device=inputs.device)
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(
                inputs.contiguous(), reconstructions.contiguous()
            )
            p_loss = torch.mean(p_loss)

        # codebook loss
        cb_loss = (
            codebook_loss[0] + codebook_loss[1] + codebook_loss[2]
        )  # vq_loss, commit_loss, entropy_loss

        loss = (
            self.rec_weight * rec_loss
            + self.perceptual_weight * p_loss
            + self.codebook_weight * cb_loss
        )

        if global_step % log_every == 0:
            logger.info(
                f"(Generator) rec_loss: {rec_loss:.4f}, perceptual_loss: {p_loss:.4f}, "
                f"vq_loss: {codebook_loss[0]:.4f}, commit_loss: {codebook_loss[1]:.4f}, "
                f"entropy_loss: {codebook_loss[2]:.4f}, codebook_usage: {codebook_loss[3]:.4f}"
            )
        return loss
