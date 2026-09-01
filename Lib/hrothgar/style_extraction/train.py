"""Training loop for the style-extraction autoencoder.

Reconstruction is the objective *and* the test: we checkpoint on held-out
(test fonts, test codepoints) LPIPS, which measures whether the style summary
is rich enough to be extracted back to high fidelity on unseen compositions.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import replace
from typing import Optional

import torch
import torch.nn as nn
import torchvision
from torchmetrics.image import StructuralSimilarityIndexMeasure

from glyphloss import GlyphReconstructionLoss
from hrothgar.glyphloss_curvature import CurvatureWeightedGlyphLoss
from hrothgar.gtok.llamagen_lpips import LPIPS
from hrothgar.style_extraction.config import (
    LossSchedule,
    StyleExtractionLossWeights,
    StyleExtractionV2Config,
    StyleExtractionV3Config,
)
from hrothgar.style_extraction.dataset import StyleExtractionDatasetMaker
from hrothgar.style_extraction.losses import (
    NLayerDiscriminator,
    adversarial_discriminator_loss,
    adversarial_generator_loss,
    ink_coverage_loss,
    position_dependence_loss,
    reconstruction_loss,
    style_contrastive_loss,
    style_token_diversity_loss,
)
from hrothgar.style_extraction.model_v2 import StyleExtractionModelV2
from hrothgar.style_extraction.model_v3 import StyleExtractionModelV3
from hrothgar.utils import TrainingLoop


class StyleExtractionTrainingLoop(TrainingLoop):
    def post_init(self, train_args) -> None:
        self.model_version = getattr(train_args, "model_version", "v3")
        if self.model_version == "v3":
            config = StyleExtractionV3Config(
                image_size=getattr(train_args, "image_size", 128),
                num_evidence_glyphs=getattr(train_args, "num_evidence_glyphs", 16),
                decoder_self_attn_layers=getattr(train_args, "decoder_self_attn_layers", 2),
                cross_attn_pos_bias=getattr(train_args, "cross_attn_pos_bias", True),
            )
            model = StyleExtractionModelV3(config).to(self.device)
        else:
            config = StyleExtractionV2Config(
                image_size=getattr(train_args, "image_size", 128),
                num_evidence_glyphs=getattr(train_args, "num_evidence_glyphs", 16),
            )
            model = StyleExtractionModelV2(config).to(self.device)
        config.save_sidecar(train_args.model_path)

        self.loss_weights = StyleExtractionLossWeights(
            l1=getattr(train_args, "l1_weight", 1.0),
            glyphloss=getattr(train_args, "glyphloss_weight", 1.0),
            adversarial=getattr(train_args, "adversarial", 0.0),
            ink_coverage=getattr(train_args, "ink_coverage", 0.5),
            style_contrastive=getattr(train_args, "style_contrastive", 0.1),
            token_diversity=getattr(train_args, "token_diversity", 0.1),
            position_reg=getattr(train_args, "position_reg", 0.0),
        )
        self.schedule = LossSchedule(
            schedule_steps=getattr(train_args, "schedule_steps", 20000),
            l1_final=getattr(train_args, "l1_final_weight", 0.05),
            glyphloss_final=getattr(train_args, "glyphloss_final_weight", 3.0),
            lpips_final=getattr(train_args, "lpips_final_weight", 0.3),
        )
        self.position_reg_floor = getattr(train_args, "position_reg_floor", 1e-4)
        self.style_contrastive_temperature = getattr(
            train_args, "style_contrastive_temperature", 0.07
        )

        maker = StyleExtractionDatasetMaker(
            repo_url=str(train_args.dataset_path),
            batch_size=train_args.batch_size,
            image_size=config.image_size,
            character_set=config.character_set,
            num_evidence_glyphs=config.num_evidence_glyphs,
            canary_size=train_args.limit_dataset_size,
        )
        self.train_loader = maker.train_loader()
        self.test_loader = maker.test_loader()

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_args.learning_rate,
            weight_decay=train_args.weight_decay,
        )

        # Fine-detail glyphloss: curvature- and spectral-weighted.
        curvature_weight = getattr(train_args, "curvature_weight", 20.0)
        spectral_weight = getattr(train_args, "glyphloss_spectral_weight", 2.5)
        if curvature_weight > 0:
            self.glyphloss_fn = CurvatureWeightedGlyphLoss(
                k=curvature_weight,
                lambda_pixel=0.0,
                lambda_spectral=spectral_weight,
            ).to(self.device)
        else:
            self.glyphloss_fn = GlyphReconstructionLoss(
                lambda_pixel=0.0, lambda_spectral=spectral_weight
            ).to(self.device)

        self.lpips = LPIPS().to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

        # Optional adversarial training.
        self.discriminator: Optional[nn.Module] = None
        self.disc_optimizer: Optional[torch.optim.Optimizer] = None
        if self.loss_weights.adversarial > 0:
            cond_channels = 2 * config.glyph_encoder_feature_dim
            self.discriminator = NLayerDiscriminator(input_nc=cond_channels + 1).to(self.device)
            disc_lr = getattr(train_args, "discriminator_lr", None) or train_args.learning_rate
            self.disc_optimizer = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=disc_lr,
                betas=(0.5, 0.999),
            )

        self.model = model
        self.target_steps = train_args.target_steps
        self.num_epochs = getattr(train_args, "num_epochs", 100)
        self.validation_direction = "lower"  # minimize LPIPS
        self.validation_every = train_args.validation_every
        self.validation_batches = train_args.validation_batches

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, batch):
        style_images = batch["style_images"].to(self.device)
        style_codepoint_idx = batch["style_codepoint_idx"].to(self.device)
        target_images = batch["target_images"].to(self.device)
        target_idx = batch["target_codepoint_idx"].to(self.device)

        style_tokens = self.model.encode_style(
            style_images, style_codepoint_idx=style_codepoint_idx
        )
        if self.model_version == "v3":
            reconstructed, attn = self.model.decode(
                target_idx, style_tokens, return_attention=True
            )
        else:
            reconstructed = self.model.decode(target_idx, style_tokens)
            attn = None

        terms: dict[str, torch.Tensor] = {}

        # Token diversity: keep the K style tokens mutually decorrelated so
        # cross-attention has distinct axes to read rather than K copies of one
        # mean vector.
        token_div = style_token_diversity_loss(style_tokens)

        # Position-dependence guard: keep the decoder's cross-attention per-position
        # rather than collapsing to a global style vector (the v2 failure mode).
        position_reg = torch.tensor(0.0, device=self.device)
        if attn is not None and self.loss_weights.position_reg > 0:
            position_reg = position_dependence_loss(attn, self.position_reg_floor)

        adv_g = torch.tensor(0.0, device=self.device)
        if self.discriminator is not None and self.loss_weights.adversarial > 0:
            # Conditional discriminator input = [codepoint | style summary | image].
            # The conditioning is detached so the adversarial gradient flows only
            # through the generated image (the decoder), not the style encoder /
            # codepoint embedding.
            codepoint_emb = self.model.codepoint_embedding(target_idx)  # (B, D)
            style_summary = style_tokens.mean(dim=1)  # (B, D)
            cond = torch.cat([codepoint_emb, style_summary], dim=1)  # (B, 2D)
            cond_map = cond[:, :, None, None].expand(
                -1, -1, *target_images.shape[-2:]
            ).detach()  # (B, 2D, H, W)
            real_input = torch.cat([cond_map, target_images], dim=1)
            fake_input = torch.cat([cond_map, reconstructed], dim=1)

            # 1. Discriminator update on detached real/fake.
            self.disc_optimizer.zero_grad(set_to_none=True)
            real_scores = self.discriminator(real_input.detach())
            fake_scores = self.discriminator(fake_input.detach())
            d_loss = adversarial_discriminator_loss(real_scores, fake_scores)
            d_loss.backward()
            self.disc_optimizer.step()
            terms["d_loss"] = d_loss.detach()

            # 2. Generator adversarial loss (through decoder output).
            g_scores = self.discriminator(fake_input)
            adv_g = adversarial_generator_loss(g_scores)

        # Style contrastive: the generated glyph's style should match its own
        # evidence font, not other fonts in the batch.
        style_contr = torch.tensor(0.0, device=self.device)
        if self.loss_weights.style_contrastive > 0:
            output_style = self.model.encode_style(
                reconstructed.unsqueeze(1)
            ).mean(dim=1)
            evidence_style = style_tokens.mean(dim=1).detach()
            style_contr = style_contrastive_loss(
                output_style,
                evidence_style,
                temperature=self.style_contrastive_temperature,
            )

        # Coarse-to-fine schedule: L1/LPIPS ramp down (they blur fine detail late),
        # glyphloss ramps up (it's the localized fine-detail signal).
        l1_w = self.schedule.l1(self.global_step, self.loss_weights.l1)
        glyph_w = self.schedule.glyphloss(self.global_step, self.loss_weights.glyphloss)
        lpips_w = self.schedule.lpips(self.global_step, self.loss_weights.perceptual_lpips)
        weights = replace(
            self.loss_weights, l1=l1_w, glyphloss=glyph_w, perceptual_lpips=lpips_w
        )

        recon_total, recon_terms = reconstruction_loss(
            reconstructed,
            target_images,
            weights=weights,
            lpips_metric=self.lpips,
            glyphloss_fn=self.glyphloss_fn,
        )

        ink = ink_coverage_loss(reconstructed, target_images)

        total = (
            recon_total
            + self.loss_weights.adversarial * adv_g
            + self.loss_weights.ink_coverage * ink
            + self.loss_weights.style_contrastive * style_contr
            + self.loss_weights.token_diversity * token_div
            + self.loss_weights.position_reg * position_reg
        )
        terms.update(recon_terms)
        terms["ink_coverage"] = ink.detach()
        terms["style_contrastive"] = style_contr.detach()
        terms["token_diversity"] = token_div.detach()
        terms["position_reg"] = position_reg.detach()
        terms["l1_weight"] = torch.tensor(l1_w, device=self.device)
        terms["glyphloss_weight"] = torch.tensor(glyph_w, device=self.device)
        terms["lpips_weight"] = torch.tensor(lpips_w, device=self.device)
        if self.discriminator is not None and self.loss_weights.adversarial > 0:
            terms["adversarial_g"] = adv_g.detach()
            terms["adversarial_g_weighted"] = (
                self.loss_weights.adversarial * adv_g
            ).detach()

        terms["total"] = total.detach()
        return total, terms

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def post_train_step(self):
        if self.global_step % self.validation_every != 0:
            return

        self.model.eval()
        ssims: list[torch.Tensor] = []
        lpipss: list[torch.Tensor] = []
        glyphlosses: list[torch.Tensor] = []
        token_divs: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in itertools.islice(self.test_loader, self.validation_batches):
                style_images = batch["style_images"].to(self.device)
                style_codepoint_idx = batch["style_codepoint_idx"].to(self.device)
                target_images = batch["target_images"].to(self.device)
                target_idx = batch["target_codepoint_idx"].to(self.device)

                style_tokens = self.model.encode_style(
                    style_images, style_codepoint_idx=style_codepoint_idx
                )
                reconstructed = self.model.decode(target_idx, style_tokens)

                ssims.append(
                    self.ssim(reconstructed, target_images)
                )
                lpipss.append(
                    self.lpips(reconstructed.clamp(0, 1), target_images.clamp(0, 1)).mean()
                )
                glyphlosses.append(self.glyphloss_fn(reconstructed, target_images))
                token_divs.append(style_token_diversity_loss(style_tokens))

        if ssims:
            avg_ssim = torch.mean(torch.stack(ssims))
            avg_lpips = torch.mean(torch.stack(lpipss))
            avg_glyphloss = torch.mean(torch.stack(glyphlosses))
            avg_token_div = torch.mean(torch.stack(token_divs))
            self.write_scalar("Validation/SSIM", avg_ssim)
            self.write_scalar("Validation/LPIPS", avg_lpips)
            self.write_scalar("Validation/glyphloss", avg_glyphloss)
            self.write_scalar("Validation/token_diversity", avg_token_div)
            self.visualize()
            self.checkpoint_if_best(avg_lpips)

        self.model.train()

    def visualize(self):
        batch = next(iter(self.test_loader))
        style_images = batch["style_images"].to(self.device)
        style_codepoint_idx = batch["style_codepoint_idx"].to(self.device)
        target_images = batch["target_images"].to(self.device)
        target_idx = batch["target_codepoint_idx"].to(self.device)

        with torch.no_grad():
            reconstructed = self.model(
                style_images, target_idx, style_codepoint_idx=style_codepoint_idx
            )

        n = min(8, target_images.shape[0])
        grid = torch.cat([target_images[:n], reconstructed[:n]], dim=0)
        self.writer.add_image(
            "Reconstruction/GT_vs_Recon",
            torchvision.utils.make_grid(grid, nrow=n),
            self.global_step,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the style-extraction autoencoder")
    parser.add_argument("--dataset-path", type=str, default=os.environ.get("GOOGLE_FONTS_REPO"))
    parser.add_argument("--tag", type=str)
    parser.add_argument("--model-path", type=str, default="models/style_extraction.pth")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-evidence-glyphs", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-steps", type=int, default=500_000)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--validation-every", type=int, default=1000)
    parser.add_argument("--validation-batches", type=int, default=50)
    parser.add_argument("--limit-dataset-size", type=int, default=None)
    parser.add_argument("--curvature-weight", type=float, default=20.0)
    parser.add_argument("--glyphloss-spectral-weight", type=float, default=2.5)
    parser.add_argument("--adversarial", type=float, default=0.0)
    parser.add_argument("--model-version", type=str, default="v3", choices=["v2", "v3"],
                        help="Which decoder to train (v3 = SPADE, v2 = additive cross-attention).")
    parser.add_argument("--decoder-self-attn-layers", type=int, default=2,
                        help="Self-attention layers on content queries before cross-attention (v3).")
    parser.add_argument("--cross-attn-pos-bias", dest="cross_attn_pos_bias",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Add a learned positional bias to the cross-attention (breaks global collapse).")
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--l1-final-weight", type=float, default=0.05,
                        help="L1 weight at the end of the coarse-to-fine schedule.")
    parser.add_argument("--lpips-final-weight", type=float, default=0.3,
                        help="LPIPS weight at the end of the schedule.")
    parser.add_argument("--glyphloss-weight", type=float, default=1.0,
                        help="glyphloss weight at the start of the schedule.")
    parser.add_argument("--glyphloss-final-weight", type=float, default=3.0,
                        help="glyphloss weight at the end of the schedule.")
    parser.add_argument("--schedule-steps", type=int, default=20000,
                        help="Horizon (in steps) over which the cosine schedule runs.")
    parser.add_argument("--position-reg", type=float, default=0.0,
                        help="Weight of the position-dependence guard on cross-attention "
                             "(0.0 = off).  Soft nudge only; needs a large weight to matter.")
    parser.add_argument("--position-reg-floor", type=float, default=1e-3,
                        help="q_var floor below which the position-dependence loss is active.")
    parser.add_argument("--ink-coverage", type=float, default=0.5)
    parser.add_argument("--style-contrastive", type=float, default=0.1)
    parser.add_argument("--style-contrastive-temperature", type=float, default=0.07)
    parser.add_argument("--token-diversity", type=float, default=0.1,
                        help="Weight of the style-token diversity loss (0.0 to "
                             "disable).  Keeps the K Perceiver latents mutually "
                             "decorrelated instead of collapsing to one summary.")
    parser.add_argument("--discriminator-lr", type=float, default=None)
    args = parser.parse_args()

    if not args.dataset_path:
        raise ValueError("GOOGLE_FONTS_REPO environment variable not set")

    loop = StyleExtractionTrainingLoop(args)
    loop.train()
