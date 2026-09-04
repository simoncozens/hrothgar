"""Training loop for the factorized (codepoint, font-ID) diffusion model.

The real thing: a full Google Fonts dataset with a codepoint-based train/val
split (hold out ``(font, codepoint)`` pairs), training the VecFusion
"incomplete fonts" recipe and validating on the held-out pairs.
"""

from __future__ import annotations

import itertools
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from torchmetrics.image import StructuralSimilarityIndexMeasure

from hrothgar.diffusion.config import FontIdDiffusionConfig
from hrothgar.diffusion.dataset_fontid import FontIdDatasetMaker
from hrothgar.diffusion.fontid import build_fontid_model
from hrothgar.glyph_rendering import GEOMETRY_NAMES
from hrothgar.gtok.llamagen_lpips import LPIPS
from hrothgar.style_extraction.render_utils import render_glyph_with_geometry
from hrothgar.utils import TrainingLoop


class FontIdTrainingLoop(TrainingLoop):
    """Factorized font-ID diffusion training loop."""

    def post_init(self, train_args) -> None:
        maker = FontIdDatasetMaker(
            train_args.dataset_path,
            batch_size=train_args.batch_size,
            image_size=train_args.image_size,
            character_set=train_args.character_set,
            extra_codepoints=train_args.extra_codepoints,
            remove_codepoints=train_args.remove_codepoints,
            oversample_codepoints=train_args.oversample_codepoints,
            heldout_fraction=train_args.heldout_fraction,
            min_train_fonts_per_codepoint=train_args.min_train_fonts_per_codepoint,
            split_seed=train_args.split_seed,
            canary_size=train_args.limit_dataset_size,
        )
        self.maker = maker

        config = FontIdDiffusionConfig(
            image_size=train_args.image_size,
            num_codepoints=maker.num_codepoints,
            num_fonts=maker.num_fonts,
            dim=train_args.dim,
            timesteps=train_args.timesteps,
            sampling_timesteps=train_args.sampling_timesteps,
            learning_rate=train_args.learning_rate,
            geometry_loss_weight=train_args.geometry_weight,
        )
        config.save_sidecar(train_args.model_path)
        # Persist the font ordering so inference can map a font back to its id.
        with Path(str(train_args.model_path) + ".fonts.json").open("w") as f:
            json.dump([str(font.path) for font in maker.fonts], f, indent=2)
            f.write("\n")

        self.model = build_fontid_model(config).to(self.device)
        self.geometry_weight = train_args.geometry_weight
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=train_args.learning_rate
        )

        # Mixed precision.  bf16 needs no loss scaling (same exponent range as
        # fp32); the FFT-based glyphloss is not in this model's path, so bf16
        # is safe.
        self.use_amp = train_args.precision == "bf16"
        self.amp_dtype = torch.bfloat16 if self.use_amp else None
        if self.use_amp and self.device.type != "cuda":
            raise ValueError(
                f"precision={train_args.precision} requires CUDA, got {self.device}"
            )

        self.train_loader = maker.train_loader()
        self.val_loader = maker.val_loader()

        self.lpips = LPIPS().to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

        self.target_steps = train_args.target_steps
        self.validation_every = train_args.validation_every
        self.validation_batches = train_args.validation_batches
        self.num_epochs = (self.target_steps // max(len(self.train_loader), 1)) + 1
        self.validation_direction = "lower"  # minimize held-out LPIPS

    def _autocast_context(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def train_step(self, batch):
        images = batch["images"].to(self.device)
        codepoints = batch["codepoints"].to(self.device)
        font_ids = batch["font_ids"].to(self.device)
        geometry = batch["geometry"].to(self.device)

        with self._autocast_context():
            diffusion_loss = self.model(images, codepoints, font_ids)
            geometry_loss = self.model.geometry_loss(codepoints, font_ids, geometry)
            loss = diffusion_loss + self.geometry_weight * geometry_loss
        return loss, {
            "loss": loss.detach().float(),
            "diffusion": diffusion_loss.detach().float(),
            "geometry": geometry_loss.detach().float(),
        }

    def post_train_step(self):
        if self.global_step % self.validation_every != 0:
            return

        self.model.eval()
        with torch.no_grad():
            l1s, lpipss, ssims = [], [], []
            geo_mses, geo_per_value = [], []
            for batch in itertools.islice(self.val_loader, self.validation_batches):
                codepoints = batch["codepoints"].to(self.device)
                font_ids = batch["font_ids"].to(self.device)
                gts = batch["images"].to(self.device).float()
                geometry = batch["geometry"].to(self.device).float()

                with self._autocast_context():
                    recs = self.model.sample(codepoints, font_ids)
                    pred_geometry = self.model.predict_geometry(codepoints, font_ids)
                # Metrics in fp32 (LPIPS/SSIM are precision-sensitive).
                recs = recs.float().clamp(0.0, 1.0)
                pred_geometry = pred_geometry.float()
                l1s.append(F.l1_loss(recs, gts))
                lpipss.append(self.lpips(recs, gts).mean())
                ssims.append(self.ssim(recs, gts))
                geo_mses.append(F.mse_loss(pred_geometry, geometry))
                geo_per_value.append(((pred_geometry - geometry) ** 2).mean(0))

            l1 = torch.stack(l1s).mean()
            lpips = torch.stack(lpipss).mean()
            ssim = torch.stack(ssims).mean()
            self.write_scalar("Validation/L1", l1)
            self.write_scalar("Validation/LPIPS", lpips)
            self.write_scalar("Validation/SSIM", ssim)

            geometry_mse = torch.stack(geo_mses).mean()
            self.write_scalar("Validation/geometry_mse", geometry_mse)
            per_value = torch.stack(geo_per_value).mean(0)
            for name, value in zip(GEOMETRY_NAMES, per_value):
                self.write_scalar(f"Validation/geometry_{name}", value)

            self.checkpoint_if_best(lpips)
            self.visualize()

        self.model.train()

    def visualize(self):
        batch = next(iter(self.val_loader))
        n = min(8, batch["images"].shape[0])
        codepoints = batch["codepoints"][:n].to(self.device)
        font_ids = batch["font_ids"][:n].to(self.device)
        gts = batch["images"][:n].to(self.device).float()

        with torch.no_grad():
            with self._autocast_context():
                recs = self.model.sample(codepoints, font_ids)
        recs = recs.float().clamp(0.0, 1.0)

        # A second codepoint from each font, as a style reference for the eye,
        # plus the (family, codepoint) identity of each row for debugging.
        rng = random.Random(self.maker.split_seed)
        refs, text_lines = [], []
        for i in range(n):
            fid = font_ids[i].item()
            target_cp = self.maker.cp_list[codepoints[i].item()]
            font = self.maker.fonts[fid]
            avail = sorted(
                (set(font.codepoints) & set(self.maker.character_set)) - {target_cp}
            )
            ref_cp = rng.choice(avail) if avail else target_cp
            ref_img, _ = render_glyph_with_geometry(
                font, ref_cp, self.maker.image_size
            )
            refs.append(ref_img.unsqueeze(0))
            text_lines.append(
                f"{font.family} | target {chr(target_cp)!r} U+{target_cp:04X} "
                f"| ref {chr(ref_cp)!r} U+{ref_cp:04X}"
            )
        refs = torch.stack(refs).to(self.device)

        grid = torch.cat([gts, refs, recs], dim=0)
        self.writer.add_image(
            "Validation/gt_ref_recon",
            torchvision.utils.make_grid(grid, nrow=n),
            self.global_step,
        )
        self.writer.add_text("Validation/pairs", "\n".join(text_lines), self.global_step)


# ══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    from hrothgar.dataset import LATIN_KERNEL

    parser = argparse.ArgumentParser(description="Train factorized font-ID diffusion model")
    parser.add_argument("--dataset-path", type=str, default=os.environ.get("GOOGLE_FONTS_REPO"),
                        help="Path to the Google Fonts repository")
    parser.add_argument("--tag", type=str, help="Tag for the training run")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Allow training with uncommitted changes")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--character-set", type=str, default="LATIN_KERNEL",
                        help="Name of a character set in hrothgar.dataset (default LATIN_KERNEL)")
    parser.add_argument("--extra-codepoints", type=str, default="",
                        help="Extra codepoints to add to the vocabulary (a string of chars, e.g. '₹')")
    parser.add_argument("--remove-codepoints", type=str, default="",
                        help="Codepoints to remove from the vocabulary (e.g. 'ga' for bimodal g/a)")
    parser.add_argument("--oversample-codepoints", type=str, default="",
                        help="Codepoints to oversample in training (e.g. '₹')")
    parser.add_argument("--oversample-factor", type=int, default=20,
                        help="Duplication factor for --oversample-codepoints")
    parser.add_argument("--heldout-fraction", type=float, default=0.25,
                        help="Fraction of fonts to hold out per codepoint")
    parser.add_argument("--min-train-fonts-per-codepoint", type=int, default=20,
                        help="Minimum training fonts kept per codepoint")
    parser.add_argument("--split-seed", type=int, default=1234)
    parser.add_argument("--target-steps", type=int, default=600_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--geometry-weight", type=float, default=1.0,
                        help="Weight of the geometry regression objective (em-unit labels)")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=250)
    parser.add_argument("--sampling-timesteps", type=int, default=50)
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="bf16",
                        help="Training precision (bf16 = AMP, fp32 = no AMP)")
    parser.add_argument("--validation-every", type=int, default=1000)
    parser.add_argument("--validation-batches", type=int, default=20)
    parser.add_argument("--model-path", type=str,
                        default="models/fontid_diffusion.pth")
    parser.add_argument("--limit-dataset-size", type=int, default=None,
                        help="Limit to this many fonts for a canary run")

    args = parser.parse_args()
    if not args.dataset_path:
        raise ValueError("GOOGLE_FONTS_REPO not set and --dataset-path not given")

    # Resolve the named character set.
    import hrothgar.dataset as ds

    args.character_set = getattr(ds, args.character_set, None)
    if args.character_set is None:
        raise ValueError(f"unknown character set: {args.character_set}")

    # Parse codepoint strings into int lists / oversample dict.
    args.extra_codepoints = [ord(c) for c in args.extra_codepoints]
    args.remove_codepoints = [ord(c) for c in args.remove_codepoints]
    oversample_cps = [ord(c) for c in args.oversample_codepoints]
    args.oversample_codepoints = {cp: args.oversample_factor for cp in oversample_cps}

    loop = FontIdTrainingLoop(args)
    loop.train()
