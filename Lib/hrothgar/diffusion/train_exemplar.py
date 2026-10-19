"""Training loop for the exemplar-conditional diffusion model.

Like the fontid loop, this trains on a full Google Fonts dataset with a
codepoint-based train/val split, but the style conditioning is a set of
*evidence glyphs* (selected content-aware, see ``dataset_exemplar.py``) encoded
into a style context map that the UNet cross-attends to — no per-font lookup.

Run ``python -m hrothgar.diffusion.train_exemplar --dataset-path <checkout>
--num-instances N --model-path models/exemplar_diffusion.pth``.
"""

from __future__ import annotations

import itertools
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchmetrics.image import StructuralSimilarityIndexMeasure

from hrothgar.diffusion.config import ExemplarDiffusionConfig
from hrothgar.diffusion.dataset_exemplar import ExemplarDatasetMaker
from hrothgar.diffusion.exemplar import build_exemplar_model
from hrothgar.llamagen_lpips import LPIPS
from hrothgar.render import GEOMETRY_NAMES, place_glyph
from hrothgar.utils import TrainingLoop, apply_cli_defaults, load_cli_defaults


class ExemplarTrainingLoop(TrainingLoop):
    """Exemplar-conditional diffusion training loop."""

    def post_init(self, train_args) -> None:
        maker = ExemplarDatasetMaker(
            train_args.dataset_path,
            batch_size=train_args.batch_size,
            image_size=train_args.image_size,
            num_evidence_glyphs=train_args.num_evidence_glyphs,
            character_set=train_args.character_set,
            extra_codepoints=train_args.extra_codepoints,
            remove_codepoints=train_args.remove_codepoints,
            oversample_codepoints=train_args.oversample_codepoints,
            num_instances=train_args.num_instances,
            strata=train_args.subset_strata,
            max_per_unit=train_args.max_per_unit,
            avg_instances=train_args.avg_instances,
            target_frac=train_args.target_frac,
            prefer_multi_target=train_args.prefer_multi_target,
            replacement=train_args.replacement,
            subset_seed=train_args.subset_seed,
            heldout_fraction=train_args.heldout_fraction,
            min_train_fonts_per_codepoint=train_args.min_train_fonts_per_codepoint,
            split_seed=train_args.split_seed,
        )
        self.maker = maker

        config = ExemplarDiffusionConfig(
            image_size=train_args.image_size,
            num_codepoints=maker.num_codepoints,
            context_dim=train_args.context_dim,
            style_out_res=train_args.style_out_res,
            style_encoder_base_channels=train_args.style_encoder_base_channels,
            num_evidence_glyphs=train_args.num_evidence_glyphs,
            dim=train_args.dim,
            timesteps=train_args.timesteps,
            sampling_timesteps=train_args.sampling_timesteps,
            learning_rate=train_args.learning_rate,
            warmup_steps=train_args.warmup_steps,
            geometry_loss_weight=train_args.geometry_weight,
            geometry_std=maker.geometry_std,
            min_snr_gamma=train_args.min_snr_gamma,
        )
        config.save_sidecar(
            train_args.model_path,
            extra={"cli_args": getattr(train_args, "cli_args", {})},
        )
        with Path(str(train_args.model_path) + ".codepoints.json").open("w") as f:
            json.dump(maker.cp_list, f, indent=2)
            f.write("\n")
        with Path(str(train_args.model_path) + ".instances.json").open("w") as f:
            json.dump(maker.instance_sidecar(), f, indent=2)
            f.write("\n")

        self.model = build_exemplar_model(config).to(self.device)
        self.geometry_weight = train_args.geometry_weight
        self.base_lr = train_args.learning_rate
        self.warmup_steps = train_args.warmup_steps
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=train_args.learning_rate
        )

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

    def _set_lr(self) -> float:
        """Linear LR warmup from 0 to ``base_lr`` over ``warmup_steps``."""
        if self.warmup_steps > 0:
            frac = min(1.0, (self.global_step + 1) / self.warmup_steps)
            lr = self.base_lr * frac
        else:
            lr = self.base_lr
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def train_step(self, batch):
        self._set_lr()
        images = batch["images"].to(self.device)
        evidence = batch["evidence"].to(self.device)
        codepoints = batch["codepoints"].to(self.device)
        geometry = batch["geometry"].to(self.device)

        with self._autocast_context():
            context = self.model.encode_context(evidence)
            diffusion_loss, x0_pred = self.model.forward_with_x0(
                images, context, codepoints
            )
            geometry_loss = self.model.geometry_loss(
                codepoints, context, x0_pred, geometry
            )
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
                evidence = batch["evidence"].to(self.device)
                codepoints = batch["codepoints"].to(self.device)
                gts = batch["images"].to(self.device).float()
                geometry = batch["geometry"].to(self.device).float()

                with self._autocast_context():
                    context = self.model.encode_context(evidence)
                    recs = self.model.sample(context, codepoints)
                    pred_geometry = self.model.predict_geometry(
                        codepoints, context, recs
                    )
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
            self.writer.add_scalar(
                "LearningRate",
                self.optimizer.param_groups[0]["lr"],
                self.global_step,
            )

            geometry_mse = torch.stack(geo_mses).mean()
            self.write_scalar("Validation/geometry_mse", geometry_mse)
            per_value = torch.stack(geo_per_value).mean(0)
            for name, value in zip(GEOMETRY_NAMES, per_value):
                self.write_scalar(f"Validation/geometry_{name}", value)

            self.checkpoint_if_best(lpips)
            self.visualize()

        self.model.train()

    def visualize(self):
        batch = self.maker.random_val_batch(8)
        n = min(8, batch["images"].shape[0])
        codepoints = batch["codepoints"][:n].to(self.device)
        evidence = batch["evidence"][:n].to(self.device)
        evidence_cps = batch["evidence_cps"][:n]
        instance_ids = batch["instance_ids"][:n].to(self.device)
        gts = batch["images"][:n].to(self.device).float()
        geometry = batch["geometry"][:n]

        with torch.no_grad(), self._autocast_context():
            context = self.model.encode_context(evidence)
            recs = self.model.sample(context, codepoints)
            pred_geometry = self.model.predict_geometry(codepoints, context, recs)
        recs = recs.float().clamp(0.0, 1.0)
        pred_geometry = pred_geometry.float()

        # The first evidence glyph (uniform frame) as a style reference for the
        # eye, plus a text line naming the instance, target and evidence set.
        refs = evidence[:, 0].float()  # (n, 1, H, W)
        text_lines = []
        for i in range(n):
            inst = self.maker.instances[instance_ids[i].item()]
            target_cp = self.maker.cp_list[codepoints[i].item()]
            ev_chars = "".join(
                chr(c) if 32 <= c < 0x110000 and chr(c).isprintable() else "?"
                for c in evidence_cps[i].tolist()
            )
            text_lines.append(
                f"{inst.family} [{inst.style} {inst.weight}] "
                f"| target {chr(target_cp)!r} U+{target_cp:04X} "
                f"| evidence {ev_chars!r}"
            )

        grid = torch.cat([gts, refs, recs], dim=0)
        self.writer.add_image(
            "Validation/gt_evidence_recon",
            torchvision.utils.make_grid(grid, nrow=n),
            self.global_step,
        )
        self.writer.add_text(
            "Validation/pairs", "\n".join(text_lines), self.global_step
        )

        self._geometry_montage(gts, recs, geometry, pred_geometry, n)

    def _geometry_montage(self, gts, recs, geometry, pred_geometry, n):
        """Denormalize each glyph with its geometry and place it on the baseline."""
        gts_np = gts.cpu().numpy()[:, 0]
        recs_np = recs.cpu().numpy()[:, 0]
        gt_geo = (
            geometry.numpy()
            if isinstance(geometry, torch.Tensor)
            else np.asarray(geometry)
        )
        pred_geo = pred_geometry.cpu().numpy()

        cells = []
        for i in range(n):
            cells.append(self._place_rgb(gts_np[i], gt_geo[i]))
        for i in range(n):
            cells.append(self._place_rgb(recs_np[i], pred_geo[i]))

        grid = torchvision.utils.make_grid(torch.stack(cells), nrow=n)
        self.writer.add_image("Validation/geometry_placed", grid, self.global_step)

    @staticmethod
    def _place_rgb(image: np.ndarray, geometry: np.ndarray) -> torch.Tensor:
        """Place a single glyph and overlay origin/baseline/advance markers."""
        canvas, origin_x, baseline_y, advance_x = place_glyph(image, geometry.tolist())
        rgb = np.repeat(canvas[..., None], 3, axis=2)

        thickness = 2
        rgb[:, origin_x : origin_x + thickness] = (0.0, 0.0, 1.0)
        rgb[baseline_y : baseline_y + thickness, :] = (0.0, 0.0, 1.0)
        rgb[:, advance_x : advance_x + thickness] = (0.0, 1.0, 0.0)

        return torch.from_numpy(rgb).float().permute(2, 0, 1)


def _parse_strata(spec: str) -> dict[str, float]:
    """Parse a ``'sans:0.25,serif:0.25,...'`` stratum-fraction spec."""
    out: dict[str, float] = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        name, frac = part.split(":")
        out[name.strip()] = float(frac)
    return out


# ══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    import argparse
    import sys

    DEFAULT_MODEL_PATH = "models/exemplar_diffusion.pth"

    parser = argparse.ArgumentParser(
        description="Train exemplar-conditional diffusion model"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=os.environ.get("GOOGLE_FONTS_REPO"),
        help="Path to the Google Fonts repository",
    )
    parser.add_argument("--tag", type=str, help="Tag for the training run")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow training with uncommitted changes",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument(
        "--character-set",
        type=str,
        default="LATIN_KERNEL",
        help="Name of a character set in hrothgar.dataset (default LATIN_KERNEL)",
    )
    parser.add_argument(
        "--extra-codepoints",
        type=str,
        default="",
        help="Extra codepoints to add to the vocabulary (a string of chars)",
    )
    parser.add_argument(
        "--remove-codepoints",
        type=str,
        default="",
        help="Codepoints to remove from the vocabulary (e.g. 'ga')",
    )
    parser.add_argument(
        "--oversample-codepoints",
        type=str,
        default="",
        help="Codepoints to oversample in training (e.g. '₹')",
    )
    parser.add_argument("--oversample-factor", type=int, default=4)
    parser.add_argument(
        "--num-instances",
        type=int,
        required=True,
        help="Number of font instances in the stratified subset (required)",
    )
    parser.add_argument(
        "--subset-strata",
        type=str,
        default=None,
        help="Stratum fractions, e.g. 'sans:0.25,serif:0.25,display:0.2,"
        "script:0.2,handwriting:0.1'",
    )
    parser.add_argument("--max-per-unit", type=int, default=3)
    parser.add_argument("--avg-instances", type=float, default=1.6)
    parser.add_argument("--target-frac", type=float, default=0.7)
    parser.add_argument(
        "--no-prefer-multi-target",
        dest="prefer_multi_target",
        action="store_false",
        help="Do not prefer families that can supply multiple ₹ instances",
    )
    parser.add_argument("--no-replacement", dest="replacement", action="store_false")
    parser.add_argument("--subset-seed", type=int, default=1234)
    parser.add_argument("--heldout-fraction", type=float, default=0.25)
    parser.add_argument("--min-train-fonts-per-codepoint", type=int, default=20)
    parser.add_argument("--split-seed", type=int, default=1234)
    parser.add_argument("--target-steps", type=int, default=600_000)
    parser.add_argument("--learning-rate", type=float, default=3.24e-5)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--geometry-weight", type=float, default=0.1)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--context-dim", type=int, default=256)
    parser.add_argument("--style-out-res", type=int, default=8)
    parser.add_argument("--style-encoder-base-channels", type=int, default=64)
    parser.add_argument("--num-evidence-glyphs", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sampling-timesteps", type=int, default=100)
    parser.add_argument("--min-snr-gamma", type=float, default=5.0)
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "bf16"],
        default="bf16",
        help="Training precision (bf16 = AMP, fp32 = no AMP)",
    )
    parser.add_argument("--validation-every", type=int, default=1000)
    parser.add_argument("--validation-batches", type=int, default=20)
    parser.add_argument(
        "--model-path", type=str, default=DEFAULT_MODEL_PATH
    )

    argv = sys.argv[1:]

    # Resolve --model-path first so we can read any previous run's sidecar and
    # use its recorded command-line arguments as defaults for this invocation.
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    probe_args, _ = probe.parse_known_args(argv)
    apply_cli_defaults(parser, load_cli_defaults(probe_args.model_path))

    args = parser.parse_args(argv)
    if not args.dataset_path:
        raise ValueError("GOOGLE_FONTS_REPO not set and --dataset-path not given")

    # Record the raw (pre-transform) command-line arguments so the training loop
    # can persist them to the sidecar and a later resume can reload them.
    args.cli_args = vars(args).copy()

    import hrothgar.dataset as ds

    args.character_set = getattr(ds, args.character_set, None)
    if args.character_set is None:
        raise ValueError(f"unknown character set: {args.character_set}")

    args.extra_codepoints = [ord(c) for c in args.extra_codepoints]
    args.remove_codepoints = [ord(c) for c in args.remove_codepoints]
    oversample_cps = [ord(c) for c in args.oversample_codepoints]
    args.oversample_codepoints = {cp: args.oversample_factor for cp in oversample_cps}
    args.subset_strata = _parse_strata(args.subset_strata)

    loop = ExemplarTrainingLoop(args)
    loop.train()
