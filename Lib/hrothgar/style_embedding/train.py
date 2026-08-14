"""Training loop for font style embedding.

Follows the project's standard ``TrainingLoop`` conventions (TensorBoard logging,
JSON sidecar, ``SaveLoadModel``, pkbar progress bars, etc.).
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from hrothgar.style_embedding.config import (
    FontStyleEmbedderConfig,
    FontStyleEmbeddingLossWeights,
)
from hrothgar.style_embedding.dataset import FontStyleDatasetMaker
from hrothgar.style_embedding.losses import compute_losses
from hrothgar.style_embedding.model import FontStyleEmbedder
from hrothgar.utils import TrainingLoop, progress_values

# ── Hyper-parameters ────────────────────────────────────────────────────────

LEARNING_RATE = 1e-4
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0


# ── Tag discovery ───────────────────────────────────────────────────────────


def _collect_all_tags(repo_url: str | Path) -> list[str]:
    """Scan the tag CSV and return every unique tag name."""
    tags: set[str] = set()
    tag_csv = Path(repo_url) / "tags" / "all" / "families.csv"
    if not tag_csv.exists():
        return []
    with tag_csv.open() as f:
        for line in csv.reader(f):
            if len(line) >= 3:
                tags.add(line[2])
    return sorted(tags)


def _filter_tags_by_prefix(tag_names: list[str], prefixes: str) -> list[str]:
    """Keep only tags whose path starts with one of the given prefixes.

    *prefixes* is a comma-separated string like ``"Expressive,Sans,Serif"``.
    Each prefix is matched against the last path component
    (e.g. ``/Expressive/Happy`` matches ``Expressive``).
    """
    keep = set(p.strip() for p in prefixes.split(",") if p.strip())
    if not keep:
        return tag_names
    result = []
    for name in tag_names:
        # Extract the category from the path: /Expressive/Happy → Expressive
        parts = name.strip("/").split("/")
        if len(parts) >= 2 and parts[0] in keep:
            result.append(name)
        elif len(parts) == 1 and parts[0] in keep:
            result.append(name)
    return result


def _layout_loss(
    model: FontStyleEmbedder,
    style: torch.Tensor,
    glyph_mask: torch.Tensor,
    glyph_bboxes: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """Predict ``num_samples`` hidden glyphs' ink bounding boxes per font.

    Args:
        model: the style embedder (provides ``predict_layout``).
        style: ``(B, F)`` summary vectors for one contrastive view.
        glyph_mask: ``(B, G)`` boolean, ``True`` = visible.
        glyph_bboxes: ``(B, G, 4)`` normalized ``(x0, y0, x1, y1)`` targets.
        num_samples: number of hidden glyphs to predict per font.
    """
    b = style.shape[0]
    hidden = ~glyph_mask  # (B, G)
    # Uniform over hidden slots (+ epsilon so multinomial never sees all-zeros).
    probs = hidden.float() + 1e-8
    probs = probs / probs.sum(dim=1, keepdim=True)
    sampled = torch.multinomial(probs, num_samples=num_samples, replacement=True)

    font_idx = (
        torch.arange(b, device=style.device)
        .unsqueeze(1)
        .expand(b, num_samples)
        .reshape(-1)
    )
    slots = sampled.reshape(-1)
    pred = model.predict_layout(style[font_idx], slots)
    targets = glyph_bboxes[font_idx, slots]
    # Compute in float32 so the loss is dtype-consistent under mixed precision.
    return F.l1_loss(pred.float(), targets.float())


# ── Training loop ───────────────────────────────────────────────────────────


class FontStyleEmbeddingTrainingLoop(TrainingLoop):
    """Training loop for the font-level style embedder."""

    def post_init(self, train_args) -> None:
        # ── Config & model ──────────────────────────────────────────────
        tag_names: Optional[list[str]] = None
        if train_args.tags:
            if train_args.tags.lower() == "all":
                tag_names = _collect_all_tags(train_args.dataset_path)
                print(f"Collected {len(tag_names)} unique tags")
            else:
                tag_names = [t.strip() for t in train_args.tags.split(",")]

            # Filter to visually-grounded categories if requested.
            tag_filter = getattr(train_args, "tag_filter", None)
            if tag_filter and tag_names:
                tag_names = _filter_tags_by_prefix(tag_names, tag_filter)
                print(f"After filtering by [{tag_filter}]: {len(tag_names)} tags")

        text_encoder_name = getattr(train_args, "text_encoder", None) or ""
        multipos_family = getattr(train_args, "multipos_family_positives", True)
        config = FontStyleEmbedderConfig(
            glyph_size=getattr(train_args, "glyph_size", 64),
            per_glyph_tokens=getattr(train_args, "per_glyph_tokens", 4),
            style_latents=getattr(train_args, "style_latents", 16),
            encoder_downsample=getattr(train_args, "encoder_downsample", 4),
            aggregation_heads=getattr(train_args, "aggregation_heads", 8),
            layout_samples=getattr(train_args, "layout_samples", 4),
            tag_names=tag_names or [],
            contrastive_temperature=train_args.contrastive_temperature,
            tag_num_classes=train_args.tag_num_classes,
            tag_dropout=train_args.tag_dropout,
            use_category_head=train_args.use_category_head,
            text_encoder_name=text_encoder_name,
            multipos_use_family_positives=multipos_family,
        )
        if getattr(train_args, "input_glyphs", None):
            config.input_codepoints = [
                ord(c) for c in train_args.input_glyphs
            ]
        model = FontStyleEmbedder(config).to(self.device)
        config.save_sidecar(train_args.model_path)
        print(
            f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}"
        )

        # ── Data ────────────────────────────────────────────────────────
        maker = FontStyleDatasetMaker(
            repo_url=str(train_args.dataset_path),
            batch_size=train_args.batch_size,
            glyph_size=config.glyph_size,
            input_codepoints=config.input_codepoints,
            mask_probability=getattr(train_args, "mask_probability", 0.5),
            canary_size=train_args.limit_dataset_size,
            tag_names=tag_names or [],
            tag_num_classes=config.tag_num_classes,
            text_encoder_name=config.text_encoder_name or None,
            text_embedding_dim=config.text_embedding_dim,
            class_balanced=not getattr(train_args, "no_class_balance", False),
        )
        self.train_loader = maker.train_loader()
        self.test_loader = maker.test_loader()
        print(
            f"Train batches: {len(self.train_loader)}, "
            f"Test batches: {len(self.test_loader)}"
        )

        # ── Optimiser ───────────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            betas=(ADAM_BETA1, ADAM_BETA2),
            weight_decay=WEIGHT_DECAY,
        )

        # ── Loss ────────────────────────────────────────────────────────
        self.loss_weights = FontStyleEmbeddingLossWeights(
            use_family_positives=config.multipos_use_family_positives,
            tag_positive_weight=getattr(train_args, "tag_positive_weight", 1.0),
            layout=getattr(train_args, "layout_weight", 1.0),
        )

        # ── Bookkeeping ─────────────────────────────────────────────────
        self.model = model
        self.target_steps = train_args.target_steps
        self.num_epochs = getattr(train_args, "num_epochs", 100)
        self.validation_direction = "lower"
        self.validation_every = train_args.validation_every
        self.validation_batches = train_args.validation_batches

        # ── AMP ─────────────────────────────────────────────────────────
        self.use_amp = train_args.precision in {"bf16", "fp16"}
        if train_args.precision == "bf16":
            self.amp_dtype = torch.bfloat16
        elif train_args.precision == "fp16":
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = None

        if self.use_amp and self.device.type != "cuda":
            raise ValueError(
                f"precision={train_args.precision} requires CUDA, "
                f"got device {self.device}"
            )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp and self.amp_dtype == torch.float16,
        )

    # ------------------------------------------------------------------
    # AMP helper
    # ------------------------------------------------------------------

    def _autocast_context(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    # ------------------------------------------------------------------
    # Training (with gradient clipping)
    # ------------------------------------------------------------------

    def train(self):
        if len(self.train_loader) == 0:
            raise ValueError("Training loader is empty; cannot start training.")
        import pkbar

        try:
            while not self.must_stop():
                kbar = pkbar.Kbar(
                    target=len(self.train_loader),
                    epoch=self.epoch,
                    num_epochs=self.num_epochs,
                )
                self.model.train()
                for i, batch in enumerate(self.train_loader):
                    if self.must_stop():
                        break
                    self.optimizer.zero_grad(set_to_none=True)
                    loss, loss_info = self.train_step(batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), GRAD_CLIP_NORM
                    )
                    self.optimizer.step()

                    self.global_step += 1
                    kbar.update(i, values=progress_values(loss_info))
                    for key, value in loss_info.items():
                        self.write_scalar("Losses/" + key, value)
                    if self.global_step % 100 == 0:
                        self.writer.flush()
                    self.post_train_step()
                self.post_train_epoch()
                self.epoch += 1
                self.validation()
        finally:
            self.writer.close()

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, batch):
        # Images are already (2B, G, 1, H, W) — view 1 stacked on view 2.
        images = batch["images"].to(self.device)
        glyph_mask = batch["glyph_mask"].to(self.device)
        glyph_bboxes = batch["glyph_bboxes"].to(self.device)
        target_tags = {k: v.to(self.device) for k, v in batch["tags"].items()}
        tag_masks = {k: v.to(self.device) for k, v in batch.get("tag_masks", {}).items()}
        family = batch.get("family", None)
        category = batch.get("category")
        tag_vectors = batch.get("tag_vectors")
        if tag_vectors is not None:
            tag_vectors = tag_vectors.to(self.device)
        text_embeddings = batch.get("text_embeddings")
        if text_embeddings is not None:
            text_embeddings = text_embeddings.to(self.device)
            text_embeddings = self.model.project_text(text_embeddings)

        target_tags_2x = {k: torch.cat([v, v], dim=0) for k, v in target_tags.items()}  # type: ignore[arg-type]
        tag_masks_2x = {k: torch.cat([v, v], dim=0) for k, v in tag_masks.items()}  # type: ignore[arg-type]
        family_2x = family + family if family else None
        category_2x = torch.cat([category, category], dim=0).to(self.device) if category is not None else None

        with self._autocast_context():
            _embedding, projection, predicted_tags, category_logits = self.model(
                images, glyph_mask=glyph_mask
            )

            loss, loss_info = compute_losses(
                projections=projection,
                predicted_tags=predicted_tags,
                target_tags=target_tags_2x,
                tag_masks=tag_masks_2x,
                weights=self.loss_weights,
                temperature=self.model.config.contrastive_temperature,
                family_labels=family_2x,
                category_logits=category_logits,
                category_targets=category_2x,
                text_embeddings=text_embeddings,
                tag_vectors=tag_vectors,
            )

            if self.model.config.use_layout:
                b = glyph_mask.shape[0] // 2
                k = self.model.config.layout_samples
                layout_loss = 0.5 * (
                    _layout_loss(
                        self.model, _embedding[:b], glyph_mask[:b], glyph_bboxes, k
                    )
                    + _layout_loss(
                        self.model, _embedding[b:], glyph_mask[b:], glyph_bboxes, k
                    )
                )
                loss = loss + self.loss_weights.layout * layout_loss
                loss_info["layout"] = layout_loss.detach()
                loss_info["layout_weighted"] = (
                    self.loss_weights.layout * layout_loss
                ).detach()

        return loss, loss_info

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def post_train_step(self):
        if self.global_step % self.validation_every != 0:
            return

        self.model.eval()

        val_contrastive = []
        val_tag_pred = []
        val_cat_acc = []
        # Per-tag metric accumulators: {tag_name: [values across batches]}.
        val_tag_metrics: dict[str, list[torch.Tensor]] = {}

        # Retrieval metrics — temperature-independent, so comparable across runs.
        retrieval_top1: list[torch.Tensor] = []
        retrieval_recall3: list[torch.Tensor] = []
        retrieval_recall5: list[torch.Tensor] = []

        # Layout (bounding-box) loss on held-out fonts.
        val_layout: list[torch.Tensor] = []

        with torch.no_grad():
            for val_batch in itertools.islice(
                self.test_loader, self.validation_batches
            ):
                images = val_batch["images"].to(self.device)
                glyph_mask = val_batch["glyph_mask"].to(self.device)
                glyph_bboxes = val_batch["glyph_bboxes"].to(self.device)
                target_tags = {
                    k: v.to(self.device)
                    for k, v in val_batch["tags"].items()
                }
                tag_masks = {
                    k: v.to(self.device)
                    for k, v in val_batch.get("tag_masks", {}).items()
                }
                family = val_batch.get("family", None)
                category = val_batch.get("category")
                tag_vectors = val_batch.get("tag_vectors")
                if tag_vectors is not None:
                    tag_vectors = tag_vectors.to(self.device)
                text_embeddings = val_batch.get("text_embeddings")
                if text_embeddings is not None:
                    text_embeddings = text_embeddings.to(self.device)
                    text_embeddings = self.model.project_text(text_embeddings)

                target_tags_2x = {
                    k: torch.cat([v, v], dim=0)  # type: ignore[arg-type]
                    for k, v in target_tags.items()
                }
                tag_masks_2x = {
                    k: torch.cat([v, v], dim=0)  # type: ignore[arg-type]
                    for k, v in tag_masks.items()
                }
                family_2x = family + family if family else None
                category_2x = torch.cat([category, category], dim=0).to(self.device) if category is not None else None

                with self._autocast_context():
                    _emb, projection, predicted_tags, category_logits = self.model(
                        images, glyph_mask=glyph_mask
                    )
                    _loss, loss_info = compute_losses(
                        projections=projection,
                        predicted_tags=predicted_tags,
                        target_tags=target_tags_2x,
                        tag_masks=tag_masks_2x,
                        weights=self.loss_weights,
                        temperature=self.model.config.contrastive_temperature,
                        family_labels=family_2x,
                        category_logits=category_logits,
                        category_targets=category_2x,
                        text_embeddings=text_embeddings,
                        tag_vectors=tag_vectors,
                    )

                    if self.model.config.use_layout:
                        b = glyph_mask.shape[0] // 2
                        k = self.model.config.layout_samples
                        layout_loss = 0.5 * (
                            _layout_loss(
                                self.model, _emb[:b], glyph_mask[:b], glyph_bboxes, k
                            )
                            + _layout_loss(
                                self.model, _emb[b:], glyph_mask[b:], glyph_bboxes, k
                            )
                        )
                        val_layout.append(layout_loss)

                val_contrastive.append(loss_info["contrastive"])
                if "tag_prediction" in loss_info:
                    val_tag_pred.append(loss_info["tag_prediction"])
                if "category_accuracy" in loss_info:
                    val_cat_acc.append(loss_info["category_accuracy"])
                # Accumulate per-tag metrics.
                for key, value in loss_info.items():
                    if key.startswith("tag_acc_") or key.startswith("tag_mse_"):
                        val_tag_metrics.setdefault(key, []).append(value)

                # Retrieval accuracy: for each view-1 embedding, is its
                # view-2 counterpart the nearest neighbour?  Uses raw
                # cosine similarity (no temperature) so it is comparable
                # across runs with different loss hyper-parameters.
                B = projection.shape[0] // 2
                v1 = projection[:B]   # (B, D)
                v2 = projection[B:]   # (B, D)
                sim = torch.matmul(v1, v2.T)   # (B, B) cosine similarity
                # Rank view-2 items by similarity (descending).
                ranks = sim.argsort(dim=-1, descending=True)  # (B, B)
                # For query i, correct match is at column i.
                label_col = torch.arange(B, device=ranks.device).unsqueeze(1)  # (B, 1)
                correct_rank = (ranks == label_col).nonzero(as_tuple=True)[1]  # (B,)
                retrieval_top1.append((correct_rank < 1).float().mean())
                retrieval_recall3.append((correct_rank < 3).float().mean())
                retrieval_recall5.append((correct_rank < 5).float().mean())

        avg_contrastive = torch.mean(torch.stack(val_contrastive))
        self.write_scalar("Validation/contrastive", avg_contrastive)

        # Log retrieval metrics (temperature-independent).
        if retrieval_top1:
            self.write_scalar(
                "Validation/retrieval_top1",
                torch.mean(torch.stack(retrieval_top1)),
            )
            self.write_scalar(
                "Validation/retrieval_recall3",
                torch.mean(torch.stack(retrieval_recall3)),
            )
            self.write_scalar(
                "Validation/retrieval_recall5",
                torch.mean(torch.stack(retrieval_recall5)),
            )

        if val_tag_pred:
            avg_tag = torch.mean(torch.stack(val_tag_pred))
            self.write_scalar("Validation/tag_prediction", avg_tag)

        if val_cat_acc:
            avg_cat_acc = torch.mean(torch.stack(val_cat_acc))
            self.write_scalar("Validation/category_accuracy", avg_cat_acc)

        if val_layout:
            avg_layout = torch.mean(torch.stack(val_layout))
            self.write_scalar("Validation/layout", avg_layout)

        # Log per-tag metrics and write a sorted summary to TensorBoard.
        if val_tag_metrics:
            summary_lines = []
            for key in sorted(val_tag_metrics.keys()):
                avg = torch.mean(torch.stack(val_tag_metrics[key]))
                self.write_scalar(f"Validation/{key}", avg)
                is_acc = key.startswith("tag_acc_")
                tag_name = key[len("tag_acc_" if is_acc else "tag_mse_"):]
                summary_lines.append((float(avg), tag_name, "acc" if is_acc else "mse"))
            reverse = any("tag_acc_" in k for k in val_tag_metrics.keys())
            sorted_tags = sorted(summary_lines, key=lambda x: x[0], reverse=reverse)
            lines = [f"Per-tag {'accuracy' if reverse else 'MSE'} (step {self.global_step})\n"]
            lines.append(f"{'Best':>6} {'':40} | {'Worst':>6}\n")
            lines.append("-" * 100 + "\n")
            for (v1, n1, _), (v2, n2, _) in zip(sorted_tags[:15], sorted_tags[-15:]):
                lines.append(f"{n1:<45} {v1:6.3f} | {n2:<45} {v2:6.3f}\n")
            self.writer.add_text("Validation/per_tag_metrics", "".join(lines), self.global_step)

        # Checkpoint on retrieval accuracy — temperature-independent so
        # comparable across runs with different hyper-parameters.
        if retrieval_top1:
            self.checkpoint_if_best(-torch.mean(torch.stack(retrieval_top1)))
        else:
            self.checkpoint_if_best(avg_contrastive)

        # Visualize layout (always, when enabled) and tags (when active).
        if self.model.config.use_layout:
            self.visualize_layout()
        if self.model.config.tag_names:
            self.visualize()

        self.model.train()

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize_layout(self):
        """Overlay ground-truth vs predicted glyph bounding boxes."""
        if not self.model.config.use_layout:
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        val_batch = next(iter(self.test_loader))
        images = val_batch["images"].to(self.device)
        glyph_mask = val_batch["glyph_mask"].to(self.device)
        target_glyphs = val_batch["target_glyphs"].to(self.device)
        glyph_bboxes = val_batch["glyph_bboxes"].to(self.device)

        b = glyph_mask.shape[0] // 2
        mask_v1 = glyph_mask[:b]

        with torch.no_grad():
            with self._autocast_context():
                embedding, _proj, _tags, _cat = self.model(
                    images, glyph_mask=glyph_mask
                )
        style_v1 = embedding[:b]

        n_fonts = min(2, b)
        k = 6
        fig, axes = plt.subplots(n_fonts, k, figsize=(k * 1.1, n_fonts * 1.1))
        if n_fonts == 1:
            axes = axes[None, :]

        for i in range(n_fonts):
            hidden = (~mask_v1[i]).nonzero(as_tuple=False).squeeze(-1)
            if hidden.numel() < k:
                continue
            slots = hidden[:k]
            style = style_v1[i : i + 1].expand(k, -1)
            with torch.no_grad():
                with self._autocast_context():
                    pred = self.model.predict_layout(style, slots)

            for j, s in enumerate(slots):
                glyph = target_glyphs[i, s, 0].cpu().numpy()  # (H, W)
                h, w = glyph.shape
                ax = axes[i, j]
                ax.imshow(glyph, cmap="gray", vmin=0.0, vmax=1.0)
                ax.set_xticks([])
                ax.set_yticks([])

                scale = np.array([w, h, w, h], dtype=np.float32)
                gt = glyph_bboxes[i, s].float().cpu().numpy() * scale
                pr = pred[j].float().cpu().numpy() * scale
                ax.add_patch(Rectangle(
                    (gt[0], gt[1]), gt[2] - gt[0], gt[3] - gt[1],
                    fill=False, edgecolor="lime", linewidth=1.0,
                ))
                ax.add_patch(Rectangle(
                    (pr[0], pr[1]), pr[2] - pr[0], pr[3] - pr[1],
                    fill=False, edgecolor="red", linewidth=1.0,
                ))

        for ax_row in axes:
            for ax in ax_row:
                ax.axis("off")

        fig.suptitle(
            f"GT bbox = green, pred bbox = red (step {self.global_step})", fontsize=8
        )
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, facecolor="white")
        buf.seek(0)
        img = plt.imread(buf)
        buf.close()
        plt.close(fig)

        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        self.writer.add_image("Validation/layout", img_tensor, self.global_step)

    def visualize(self):
        """Render tag prediction comparison charts for a few validation fonts."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        val_batch = next(iter(self.test_loader))
        images = val_batch["images"].to(self.device)
        glyph_mask = val_batch["glyph_mask"].to(self.device)
        B = images.shape[0] // 2
        images_v1 = images[:B]
        glyph_mask_v1 = glyph_mask[:B]
        target_tags = {k: v.to(self.device) for k, v in val_batch["tags"].items()}
        tag_masks = {k: v.to(self.device) for k, v in val_batch.get("tag_masks", {}).items()}
        font_names = val_batch.get("family", [f"font_{i}" for i in range(B)])

        with torch.no_grad():
            with self._autocast_context():
                _emb, _proj, predicted_tags, category_logits = self.model(
                    images_v1, glyph_mask=glyph_mask_v1
                )

        # Also log the glyphs so we can see what the model sees.
        if images_v1.shape[0] > 0:
            n = min(8, images_v1.shape[1])
            strip = images_v1[0][:n].squeeze(1).permute(1, 0, 2)
            strip = strip.reshape(1, strip.shape[0], strip.shape[1] * strip.shape[2])
            self.writer.add_image(
                "Validation/glyph_sample", strip.clamp(0, 1), self.global_step
            )


        tag_names = self.model.config.tag_names
        if not tag_names or predicted_tags is None:
            return

        nc = self.model.config.tag_num_classes
        n_fonts = min(3, images_v1.shape[0])
        n_tags = len(tag_names)

        fig, axes = plt.subplots(
            n_fonts, 1,
            figsize=(max(8, n_tags * 0.4), 3 * n_fonts),
            squeeze=False,
        )
        mode_str = f"{nc}-class" if nc > 0 else "regression"
        fig.suptitle(
            f"Tag predictions — {mode_str} (step {self.global_step})\n"
            "blue = GT, orange = pred, grey = tag absent",
            fontsize=10,
        )

        x = np.arange(n_tags)
        width = 0.35

        for font_i in range(n_fonts):
            ax = axes[font_i, 0]
            gt_vals = []
            pred_vals = []
            present = []
            for tag_i, name in enumerate(tag_names):
                present_i = bool(
                    tag_masks.get(name, torch.ones(1, dtype=torch.bool))[font_i].item()
                )
                present.append(present_i)

                if nc > 0:
                    # Classification: GT is class index, pred is softmax'd class.
                    gt_vals.append(float(target_tags[name][font_i].cpu()))
                    logits = predicted_tags[name][font_i]  # (nc,)
                    pred_vals.append(float(logits.softmax(dim=-1).argmax().cpu()))
                else:
                    # Regression: both in [0, 1].
                    gt_vals.append(float(target_tags[name][font_i].cpu()))
                    pred_vals.append(float(predicted_tags[name][font_i].cpu()))

            gt_colors = ["#2196F3" if p else "#BDBDBD" for p in present]
            pred_colors = ["#FF9800" if p else "#BDBDBD" for p in present]

            ax.bar(x - width / 2, gt_vals, width, color=gt_colors, label="GT")
            ax.bar(x + width / 2, pred_vals, width, color=pred_colors, label="Pred")
            ax.set_ylabel(font_names[font_i], fontsize=8)
            ax.set_ylim(0, max(1, nc - 1))
            ax.set_xticks(x)
            ax.set_xticklabels(
                [n.replace("/Expressive/", "").replace("/Sans/", "").replace("/Serif/", "")
                 for n in tag_names],
                rotation=45, ha="right", fontsize=6,
            )
            if font_i == 0:
                ax.legend(fontsize=7, loc="upper right")

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, facecolor="white")
        buf.seek(0)
        img = plt.imread(buf)
        buf.close()
        plt.close(fig)

        # Convert HWC → CHW for TensorBoard.
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        self.writer.add_image("Validation/tag_predictions", img_tensor, self.global_step)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train font style embedder")
    p.add_argument("--dataset-path", type=Path, required=True,
                   help="Path to Google Fonts checkout")
    p.add_argument("--model-path", type=str, required=True,
                   help="Path to save model checkpoint")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--target-steps", type=int, default=None,
                   help="Maximum training steps (None = run all epochs)")
    p.add_argument("--validation-every", type=int, default=1000,
                   help="Run validation every N training steps")
    p.add_argument("--validation-batches", type=int, default=10,
                   help="Number of test batches per validation run")
    p.add_argument("--contrastive-temperature", type=float, default=0.07)
    p.add_argument("--glyph-size", type=int, default=64,
                   help="Rendered glyph size (square).")
    p.add_argument("--input-glyphs", type=str, default=None,
                   help="Override the glyph set (e.g. 'aAbB012&?').")
    p.add_argument("--per-glyph-tokens", type=int, default=4,
                   help="Tokens per glyph after spatial attention pooling.")
    p.add_argument("--style-latents", type=int, default=16,
                   help="Number of global style tokens after cross-glyph pooling.")
    p.add_argument("--encoder-downsample", type=int, default=4,
                   choices=[2, 4, 8],
                   help="Spatial downsample ratio of the per-glyph encoder.")
    p.add_argument("--aggregation-heads", type=int, default=8,
                   help="Attention heads in the cross-glyph aggregator.")
    p.add_argument("--mask-probability", type=float, default=0.5,
                   help="Probability a glyph is masked out of a contrastive view.")
    p.add_argument("--layout-samples", type=int, default=4,
                   help="Hidden glyphs whose bounding box is predicted per view.")
    p.add_argument("--layout-weight", type=float, default=1.0,
                   help="Weight of the glyph-layout (bounding-box) loss.")
    p.add_argument("--tags", type=str, default=None,
                   help="Comma-separated tag names to predict, or 'all'")
    p.add_argument("--tag-filter", type=str, default=None,
                   help="When --tags=all, only keep tags under these prefixes "
                        "(e.g. 'Expressive,Sans,Serif').  Filters /Quality/* "
                        "and other unlearnable-from-glyphs tags.")
    p.add_argument("--tag-num-classes", type=int, default=0,
                   help="Quantize tag targets into N classes (0=regression, "
                        "2=binary, 4=quartiles)")
    p.add_argument("--tag-dropout", type=float, default=0.3,
                   help="Dropout rate in tag prediction heads")
    p.add_argument("--use-category-head", action="store_true",
                   help="Add a 6-way broad category classification head "
                        "(Serif/Sans/Handwriting/Script/Monospace/Display)")
    p.add_argument("--text-encoder", type=str, default=None,
                   help="HuggingFace model for frozen text conditioning "
                        "(e.g. 'sentence-transformers/all-MiniLM-L6-v2').  "
                        "Enables multi-positive contrastive loss.")
    p.add_argument("--no-family-positives", action="store_false",
                   dest="multipos_family_positives",
                   help="Disable same-family image positives in multi-positive "
                        "loss.  Only same-font other-view + text are positives.  "
                        "Use this if contrastive loss overfits.")
    p.add_argument("--no-class-balance", action="store_true",
                   help="Disable class-balanced batch sampling.  Without this, "
                        "each batch is guaranteed to contain fonts from all 6 "
                        "categories, creating only trivially-easy negatives.  "
                        "Disabling it allows harder within-category contrasts "
                        "(e.g. Garamond vs Caslon) at the cost of potential "
                        "category imbalance.")
    p.add_argument("--tag-positive-weight", type=float, default=1.0,
                   help="Weight of tag-based soft positives in contrastive loss "
                        "(0.0 = pure visual, 1.0 = full tag signal).  "
                        "Use small values (e.g. 0.1) for gentle style nudges "
                        "on top of a converged visual model.")
    p.add_argument("--tag", type=str, default=None,
                   help="Optional human-readable tag for the TensorBoard run")
    p.add_argument("--precision", type=str, default="fp32",
                   choices=["fp32", "fp16", "bf16"],
                   help="Training precision")
    p.add_argument("--limit-dataset-size", type=int, default=None,
                   help="Limit number of fonts for debugging")
    p.add_argument("--allow-dirty", action="store_true",
                   help="Allow training from a dirty git checkout")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    loop = FontStyleEmbeddingTrainingLoop(args)
    loop.train()


if __name__ == "__main__":
    main()
