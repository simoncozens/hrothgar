"""Dataset utilities for AR phase-1 visual pretraining.

This module reuses the shared font split logic from ``hrothgar.dataset`` and
provides a collation function tailored to the AR generator inputs.

Style is provided as pre-computed font-level embedding vectors from
``FontStyleEmbedder``.  Embeddings are computed once per font during
dataset initialisation and cached in memory.
"""

from __future__ import annotations

import random
import math
from typing import Optional, Sequence, Set

import torch
import tqdm
from torch.utils.data import BatchSampler, DataLoader

import uharfbuzz as hb

from hrothgar.ar.style_sampling import (
    _font_has_codepoint,
    _has_non_empty_glyph,
    _is_blank_rendering,
    _sample_style_codepoints,
)
from hrothgar.dataset import Dataset, DatasetMaker
from hrothgar.dataset_constants import LATIN_CORE
from hrothgar.glyph_rendering import bbox_size, crop_to_ink
from hrothgar.googlefonts import GoogleFont
from hrothgar.style_embedding.model import FontStyleEmbedder


class _OversampledTargetDataset(Dataset):
    """Dataset that duplicates configured target codepoints in item order."""

    def __init__(
        self,
        fonts,
        *,
        codepoint_filter_fn,
        oversampled_codepoints: Optional[Set[int]] = None,
        oversample_factor: int = 1,
    ):
        super().__init__(fonts, codepoint_filter_fn=codepoint_filter_fn)
        if not oversampled_codepoints or oversample_factor <= 1:
            return

        oversampled_items = [
            item for item in self.order if item[1] in oversampled_codepoints
        ]
        if not oversampled_items:
            return

        self.order.extend(oversampled_items * (oversample_factor - 1))


class ARPhase1DatasetMaker(DatasetMaker):
    """Dataset maker for AR phase-1 visual pretraining.

    Pre-computes a font-level style embedding for every font so that
    collation only requires a dictionary lookup — no rendering or CNN
    forward passes happen inside the training loop.
    """

    def __init__(
        self,
        repo_url: str,
        batch_size: int,
        *,
        font_style_embedder: FontStyleEmbedder,
        embedder_device: torch.device,
        having: Optional[Set[int]] = None,
        target_codepoints: Optional[Sequence[int]] = None,
        canary_size: Optional[int] = None,
        image_size: int = 128,
        style_glyph_count: int = 8,
        common_style_codepoints: Optional[Sequence[int]] = None,
        class_balanced: bool = False,
        split_seed: int = 1234,
        target_codepoint_oversample_factor: int = 8,
        target_only: bool = False,
    ) -> None:
        target_codepoint_set = set(target_codepoints) if target_codepoints else None
        if common_style_codepoints and style_glyph_count < len(common_style_codepoints):
            style_glyph_count = len(common_style_codepoints)
        if style_glyph_count <= 0:
            raise ValueError(
                f"style_glyph_count must be positive, got {style_glyph_count}"
            )
        super().__init__(
            repo_url=repo_url,
            batch_size=batch_size,
            having=having,
            target_codepoints=None,
            canary_size=canary_size,
            image_size=image_size,
            split_seed=split_seed,
        )
        if target_codepoint_oversample_factor <= 0:
            raise ValueError(
                "target_codepoint_oversample_factor must be positive, got "
                f"{target_codepoint_oversample_factor}"
            )
        self.class_balanced = class_balanced
        self.extra_target_codepoints = target_codepoint_set
        self.target_codepoint_oversample_factor = target_codepoint_oversample_factor
        self.target_only = target_only
        self.style_glyph_count = style_glyph_count
        self.common_style_codepoints = common_style_codepoints

        # Pre-compute font-level style embeddings for all fonts.
        self._font_embedding: dict[str, torch.Tensor] = {}
        self._precompute_font_embeddings(font_style_embedder, embedder_device)

    def _precompute_font_embeddings(
        self, embedder: FontStyleEmbedder, device: torch.device
    ) -> None:
        """Encode each font's full input glyph set into a style vector."""
        all_fonts = self.train_fonts + self.test_fonts
        if not all_fonts:
            return

        embedder.eval()
        embedder.to(device)
        print("Precomputing font-level style embeddings for all fonts...")
        for font in tqdm.tqdm(all_fonts):
            try:
                embedding = embedder.compute_embedding(font, device=device)
            except Exception as exc:
                print(
                    f"WARNING: Failed to render embedding for "
                    f"{font.family!r} ({font.path}): {exc}"
                )
                continue
            self._font_embedding[font.path] = embedding

        # If any font couldn't be rendered, use the mean of available
        # embeddings as a fallback so training doesn't crash.
        if self._font_embedding:
            fallback = torch.stack(list(self._font_embedding.values())).mean(dim=0)
            for font in all_fonts:
                if font.path not in self._font_embedding:
                    self._font_embedding[font.path] = fallback

    def filter_fonts(self):
        self.googlefonts.fonts = [
            font
            for font in self.googlefonts.fonts
            if not (
                "SC" in font.family  # Drop small caps families
                or "bitcount" in font.family.lower()
                or "playwrite" in font.family.lower()
                # Drop barcode/redaction/etc.
                or any("Special use" in k for k in font.tags().keys())
            )
        ]

    def train_codepoint_filter(self, font_codepoints: Set[int]) -> Set[int]:
        if self.target_only:
            return set(font_codepoints) & (self.extra_target_codepoints or set())
        chars = super().train_codepoint_filter(font_codepoints)
        chars = set(chars) & set(LATIN_CORE)
        if self.extra_target_codepoints is None:
            return chars
        return set(chars) | (set(font_codepoints) & self.extra_target_codepoints)

    def test_codepoint_filter(self, font_codepoints: Set[int]) -> Set[int]:
        if self.target_only:
            return set(font_codepoints) & (self.extra_target_codepoints or set())
        chars = super().test_codepoint_filter(font_codepoints)
        chars = set(chars) & set(LATIN_CORE)
        if self.extra_target_codepoints is None:
            return chars
        return set(chars) | (set(font_codepoints) & self.extra_target_codepoints)

    def train_set(self):
        return _OversampledTargetDataset(
            self.train_fonts,
            codepoint_filter_fn=self.train_codepoint_filter,
            oversampled_codepoints=self.extra_target_codepoints,
            oversample_factor=self.target_codepoint_oversample_factor,
        )

    def test_set(self):
        return Dataset(self.test_fonts, codepoint_filter_fn=self.test_codepoint_filter)

    def train_loader(self):
        dataset = self.train_set()
        if self.class_balanced:
            return DataLoader(
                dataset,
                batch_sampler=_ClassBalancedBatchSampler(
                    dataset.order,
                    batch_size=self.batch_size,
                    drop_last=True,
                ),
                num_workers=12,
                pin_memory=True,
                collate_fn=self.collate_fn,
            )
        return super().train_loader()

    def collate_fn(self, batch):
        """Collate samples for AR visual pretraining.

        Returns:
        - ``target_rendering``: target font glyph image (ground truth)
        - ``content_rendering``: same codepoint rendered by the reference font
        - ``style_renderings``: style support set from the target font
        - ``font_style_embedding``: pre-computed font-level style vector
        Plus metadata useful for debugging and future adaptation wiring.
        """
        chars = torch.tensor([item["char"] for item in batch], dtype=torch.long)
        target_renderings = []
        content_renderings = []
        style_renderings = []
        style_chars: list[list[int]] = []
        descriptions = []
        all_metrics: list[torch.Tensor] = []
        bbox_sizes: list[torch.Tensor] = []
        font_embeddings: list[torch.Tensor] = []
        font_embedding_fallback = torch.zeros(1)  # will be overwritten

        for item in batch:
            font: GoogleFont = item["font"]
            char = item["char"]
            reference_font = font.reference_font() or font
            upem = float(font.hb_face.upem)

            axis_pos = font.random_axis_position()

            def render_with_font(char):
                return font.render(char, size=self.image_size, axis_position=axis_pos)

            target_rendering = render_with_font(char)  # numpy (3, size, size)
            target_tensor = torch.tensor(target_rendering, dtype=torch.float32)
            bbox_sizes.append(bbox_size(target_tensor, self.image_size))
            target_renderings.append(crop_to_ink(target_tensor, self.image_size))

            # Content rendering fallback.
            if not _font_has_codepoint(
                reference_font, char
            ) or not _has_non_empty_glyph(reference_font, char):
                reference_font = font

            content_render = reference_font.render(char, size=self.image_size)
            if _is_blank_rendering(content_render):
                content_render = render_with_font(char)
            content_renderings.append(
                crop_to_ink(
                    torch.tensor(content_render, dtype=torch.float32),
                    self.image_size,
                )
            )

            sampled_style_chars = _sample_style_codepoints(
                font=font,
                target_char=char,
                style_glyph_count=self.style_glyph_count,
                common_style_codepoints=self.common_style_codepoints,
            )
            rendered_styles = []
            sanitized_style_chars: list[int] = []
            for cp in sampled_style_chars:
                style_render = render_with_font(cp)
                if _is_blank_rendering(style_render):
                    cp = char
                    style_render = render_with_font(cp)
                sanitized_style_chars.append(cp)
                rendered_styles.append(
                    crop_to_ink(
                        torch.tensor(style_render, dtype=torch.float32),
                        self.image_size,
                    )
                )
            style_chars.append(sanitized_style_chars)
            style_renderings.append(torch.stack(rendered_styles))

            descriptions.append(font.description_with_tags_and_display())

            # Font-level metrics (advance width kept for conditioning).
            vm = font.vertical_metrics()
            gid_for_advance = hb.Font(font.hb_face).get_nominal_glyph(char)
            aw = font.advance_width(gid_for_advance) / upem if upem > 0 else 0.0
            all_metrics.append(torch.tensor([
                float(vm["ascender"]) / upem if upem > 0 else 0.0,
                float(vm["descender"]) / upem if upem > 0 else 0.0,
                float(vm["x_height"]) / upem if upem > 0 else 0.0,
                float(vm["cap_height"]) / upem if upem > 0 else 0.0,
                float(vm["baseline"]) / upem if upem > 0 else 0.0,
                aw,
            ]))

            # Pre-computed font style embedding.
            font_embeddings.append(self._font_embedding[font.path])

        return {
            "char": chars,
            "target_rendering": torch.stack(target_renderings),
            "content_rendering": torch.stack(content_renderings),
            "style_renderings": torch.stack(style_renderings),
            "style_chars": torch.tensor(style_chars, dtype=torch.long),
            "font_style_embedding": torch.stack(font_embeddings),
            "description": descriptions,
            "metrics": torch.stack(all_metrics),
            "bbox_size": torch.stack(bbox_sizes),
        }


class _ClassBalancedBatchSampler(BatchSampler):
    """Batch sampler that balances font classes within each emitted batch."""

    def __init__(
        self,
        order: Sequence[tuple],
        *,
        batch_size: int,
        drop_last: bool,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if len(order) == 0:
            raise ValueError("Cannot build class-balanced sampler for empty dataset")

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.dataset_size = len(order)

        class_to_indices: dict[str, list[int]] = {}
        for idx, (font, _char) in enumerate(order):
            cls = font.classification()
            class_to_indices.setdefault(cls, []).append(idx)

        if not class_to_indices:
            raise ValueError("No classes found for class-balanced sampling")

        self.class_to_indices = class_to_indices
        self.classes = sorted(class_to_indices.keys())

    def __len__(self) -> int:
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return math.ceil(self.dataset_size / self.batch_size)

    def __iter__(self):
        num_classes = len(self.classes)
        num_batches = len(self)

        class_cursor = random.randrange(num_classes)

        for _ in range(num_batches):
            batch_indices: list[int] = []

            if num_classes <= self.batch_size:
                base = self.batch_size // num_classes
                remainder = self.batch_size % num_classes
                class_order = self.classes[:]
                random.shuffle(class_order)

                for cls in class_order:
                    indices = self.class_to_indices[cls]
                    for _ in range(base):
                        batch_indices.append(random.choice(indices))

                for cls in class_order[:remainder]:
                    batch_indices.append(random.choice(self.class_to_indices[cls]))
            else:
                selected_classes = [
                    self.classes[(class_cursor + i) % num_classes]
                    for i in range(self.batch_size)
                ]
                class_cursor = (class_cursor + self.batch_size) % num_classes
                for cls in selected_classes:
                    batch_indices.append(random.choice(self.class_to_indices[cls]))

            random.shuffle(batch_indices)
            yield batch_indices


__all__ = ["ARPhase1DatasetMaker"]
