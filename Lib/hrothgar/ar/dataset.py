"""Dataset utilities for AR phase-1 visual pretraining.

This module reuses the shared font split logic from ``hrothgar.dataset`` and
provides a collation function tailored to the AR generator inputs.
"""

from __future__ import annotations

import math
import random
from typing import Optional, Sequence, Set

import torch
import uharfbuzz as hb
from hrothgar.dataset import Dataset, DatasetMaker, LATIN_KERNEL
from torch.utils.data import BatchSampler, DataLoader


def _has_non_empty_glyph(font, codepoint: int) -> bool:
    """Return True if the font has a non-empty outline for codepoint."""
    if not hasattr(font, "hb_face"):
        # Test doubles and lightweight mocks may not expose HarfBuzz handles.
        return True
    hb_font = hb.Font(font.hb_face)  # type: ignore
    gid = hb_font.get_nominal_glyph(codepoint)
    extents = hb_font.get_glyph_extents(gid)
    if extents is None:
        return False
    return not all(x == 0 for x in extents)


def _font_has_codepoint(font, codepoint: int) -> bool:
    if hasattr(font, "has_codepoint"):
        return bool(font.has_codepoint(codepoint))
    return codepoint in getattr(font, "codepoints", set())


def _is_blank_rendering(rendered) -> bool:
    max_val = float(rendered.max())
    min_val = float(rendered.min())
    # Real blank glyph rasters are typically all-white (1.0) and occasionally
    # all-black (0.0) in this pipeline.
    return max_val == min_val and (max_val == 1.0 or max_val == 0.0)


def _sample_style_codepoints(
    *,
    font,
    target_char: int,
    style_glyph_count: int,
    common_style_codepoints: Optional[Sequence[int]],
) -> list[int]:
    """Select style codepoints for one target item.

    By default this samples per-item codepoints from the same font, excluding
    the target codepoint. If a common list is provided, it is used first and
    then padded with per-font sampling when needed.
    """

    if style_glyph_count <= 0:
        raise ValueError(f"style_glyph_count must be positive, got {style_glyph_count}")

    available = [
        cp
        for cp in font.codepoints
        if cp != target_char and _has_non_empty_glyph(font, cp)
    ]
    # Restrict to GF Latin Kernel
    available = [cp for cp in available if cp in LATIN_KERNEL]

    if not available:
        return [target_char] * style_glyph_count

    selected: list[int] = []
    if common_style_codepoints is not None:
        selected = [
            cp
            for cp in common_style_codepoints
            if cp != target_char
            and cp in font.codepoints
            and _has_non_empty_glyph(font, cp)
        ]
        random.shuffle(selected)

        # If the caller pinned style chars, never leak arbitrary codepoints in.
        if len(selected) >= style_glyph_count:
            return selected[:style_glyph_count]
        if selected:
            selected.extend(
                random.choices(selected, k=style_glyph_count - len(selected))
            )
        else:
            selected.extend([target_char] * style_glyph_count)
        return selected

    if len(selected) >= style_glyph_count:
        return selected[:style_glyph_count]

    remaining_count = style_glyph_count - len(selected)
    remaining_candidates = [cp for cp in available if cp not in selected]
    if len(remaining_candidates) >= remaining_count:
        selected.extend(random.sample(remaining_candidates, remaining_count))
    else:
        selected.extend(remaining_candidates)
        if selected:
            selected.extend(
                random.choices(selected, k=style_glyph_count - len(selected))
            )
        else:
            selected.extend([target_char] * (style_glyph_count - len(selected)))

    return selected


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

    This class reuses the same train/test split and item ordering logic as the
    GTok dataset maker, but emits AR-specific batches during collation.
    """

    def __init__(
        self,
        repo_url: str,
        batch_size: int,
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
        super().__init__(
            repo_url=repo_url,
            batch_size=batch_size,
            having=having,
            target_codepoints=None,
            canary_size=canary_size,
            image_size=image_size,
            split_seed=split_seed,
        )
        if style_glyph_count <= 0:
            raise ValueError(
                f"style_glyph_count must be positive, got {style_glyph_count}"
            )
        if target_codepoint_oversample_factor <= 0:
            raise ValueError(
                "target_codepoint_oversample_factor must be positive, got "
                f"{target_codepoint_oversample_factor}"
            )
        self.style_glyph_count = style_glyph_count
        self.common_style_codepoints = common_style_codepoints
        self.class_balanced = class_balanced
        self.extra_target_codepoints = target_codepoint_set
        self.target_codepoint_oversample_factor = target_codepoint_oversample_factor
        self.target_only = target_only

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
                # For now try only very non-display fonts
                or font.display_score() > 60.0
            )
        ]

    def train_codepoint_filter(self, font_codepoints: Set[int]) -> Set[int]:
        if self.target_only:
            return set(font_codepoints) & (self.extra_target_codepoints or set())
        chars = super().train_codepoint_filter(font_codepoints)
        if self.extra_target_codepoints is None:
            return chars
        return set(chars) | (set(font_codepoints) & self.extra_target_codepoints)

    def test_codepoint_filter(self, font_codepoints: Set[int]) -> Set[int]:
        if self.target_only:
            return set(font_codepoints) & (self.extra_target_codepoints or set())
        chars = super().test_codepoint_filter(font_codepoints)
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
        Plus metadata useful for debugging and future adaptation wiring.
        """

        chars = torch.tensor([item["char"] for item in batch], dtype=torch.long)
        target_renderings = torch.stack(
            [
                torch.tensor(item["font"].render(item["char"], size=self.image_size))
                for item in batch
            ]
        )

        content_renderings = []
        style_renderings = []
        style_chars = []
        descriptions = []

        for item in batch:
            font = item["font"]
            char = item["char"]
            reference_font = font.reference_font() or font

            # If reference font lacks a usable glyph for this character, fall back
            # to the target font so content conditioning is never blank.
            if not _font_has_codepoint(
                reference_font, char
            ) or not _has_non_empty_glyph(reference_font, char):
                reference_font = font

            content_render = reference_font.render(char, size=self.image_size)
            if _is_blank_rendering(content_render):
                content_render = font.render(char, size=self.image_size)
            content_renderings.append(torch.tensor(content_render))

            sampled_style_chars = _sample_style_codepoints(
                font=font,
                target_char=char,
                style_glyph_count=self.style_glyph_count,
                common_style_codepoints=self.common_style_codepoints,
            )
            rendered_styles = []
            sanitized_style_chars = []
            for cp in sampled_style_chars:
                style_render = font.render(cp, size=self.image_size)
                if _is_blank_rendering(style_render):
                    cp = char
                    style_render = font.render(cp, size=self.image_size)
                sanitized_style_chars.append(cp)
                rendered_styles.append(torch.tensor(style_render))

            style_chars.append(sanitized_style_chars)
            style_renderings.append(torch.stack(rendered_styles))
            descriptions.append(font.description_with_tags_and_display())

        return {
            "char": chars,
            "target_rendering": target_renderings,
            "content_rendering": torch.stack(content_renderings),
            "style_renderings": torch.stack(style_renderings),
            "style_chars": torch.tensor(style_chars, dtype=torch.long),
            "description": descriptions,
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
