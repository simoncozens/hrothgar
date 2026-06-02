"""Dataset utilities for AR phase-1 visual pretraining.

This module reuses the shared font split logic from ``blys.dataset`` and
provides a collation function tailored to the AR generator inputs.
"""

from __future__ import annotations

import random
from typing import Optional, Sequence, Set

import torch
from blys.googlefonts import GoogleFont
from blys.font import Font
from blys.dataset import (
    Dataset,
    DatasetMaker,
    LATIN_KERNEL,
    ClassBalancedBatchSampler,
)
from blys.render import is_blank_rendering
from torch.utils.data import DataLoader


def sample_style_codepoints(
    *,
    font: Font,
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
        if cp != target_char and font.has_non_empty_codepoint(cp)
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
            and font.has_non_empty_codepoint(cp)
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
                batch_sampler=ClassBalancedBatchSampler(
                    dataset.order,
                    batch_size=self.batch_size,
                    drop_last=True,
                ),
                num_workers=8,
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
                torch.tensor(
                    item["font"].render_char(item["char"], size=self.image_size)
                )
                for item in batch
            ]
        )

        content_renderings = []
        style_renderings = []
        style_chars = []
        descriptions = []

        for item in batch:
            font: GoogleFont = item["font"]
            char = item["char"]
            reference_font = font.reference_font() or font

            # If reference font lacks a usable glyph for this character, fall back
            # to the target font so content conditioning is never blank.
            if not reference_font.has_codepoint(
                char
            ) or not reference_font.has_non_empty_codepoint(char):
                reference_font = font

            content_render = reference_font.render_char(char, size=self.image_size)
            if is_blank_rendering(content_render):
                content_render = font.render_char(char, size=self.image_size)
            content_renderings.append(torch.tensor(content_render))

            sampled_style_chars = sample_style_codepoints(
                font=font,
                target_char=char,
                style_glyph_count=self.style_glyph_count,
                common_style_codepoints=self.common_style_codepoints,
            )
            rendered_styles = []
            sanitized_style_chars = []
            for cp in sampled_style_chars:
                style_render = font.render_char(cp, size=self.image_size)
                if is_blank_rendering(style_render):
                    cp = char
                    style_render = font.render_char(cp, size=self.image_size)
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


__all__ = ["ARPhase1DatasetMaker"]
