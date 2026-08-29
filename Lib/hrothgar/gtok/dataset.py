"""A dataset maker for G-Tok.

Loads the Google Fonts repository and produces batches of
(char, rendering, description) tuples for training.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Sequence

import torch
from torch.utils.data import (
    BatchSampler,
    WeightedRandomSampler,
)
from torch.utils.data import (
    Dataset as TorchDataset,
)

from hrothgar.dataset import LATIN_CORE, ClassBalancedBatchSampler, DatasetMaker
from hrothgar.glyph_rendering import crop_to_ink

# Dataset-level oversampling policy for underperforming style buckets.
# Keep this in source (not CLI args) so training setup is reproducible from code.
GTOK_CLASS_BUCKET_OVERSAMPLING: dict[str, int] = {
    "DISPLAY": 2,
    "DISPLAY_HANDWRITING": 2,
}


def _classification_oversample_factor(
    classification: str,
    class_bucket_oversampling: dict[str, int] | None,
) -> int:
    """Return the configured multiplicative factor for a class bucket."""
    if not class_bucket_oversampling:
        return 1
    key = (classification or "UNKNOWN").upper().replace("/", "_")
    factor = class_bucket_oversampling.get(key, 1)
    return max(1, int(factor))


def _limit_axis_positions(
    positions: Sequence[Sequence[float]], max_positions: int
) -> List[List[float]]:
    """Downsample axis positions deterministically to cap dataset growth."""
    if max_positions <= 0 or len(positions) <= max_positions:
        return [list(pos) for pos in positions]
    if max_positions == 1:
        return [list(positions[0])]

    step = (len(positions) - 1) / (max_positions - 1)
    selected: List[List[float]] = []
    selected_indices: set[int] = set()
    for i in range(max_positions):
        idx = int(round(i * step))
        if idx in selected_indices:
            continue
        selected_indices.add(idx)
        selected.append(list(positions[idx]))
    return selected


class GTokAxisDataset(TorchDataset):
    """Latin-Core dataset expanded across sampled variable-font axis positions."""

    def __init__(
        self,
        fonts: Iterable,
        *,
        codepoint_filter_fn,
        axis_splits: int,
        max_axis_positions_per_font: int,
        class_bucket_oversampling: dict[str, int] | None = None,
    ):
        self.order = []
        for font in fonts:
            chars = sorted(codepoint_filter_fn(set(font.codepoints)))
            axis_positions = font.sample_axis_positions(splits=axis_splits)
            axis_positions = _limit_axis_positions(
                axis_positions, max_axis_positions_per_font
            )
            oversample_factor = _classification_oversample_factor(
                font.classification(), class_bucket_oversampling
            )
            for char in chars:
                for axis_position in axis_positions:
                    for _ in range(oversample_factor):
                        self.order.append((font, char, axis_position))

    def __len__(self):
        return len(self.order)

    def __getitem__(self, idx):
        font, char, axis_position = self.order[idx]
        return {
            "char": char,
            "axis_position": axis_position,
            "font": font,
        }

    def weighted_sampler(self) -> WeightedRandomSampler:
        """Return a ``WeightedRandomSampler`` that balances samples across font classes.

        Each sample's weight is the inverse frequency of its font's classification
        within the dataset, so under-represented classes are sampled proportionally
        more often.
        """
        classifications = [font.classification() for font, _char, _pos in self.order]
        counts: Counter[str] = Counter(classifications)
        weights = [1.0 / counts[cls] for cls in classifications]
        return WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )

    def class_balanced_batch_sampler(
        self,
        *,
        batch_size: int,
        drop_last: bool,
    ) -> BatchSampler:
        """Return a batch sampler that balances font classes inside each batch.

        Unlike ``weighted_sampler`` (which balances expected class frequency over
        many draws), this sampler enforces near-equal class quotas *within each
        batch*.
        """
        return ClassBalancedBatchSampler(
            self.order,
            key=lambda item: item[0].classification(),
            batch_size=batch_size,
            drop_last=drop_last,
        )


class GTokDatasetMaker(DatasetMaker):
    def __init__(self, repo_url: str, batch_size: int, **kwargs):
        target = kwargs.pop("target_codepoints", None)
        if target is None:
            # Use the G-Tok config's character set if provided, else LATIN_CORE.
            gtok_config = kwargs.pop("gtok_config", None)
            if gtok_config is not None:
                target = set(gtok_config.character_set)
            else:
                target = set(LATIN_CORE)
        self.axis_splits = kwargs.pop("axis_splits", 3)
        self.max_axis_positions_per_font = kwargs.pop("max_axis_positions_per_font", 24)
        self.class_balanced = kwargs.pop("class_balanced", False)
        self.max_display_score = kwargs.pop("max_display_score", 0)
        self.render_time_augmentation = kwargs.pop("render_time_augmentation", False)
        if self.render_time_augmentation:
            # One entry per font — augmentation happens at render time.
            self.axis_splits = 1
            self.max_axis_positions_per_font = 1
        super().__init__(
            repo_url=repo_url,
            batch_size=batch_size,
            having=kwargs.pop("having", None),
            target_codepoints=target,
            canary_size=kwargs.pop("canary_size", None),
            image_size=kwargs.pop("image_size", 128),
            character_set=list(target) if target else None,
            **kwargs,
        )

    def filter_fonts(self):
        # Use same filter as AR.
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
        # Filter out highly display-oriented fonts.
        if self.max_display_score > 0:
            before = len(self.googlefonts.fonts)
            self.googlefonts.fonts = [
                f
                for f in self.googlefonts.fonts
                if f.display_score() <= self.max_display_score
            ]
            print(
                f"Display filter (max_score={self.max_display_score}): "
                f"{before} → {len(self.googlefonts.fonts)} fonts"
            )

    def train_set(self):
        return GTokAxisDataset(
            self.train_fonts,
            codepoint_filter_fn=self.train_codepoint_filter,
            axis_splits=self.axis_splits,
            max_axis_positions_per_font=self.max_axis_positions_per_font,
            class_bucket_oversampling=GTOK_CLASS_BUCKET_OVERSAMPLING,
        )

    def train_loader(self):
        from torch.utils.data import DataLoader

        dataset = self.train_set()
        if self.class_balanced:
            batch_sampler = dataset.class_balanced_batch_sampler(
                batch_size=self.batch_size,
                drop_last=True,
            )
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=self.collate_fn,
                pin_memory=True,
                num_workers=12,
            )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fn,
            pin_memory=True,
            num_workers=12,
        )

    def test_set(self):
        return GTokAxisDataset(
            self.test_fonts,
            codepoint_filter_fn=self.test_codepoint_filter,
            axis_splits=self.axis_splits,
            max_axis_positions_per_font=self.max_axis_positions_per_font,
            class_bucket_oversampling=None,
        )

    def collate_fn(self, batch):
        chars = torch.tensor([item["char"] for item in batch])
        renderings = []
        for item in batch:
            font = item["font"]
            axis_pos = item["axis_position"]
            if self.render_time_augmentation:
                axis_pos = font.random_axis_position()
            rendered = torch.tensor(
                font.render(
                    item["char"], size=self.image_size, axis_position=axis_pos
                ),
                dtype=torch.float32,
            )
            renderings.append(crop_to_ink(rendered, self.image_size))
        renderings = torch.stack(renderings)
        descriptions = [
            item["font"].description_with_tags_and_display() for item in batch
        ]
        classifications = [item["font"].classification() for item in batch]
        families = [item["font"].family for item in batch]

        return {
            "char": chars,
            "rendering": renderings,
            "description": descriptions,
            "classification": classifications,
            "family": families,
        }
