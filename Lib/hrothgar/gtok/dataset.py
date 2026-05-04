"""A dataset maker for G-Tok.

Loads the Google Fonts repository and produces batches of
(char, rendering, description) tuples for training.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Sequence

import torch
from torch.utils.data import Dataset as TorchDataset, WeightedRandomSampler

from hrothgar.dataset import DatasetMaker, LATIN_CORE


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
    ):
        self.order = []
        for font in fonts:
            chars = sorted(codepoint_filter_fn(set(font.codepoints)))
            axis_positions = font.sample_axis_positions(splits=axis_splits)
            axis_positions = _limit_axis_positions(
                axis_positions, max_axis_positions_per_font
            )
            for char in chars:
                for axis_position in axis_positions:
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


class GTokDatasetMaker(DatasetMaker):
    def __init__(self, repo_url: str, batch_size: int, **kwargs):
        target = kwargs.pop("target_codepoints", None)
        if target is None:
            target = set(LATIN_CORE)
        self.axis_splits = kwargs.pop("axis_splits", 3)
        self.max_axis_positions_per_font = kwargs.pop("max_axis_positions_per_font", 24)
        self.class_balanced = kwargs.pop("class_balanced", False)
        super().__init__(
            repo_url=repo_url,
            batch_size=batch_size,
            having=kwargs.pop("having", None),
            target_codepoints=target,
            canary_size=kwargs.pop("canary_size", None),
            image_size=kwargs.pop("image_size", 128),
            **kwargs,
        )

    def train_set(self):
        return GTokAxisDataset(
            self.train_fonts,
            codepoint_filter_fn=self.train_codepoint_filter,
            axis_splits=self.axis_splits,
            max_axis_positions_per_font=self.max_axis_positions_per_font,
        )

    def train_loader(self):
        from torch.utils.data import DataLoader

        dataset = self.train_set()
        if self.class_balanced:
            sampler = dataset.weighted_sampler()
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                drop_last=True,
                collate_fn=self.collate_fn,
            )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def test_set(self):
        return GTokAxisDataset(
            self.test_fonts,
            codepoint_filter_fn=self.test_codepoint_filter,
            axis_splits=self.axis_splits,
            max_axis_positions_per_font=self.max_axis_positions_per_font,
        )

    def collate_fn(self, batch):
        chars = torch.tensor([item["char"] for item in batch])
        renderings = torch.stack(
            [
                torch.tensor(
                    item["font"].render(
                        item["char"],
                        size=self.image_size,
                        axis_position=item["axis_position"],
                    )
                )
                for item in batch
            ]
        )
        descriptions = [item["font"].description_with_tags_and_display() for item in batch]
        classifications = [item["font"].classification() for item in batch]

        return {
            "char": chars,
            "rendering": renderings,
            "description": descriptions,
            "classification": classifications,
        }
