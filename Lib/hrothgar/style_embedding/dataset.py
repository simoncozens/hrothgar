"""Dataset and collation for font style embedding training (phrase input).

Each batch item is one font.  The collate function renders two different phrases
for each font to create contrastive views.  Variable fonts additionally get
different axis positions per view.
"""

from __future__ import annotations

import random
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader

from hrothgar.dataset import DatasetMaker, Dataset
from hrothgar.googlefonts import GoogleFont, ALL_CATEGORIES
from hrothgar.render import render_phrase
from hrothgar.dataset_constants import LATIN_CORE


# Phrases used for contrastive views.  Two different phrases per font
# teach the model that style is invariant to the rendered text content.
CONTRASTIVE_PHRASES = [
    "THE quick brown fox 1234",
    "HAMBURGE hamburger 5678",
    "123 SHOPLIFT shoplift 123",
    "adhesion ADHESION 9876",
]


class FontStyleDatasetMaker(DatasetMaker):
    """Dataset maker for font-level style embedding (phrase input)."""

    def __init__(
        self,
        repo_url: str | Path,
        batch_size: int,
        *,
        image_size: int = 512,
        phrase_width: int = 512,
        phrase_font_size: int = 32,
        split_seed: int = 1234,
        canary_size: Optional[int] = None,
        tag_names: Optional[list[str]] = None,
        tag_num_classes: int = 0,
        class_balanced: bool = True,
    ):
        super().__init__(
            repo_url=str(repo_url),
            batch_size=batch_size,
            image_size=image_size if image_size > phrase_width // 2 else phrase_width // 2,
            split_seed=split_seed,
            canary_size=canary_size,
            character_set=list(LATIN_CORE),
        )
        self._tag_names = tag_names or []
        self._tag_num_classes = tag_num_classes
        self._target_image_size = image_size
        self._phrase_width = phrase_width
        self._phrase_font_size = phrase_font_size
        self._class_balanced = class_balanced

    def filter_fonts(self) -> None:
        """Remove fonts that don't have the characters needed for phrases."""
        needed = set()
        for phrase in CONTRASTIVE_PHRASES:
            needed.update(ord(c) for c in phrase if not c.isspace())

        self.googlefonts.fonts = [
            font
            for font in self.googlefonts.fonts
            if needed <= font.codepoints
        ]

    def train_set(self):
        return Dataset(
            self.train_fonts,
            codepoint_filter_fn=lambda cps: set(cps),
        )

    def test_set(self):
        return Dataset(
            self.test_fonts,
            codepoint_filter_fn=lambda cps: set(cps),
        )

    def train_loader(self):
        dataset = self.train_set()
        if self._class_balanced:
            return DataLoader(
                dataset,
                batch_sampler=_ClassBalancedBatchSampler(
                    dataset.order,
                    batch_size=self.batch_size,
                    drop_last=True,
                ),
                collate_fn=self.collate_fn,
                num_workers=0,
                pin_memory=True,
            )
        return super().train_loader()

    def collate_fn(self, batch: list[dict]) -> dict:
        """Collate font-level items into a training batch.

        Returns two views per font:
        - View 1: phrase A at axis position 1.
        - View 2: phrase B at axis position 2 (different phrase, and
          different axis position for variable fonts).

        Returns:
            dict with keys:
            - ``"images"``: ``(2*B, 3, H, H)`` — view 1 stacked on view 2,
              resized to a square.
            - ``"tags"``, ``"tag_masks"``, ``"category"``, ``"family"`` as before.
        """
        fonts: list[GoogleFont] = [item["font"] for item in batch]

        images_v1: list[torch.Tensor] = []
        images_v2: list[torch.Tensor] = []

        for font in fonts:
            axis1 = font.random_axis_position()
            axis2 = font.random_axis_position()

            # Pick two distinct phrases.
            p1, p2 = random.sample(CONTRASTIVE_PHRASES, 2)

            # Render and resize to square.
            img1 = _render_and_resize(
                font, p1, self._phrase_font_size, axis1,
                self._phrase_width, self._target_image_size,
            )
            img2 = _render_and_resize(
                font, p2, self._phrase_font_size, axis2,
                self._phrase_width, self._target_image_size,
            )
            images_v1.append(img1)
            images_v2.append(img2)

        images = torch.cat([
            torch.stack(images_v1),
            torch.stack(images_v2),
        ], dim=0)  # (2B, 3, H, H)

        # Tags.
        tags: dict[str, torch.Tensor] = {}
        tag_masks: dict[str, torch.Tensor] = {}
        nc = self._tag_num_classes
        for tag_name in self._tag_names:
            values = []
            present = []
            for font in fonts:
                font_tags = font.tags()
                if tag_name in font_tags:
                    raw = font_tags[tag_name] / 100.0
                    if nc > 0:
                        raw = min(int(raw * nc), nc - 1)
                    values.append(raw)
                    present.append(True)
                else:
                    values.append(0)
                    present.append(False)
            dtype = torch.long if nc > 0 else torch.float32
            tags[tag_name] = torch.tensor(values, dtype=dtype)
            tag_masks[tag_name] = torch.tensor(present, dtype=torch.bool)

        # Category and family.
        cat_map = {c: i for i, c in enumerate(ALL_CATEGORIES)}
        categories = torch.tensor(
            [cat_map.get(font.category(), 5) for font in fonts],
            dtype=torch.long,
        )
        families = [font.family for font in fonts]

        return {
            "images": images,
            "tags": tags,
            "tag_masks": tag_masks,
            "category": categories,
            "family": families,
        }


def _render_and_resize(
    font: GoogleFont,
    phrase: str,
    font_size: int,
    axis_position,
    phrase_width: int,
    target_size: int,
) -> torch.Tensor:
    """Render a phrase in a font and resize to a square RGB tensor."""
    import torchvision.transforms.functional as TF

    arr = render_phrase(
        str(font.path), phrase,
        size=font_size,
        axis_position=axis_position if axis_position and len(axis_position) > 0 else None,
        width=phrase_width,
    )
    # arr is (H, W, 4) RGBA uint8 — strip alpha for RGB.
    img = torch.from_numpy(arr[..., :3].copy()).permute(2, 0, 1).float() / 255.0
    # Resize to square.
    img = TF.resize(img, [target_size, target_size], antialias=True)
    return img


class _ClassBalancedBatchSampler(BatchSampler):
    """Batch sampler that balances broad font categories within each batch."""

    def __init__(
        self,
        order: list[tuple],
        *,
        batch_size: int,
        drop_last: bool,
    ) -> None:
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.dataset_size = len(order)

        class_to_indices: dict[str, list[int]] = {}
        for idx, (font, _char) in enumerate(order):
            cls = font.category()  # one of ALL_CATEGORIES
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
