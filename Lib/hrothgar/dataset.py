import math
import random
from collections import defaultdict
from typing import Callable, Generic, Optional, Sequence, Set, TypeVar

import torch
import uharfbuzz as hb
from sklearn.model_selection import train_test_split
from torch.utils.data import BatchSampler, DataLoader
from torch.utils.data import Dataset as TorchDataset

from hrothgar.googlefonts import GoogleFonts

from hrothgar.dataset_constants import CAPS_ONLY, LATIN_CORE, LATIN_KERNEL, LGC_ALL


_T = TypeVar("_T")


def _has_non_empty_outline(extents) -> bool:
    """Return True when HarfBuzz extents indicate drawable geometry."""
    if extents is None:
        return False
    _x_bearing, _y_bearing, width, height = extents
    return not (width == 0 and height == 0)


def _hb_font_for_face(face):
    """Construct a HarfBuzz Font object for a face."""
    return getattr(hb, "Font")(face)


class ClassBalancedBatchSampler(BatchSampler, Generic[_T]):
    """Batch sampler that balances a class label within each emitted batch.

    Items are grouped by a caller-supplied ``key`` (for example a font's
    category or classification), and each batch draws an even quota of items
    from every class.  This stops majority classes (e.g. sans-serif fonts)
    from dominating training batches.
    """

    def __init__(
        self,
        items: Sequence[_T],
        *,
        key: Callable[[_T], str],
        batch_size: int,
        drop_last: bool,
        rng=None,
    ) -> None:
        """
        ``rng`` is the RNG used for font selection; defaults to the global
        ``random`` module.  Passing a dedicated ``random.Random`` makes batch
        composition reproducible (canary mode).
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if len(items) == 0:
            raise ValueError("Cannot build class-balanced sampler for empty dataset")

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.dataset_size = len(items)
        self._rng = rng if rng is not None else random

        class_to_indices: dict[str, list[int]] = {}
        for idx, item in enumerate(items):
            class_to_indices.setdefault(key(item), []).append(idx)

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

        class_cursor = self._rng.randrange(num_classes)

        for _ in range(num_batches):
            batch_indices: list[int] = []

            if num_classes <= self.batch_size:
                base = self.batch_size // num_classes
                remainder = self.batch_size % num_classes
                class_order = self.classes[:]
                self._rng.shuffle(class_order)

                for cls in class_order:
                    indices = self.class_to_indices[cls]
                    for _ in range(base):
                        batch_indices.append(self._rng.choice(indices))

                for cls in class_order[:remainder]:
                    batch_indices.append(self._rng.choice(self.class_to_indices[cls]))
            else:
                selected_classes = [
                    self.classes[(class_cursor + i) % num_classes]
                    for i in range(self.batch_size)
                ]
                class_cursor = (class_cursor + self.batch_size) % num_classes
                for cls in selected_classes:
                    batch_indices.append(self._rng.choice(self.class_to_indices[cls]))

            self._rng.shuffle(batch_indices)
            yield batch_indices


class DatasetMaker:
    """Create train/test splits and loaders over glyph rendering items."""

    def __init__(
        self,
        repo_url: str,
        batch_size: int,
        having: Optional[Set[int]] = None,
        target_codepoints: Optional[Set[int]] = None,
        canary_size: Optional[int] = None,
        image_size: int = 128,
        split_seed: int = 1234,
        character_set: Optional[Sequence[int]] = None,
    ):
        self.target_codepoints = set(target_codepoints) if target_codepoints else None
        if character_set is None:
            character_set = LATIN_CORE
        self._character_set: list[int] = sorted(set(character_set))
        having_filter: Optional[Set[int]] = None
        if having is not None:
            having_filter = set(having)

        self.googlefonts = GoogleFonts(
            repo_url,
            having=having_filter,
            # Canary mode only needs the first few fonts; don't scan the whole
            # repository to get them.
            max_fonts=canary_size,
        )
        self.filter_fonts()
        self.batch_size = batch_size
        self.image_size = image_size
        self.split_seed = split_seed
        # Keep data-order randomization reproducible without forcing fixed batches.
        self._train_loader_generator = torch.Generator()
        self._train_loader_generator.manual_seed(self.split_seed + 1)
        self._test_loader_generator = torch.Generator()
        self._test_loader_generator.manual_seed(self.split_seed + 2)

        # Test chars are a random split from the character set.
        _, self.test_codepoints = train_test_split(
            self._character_set,
            random_state=self.split_seed,
        )

        self.canary_size = canary_size

        if canary_size is not None:
            fonts = self.googlefonts.fonts[:canary_size]
        else:
            fonts = self.googlefonts.fonts

        self.train_fonts, self.test_fonts = self._split_fonts_by_family(
            fonts,
            split_seed=self.split_seed,
        )
        print("Train fonts:", len(self.train_fonts))
        print("Test fonts:", len(self.test_fonts))

    @staticmethod
    def _split_fonts_by_family(fonts, *, split_seed: int):
        """Split fonts into train/test by family to avoid cross-style leakage."""
        if len(fonts) < 2:
            return list(fonts), []

        family_to_fonts = defaultdict(list)
        for font in fonts:
            family_to_fonts[font.family].append(font)

        families = sorted(family_to_fonts.keys())
        if len(families) < 2:
            # If only one family is available, keep current behaviour and avoid empty train.
            return list(fonts), []

        train_families, test_families = train_test_split(
            families,
            random_state=split_seed,
        )

        train_family_set = set(train_families)
        test_family_set = set(test_families)

        train_fonts = [
            font for family in train_family_set for font in family_to_fonts[family]
        ]
        test_fonts = [
            font for family in test_family_set for font in family_to_fonts[family]
        ]
        return train_fonts, test_fonts

    def filter_fonts(self):
        pass

    def train_set(self):
        return Dataset(
            self.train_fonts, codepoint_filter_fn=self.train_codepoint_filter
        )

    def test_set(self):
        return Dataset(self.test_fonts, codepoint_filter_fn=self.test_codepoint_filter)

    def train_codepoint_filter(self, font_codepoints: Set[int]) -> Set[int]:
        if self.target_codepoints is not None:
            return set(font_codepoints) & self.target_codepoints
        return set(font_codepoints) - set(self.test_codepoints)

    def test_codepoint_filter(self, font_codepoints: Set[int]) -> Set[int]:
        if self.target_codepoints is not None:
            return set(font_codepoints) & self.target_codepoints
        return set(font_codepoints) & set(self.test_codepoints)

    def train_loader(self):
        return DataLoader(
            self.train_set(),
            batch_size=self.batch_size,
            shuffle=True,
            generator=self._train_loader_generator,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def test_loader(self):
        return DataLoader(
            self.test_set(),
            batch_size=self.batch_size,
            shuffle=True,
            generator=self._test_loader_generator,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        raise NotImplementedError("Base DatasetMaker does not implement collate_fn")


class Dataset(TorchDataset):
    def __init__(self, fonts, codepoint_filter_fn: Callable[[Set[int]], Set[int]]):
        self.fonts = fonts
        self.codepoint_filter_fn = codepoint_filter_fn
        self.order = []
        for font in self.fonts:
            hb_font = _hb_font_for_face(font.hb_face)
            chars = self.codepoint_filter_fn(set(font.codepoints))
            for char in chars:
                # Skip empty glyphs; they can destabilize training targets.
                gid = hb_font.get_nominal_glyph(char)
                extents = hb_font.get_glyph_extents(gid)
                if _has_non_empty_outline(extents):
                    self.order.append((font, char))

    def __len__(self):
        return len(self.order)

    def __getitem__(self, idx):
        font, char = self.order[idx]
        return {
            "char": char,
            "font": font,
        }


class AllGidsDataset(TorchDataset):
    def __init__(self, fonts):
        self.fonts = fonts
        self.order = []
        for font in self.fonts:
            hb_font = _hb_font_for_face(font.hb_face)
            for gid in range(1, font.hb_face.glyph_count):
                # Skip empty glyphs; they can destabilize training targets.
                extents = hb_font.get_glyph_extents(gid)
                if _has_non_empty_outline(extents):
                    self.order.append((font, gid))

    def __len__(self):
        return len(self.order)

    def __getitem__(self, idx):
        font, gid = self.order[idx]
        return {
            "gid": gid,
            "font": font,
        }
