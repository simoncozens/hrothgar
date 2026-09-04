"""Full-dataset maker for the factorized (codepoint, font-ID) diffusion model.

Each training item is a ``(glyph image, codepoint index, font index)`` triple.
The font index is a learned one-hot lookup (VecFusion's "incomplete fonts"
recipe), and the codepoint index is a separate learned lookup.  Factorization
into two tables is what lets the model compose style (font) with content
(codepoint) for pairs it never saw together.

The split is **codepoint-based**, not font-based: for each codepoint we hold out
a fraction of the fonts that contain it (the "fill in the missing glyph"
scenario).  Two invariants are enforced:

* every codepoint keeps at least ``min_train_fonts_per_codepoint`` training
  fonts (so its content embedding is learned), and
* every font keeps most of its codepoints (so its style embedding is learned).

The held-out ``(font, codepoint)`` pairs form the validation set — generating
them correctly is the actual acceptance test.
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from hrothgar.dataset import LATIN_KERNEL, _has_non_empty_outline, _hb_font_for_face
from hrothgar.googlefonts import GoogleFont, GoogleFonts
from hrothgar.glyph_rendering import geometry_tensor
from hrothgar.style_extraction.render_utils import render_glyph_with_geometry

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))


class _PairDataset(TorchDataset):
    """Yields ``(image, cp_idx, font_id)`` for a fixed list of pairs."""

    def __init__(
        self,
        pairs: list[tuple[int, int]],
        fonts: Sequence[GoogleFont],
        cp_list: Sequence[int],
        image_size: int,
    ) -> None:
        self.pairs = pairs
        self.fonts = fonts
        self.cp_list = list(cp_list)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        font_id, cp_idx = self.pairs[idx]
        cp = self.cp_list[cp_idx]
        img, geometry = render_glyph_with_geometry(
            self.fonts[font_id], cp, self.image_size
        )  # img: (H, W); geometry: dict of em-unit labels
        return {
            "image": img.unsqueeze(0),  # (1, H, W)
            "geometry": geometry_tensor(geometry),  # (5,)
            "cp_idx": cp_idx,
            "font_id": font_id,
        }


def _collate_fn(batch: list[dict]) -> dict:
    return {
        "images": torch.stack([b["image"] for b in batch]),
        "geometry": torch.stack([b["geometry"] for b in batch]),
        "codepoints": torch.tensor([b["cp_idx"] for b in batch], dtype=torch.long),
        "font_ids": torch.tensor([b["font_id"] for b in batch], dtype=torch.long),
    }


class FontIdDatasetMaker:
    """Builds train/val loaders of ``(image, codepoint, font_id)`` triples."""

    def __init__(
        self,
        repo: str | Path,
        batch_size: int,
        *,
        image_size: int = 128,
        character_set: Optional[Sequence[int]] = None,
        extra_codepoints: Optional[Sequence[int]] = None,
        remove_codepoints: Optional[Sequence[int]] = None,
        oversample_codepoints: Optional[dict[int, int]] = None,
        heldout_fraction: float = 0.25,
        min_train_fonts_per_codepoint: int = 20,
        split_seed: int = 1234,
        canary_size: Optional[int] = None,
    ) -> None:
        self.image_size = image_size
        self.batch_size = batch_size
        # Vocabulary = base set + extras - removals.
        base = set(character_set or LATIN_KERNEL)
        base |= set(extra_codepoints or [])
        base -= set(remove_codepoints or [])
        self.character_set = sorted(base)
        # Codepoint -> oversample factor.  Oversampled codepoints are fully
        # trained (no held-out split) so rare glyphs like the rupee sign get
        # every available example, then their training pairs are duplicated.
        self.oversample_codepoints = dict(oversample_codepoints or {})
        self.heldout_fraction = heldout_fraction
        self.min_train_fonts_per_codepoint = min_train_fonts_per_codepoint
        self.split_seed = split_seed

        self.googlefonts = GoogleFonts(str(repo), max_fonts=canary_size)
        self._filter_fonts()
        # Deterministic ordering so the font->id mapping is reproducible.
        self.fonts = sorted(self.googlefonts.fonts, key=lambda f: str(f.path))
        self.font_to_id = {str(f.path): i for i, f in enumerate(self.fonts)}
        self.cp_to_idx = {cp: i for i, cp in enumerate(self.character_set)}
        self.cp_list = list(self.character_set)

        self.num_fonts = len(self.fonts)
        self.num_codepoints = len(self.character_set)

        self._build_pairs()
        print(f"Fonts: {self.num_fonts}; codepoints: {self.num_codepoints}")
        print(
            f"Pairs: {len(self.train_pairs)} train / "
            f"{len(self.val_pairs)} held-out"
        )

    def _filter_fonts(self) -> None:
        needed = set(self.character_set)
        self.googlefonts.fonts = [
            f
            for f in self.googlefonts.fonts
            if len(set(f.codepoints) & needed) >= self.min_train_fonts_per_codepoint + 1
        ]

    def _build_pairs(self) -> None:
        rng = random.Random(self.split_seed)

        # Enumerate every (font, codepoint) with a non-empty outline, grouped by
        # codepoint so we can respect the per-codepoint training floor.
        cp_to_fonts: dict[int, list[int]] = defaultdict(list)
        for font_id, font in enumerate(self.fonts):
            hb_font = _hb_font_for_face(font.hb_face)
            for cp in sorted(set(font.codepoints) & set(self.character_set)):
                gid = hb_font.get_nominal_glyph(cp)
                extents = hb_font.get_glyph_extents(gid)
                if _has_non_empty_outline(extents):
                    cp_to_fonts[self.cp_to_idx[cp]].append(font_id)

        self.train_pairs: list[tuple[int, int]] = []
        self.val_pairs: list[tuple[int, int]] = []
        for cp_idx, font_ids in cp_to_fonts.items():
            cp = self.cp_list[cp_idx]
            if cp in self.oversample_codepoints:
                # Rare/acceptance codepoints: train on every font that has the
                # glyph (no held-out), duplicated by the oversample factor.
                factor = self.oversample_codepoints[cp]
                for fid in font_ids:
                    self.train_pairs.extend([(fid, cp_idx)] * factor)
                continue

            n = len(font_ids)
            max_holdout = max(0, n - self.min_train_fonts_per_codepoint)
            n_holdout = min(int(self.heldout_fraction * n), max_holdout)
            holdout = set(rng.sample(font_ids, n_holdout)) if n_holdout else set()
            for fid in font_ids:
                pair = (fid, cp_idx)
                if fid in holdout:
                    self.val_pairs.append(pair)
                else:
                    self.train_pairs.append(pair)

    def _loader(self, pairs: list[tuple[int, int]], shuffle: bool):
        dataset = _PairDataset(pairs, self.fonts, self.cp_list, self.image_size)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=True,
            collate_fn=_collate_fn,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
        )

    def train_loader(self):
        return self._loader(self.train_pairs, shuffle=True)

    def val_loader(self):
        return self._loader(self.val_pairs, shuffle=True)
