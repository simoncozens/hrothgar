"""Font-level dataset for style-extraction (autoencoding) training.

Each batch item is one font.  The collate renders:

- **Evidence glyphs**: a random sample of the font's codepoints — the style
  evidence the encoder pools into a style map.
- **Target glyph**: a single codepoint (excluded from the evidence) plus its
  ground-truth rendering — the reconstruction target.

Generalization is tested along the **font** axis (train/test families).  The
codepoint is always drawn from the full vocabulary so every codepoint embedding
row is trained; holding codepoints out would leave their embedding rows at
random init and make them unrenderable.
"""

from __future__ import annotations
import os

import random
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from hrothgar.dataset import ClassBalancedBatchSampler, DatasetMaker
from hrothgar.googlefonts import GoogleFont
from hrothgar.style_extraction.render_utils import render_glyph

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))


class _FontDataset(TorchDataset):
    """Yields one font per item (no per-glyph expansion)."""

    def __init__(self, fonts: Sequence[GoogleFont]):
        self.fonts = list(fonts)

    def __len__(self) -> int:
        return len(self.fonts)

    def __getitem__(self, idx: int) -> dict:
        return {"font": self.fonts[idx]}


class _PrecomputedBatches:
    """Fixed, pre-rendered batches (canary mode).

    Implements just the DataLoader surface the training loop touches
    (``len()`` and iteration) but yields prebuilt CPU tensors, so an epoch
    costs no worker spawn and no rendering — and every epoch sees the exact
    same data.
    """

    def __init__(self, batches: list[dict]):
        self._batches = batches

    def __len__(self) -> int:
        return len(self._batches)

    def __iter__(self):
        return iter(self._batches)


class StyleExtractionDatasetMaker(DatasetMaker):
    """Dataset maker for the style autoencoder (font-level items)."""

    def __init__(
        self,
        repo_url: str | Path,
        batch_size: int,
        *,
        image_size: int = 128,
        character_set: Optional[Sequence[int]] = None,
        num_evidence_glyphs: int = 32,
        split_seed: int = 1234,
        canary_size: Optional[int] = None,
        class_balanced: bool = True,
    ):
        # Set before super().__init__: the base class calls filter_fonts(),
        # which needs this attribute.
        self._num_evidence_glyphs = num_evidence_glyphs
        self._class_balanced = class_balanced

        super().__init__(
            repo_url=str(repo_url),
            batch_size=batch_size,
            image_size=image_size,
            split_seed=split_seed,
            canary_size=canary_size,
            character_set=character_set,
        )

        # Codepoint -> embedding index (must match model codepoint table order).
        self._cp_to_idx = {cp: i for i, cp in enumerate(self._character_set)}

        # Populated on first train_loader() call in canary mode.
        self._canary_loader: Optional[_PrecomputedBatches] = None

    def filter_fonts(self) -> None:
        """Keep only fonts with enough glyphs to split evidence from target."""
        needed = set(self._character_set)
        self.googlefonts.fonts = [
            font
            for font in self.googlefonts.fonts
            if len(set(font.codepoints) & needed) >= self._num_evidence_glyphs + 1
        ]

    def train_set(self):
        return _FontDataset(self.train_fonts)

    def test_set(self):
        return _FontDataset(self.test_fonts)

    def _render_batch(self, fonts: list[GoogleFont], rng: random.Random) -> dict:
        """Render one batch of fonts into the tensors consumed by ``train_step``.

        ``rng`` supplies all stochastic choices (target codepoint, evidence
        sample).  Passing an explicit RNG — rather than touching the global
        ``random`` module — is what makes canary batches reproducible.
        """
        b = len(fonts)
        g = self._num_evidence_glyphs
        size = self.image_size

        style_images = torch.zeros(b, g, 1, size, size, dtype=torch.float32)
        style_codepoint_idx = torch.zeros(b, g, dtype=torch.long)
        target_images = torch.zeros(b, 1, size, size, dtype=torch.float32)
        target_codepoint_idx = torch.zeros(b, dtype=torch.long)
        target_codepoint = torch.zeros(b, dtype=torch.long)

        needed = set(self._character_set)
        for i, font in enumerate(fonts):
            avail = sorted(set(font.codepoints) & needed)
            if not avail:
                continue

            # Target codepoint: any available codepoint.  Content identity is
            # known to the model (it has seen this codepoint in other fonts);
            # the generalization axis is the *font*, handled by the family split.
            target_cp = rng.choice(avail)

            # Evidence excludes the target so the model must compose style +
            # codepoint rather than copy the target glyph from evidence.
            evidence_pool = [cp for cp in avail if cp != target_cp]
            evidence = rng.sample(evidence_pool, g)

            for j, cp in enumerate(evidence):
                style_images[i, j, 0] = render_glyph(font, cp, size)
                style_codepoint_idx[i, j] = self._cp_to_idx[cp]

            target_images[i, 0] = render_glyph(font, target_cp, size)
            target_codepoint_idx[i] = self._cp_to_idx[target_cp]
            target_codepoint[i] = target_cp

        return {
            "style_images": style_images,
            "style_codepoint_idx": style_codepoint_idx,
            "target_images": target_images,
            "target_codepoint_idx": target_codepoint_idx,
            "target_codepoint": target_codepoint,
            "family": [font.family for font in fonts],
        }

    def _collate_fn(self, batch: list[dict]) -> dict:
        fonts: list[GoogleFont] = [item["font"] for item in batch]
        return self._render_batch(fonts, rng=random)

    def _build_canary_loader(self) -> _PrecomputedBatches:
        """Render every canary batch once up front.

        The canary dataset is small (tens of fonts), so we precompute the exact
        tensors the model will see.  That makes epochs cost ~nothing (no
        DataLoader workers, no per-epoch rendering) and makes the dataset
        bit-for-bit identical across epochs: font selection (via the
        class-balanced sampler) *and* codepoint/evidence sampling all draw from
        one RNG seeded by ``split_seed``.  Using the global ``random`` module
        here would silently advance between epochs, so every epoch would see
        the same codepoints rendered in different fonts.
        """
        rng = random.Random(self.split_seed)
        fonts = self.train_fonts
        if self._class_balanced:
            sampler = ClassBalancedBatchSampler(
                fonts,
                key=lambda font: font.category(),
                batch_size=self.batch_size,
                drop_last=True,
                rng=rng,
            )
            batches = [
                self._render_batch([fonts[i] for i in indices], rng=rng)
                for indices in sampler
            ]
        else:
            order = list(range(len(fonts)))
            rng.shuffle(order)
            num_batches = len(order) // self.batch_size
            batches = [
                self._render_batch(
                    [fonts[i] for i in order[k * self.batch_size : (k + 1) * self.batch_size]],
                    rng=rng,
                )
                for k in range(num_batches)
            ]
        return _PrecomputedBatches(batches)

    def train_loader(self):
        if self.canary_size is not None:
            if self._canary_loader is None:
                self._canary_loader = self._build_canary_loader()
            return self._canary_loader
        dataset = self.train_set()
        if self._class_balanced:
            return DataLoader(
                dataset,
                batch_sampler=ClassBalancedBatchSampler(
                    self.train_fonts,
                    key=lambda font: font.category(),
                    batch_size=self.batch_size,
                    drop_last=True,
                ),
                collate_fn=self._collate_fn,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                # Keep worker processes (and their font/face caches) alive
                # across epochs instead of respawning them every epoch.
                persistent_workers=NUM_WORKERS > 0,
            )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self._collate_fn,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
        )

    def test_loader(self):
        if self.canary_size is not None:
            # Test is train — the same precomputed batches.
            return self.train_loader()
        dataset = self.test_set()
        if self._class_balanced:
            return DataLoader(
                dataset,
                batch_sampler=ClassBalancedBatchSampler(
                    self.test_fonts,
                    key=lambda font: font.category(),
                    batch_size=self.batch_size,
                    drop_last=True,
                ),
                collate_fn=self._collate_fn,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                persistent_workers=NUM_WORKERS > 0,
            )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self._collate_fn,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
        )
