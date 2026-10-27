"""Dataset and collation for font style embedding training (glyph-set input).

Each batch item is one font.  The collate renders the fixed glyph set for each
font and produces two masked views (random visible-glyph subsets) so the
contrastive objective learns invariance to *which* glyphs are shown.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader
from torch.utils.data import Dataset as TorchDataset

from hrothgar.dataset import ClassBalancedBatchSampler, DatasetMaker, fvar_axes
from hrothgar.dataset_constants import LATIN_CORE
from hrothgar.googlefonts import ALL_CATEGORIES, GoogleFont
from hrothgar.render import frame_for_codepoints, render
from hrothgar.similarity.config import DEFAULT_INPUT_CODEPOINTS

# Canonical style-category and theme tags used for tag-weighted
# multi-positive contrastive examples.  These capture the major stylistic
# dimensions along which fonts vary, and are treated as a continuous
# vector (centile values normalised to [0, 1]).
STYLE_CATEGORY_TAGS = [
    "/Sans/Geometric",
    "/Sans/Glyphic",
    "/Sans/Grotesque",
    "/Sans/Humanist",
    "/Sans/Neo Grotesque",
    "/Sans/Rounded",
    "/Sans/Superellipse",
    "/Script/Formal",
    "/Script/Handwritten",
    "/Script/Informal",
    "/Script/Upright Script",
    "/Serif/Didone",
    "/Serif/Fat Face",
    "/Serif/Humanist Venetian",
    "/Serif/Modern",
    "/Serif/Old Style Garalde",
    "/Serif/Scotch",
    "/Serif/Transitional",
    "/Slab/Clarendon",
    "/Slab/Geometric",
    "/Slab/Humanist",
]

STYLE_PARENT_TAGS = ["/Sans", "/Script", "/Serif", "/Slab"]

THEME_TAGS = [
    "/Theme/Art Deco",
    "/Theme/Art Nouveau",
    "/Theme/Blackletter",
    "/Theme/Blobby",
    "/Theme/Brush",
    "/Theme/Distressed",
    "/Theme/Inline",
    "/Theme/Medieval",
    "/Theme/Pixel",
    "/Theme/Shaded",
    "/Theme/Stencil",
    "/Theme/Techno",
    "/Theme/Tuscan",
    "/Theme/Wacky",
    "/Theme/Woodtype",
]

ALL_STYLE_TAGS = STYLE_CATEGORY_TAGS + STYLE_PARENT_TAGS + THEME_TAGS


def style_tag_vector(font) -> torch.Tensor:
    """Per-font centile style-tag vector over the *fine* style tags.

    Uses only the leaf tags (``STYLE_CATEGORY_TAGS`` + ``THEME_TAGS``).  The
    coarse parent tags (``STYLE_PARENT_TAGS``, e.g. ``/Sans``) are deliberately
    excluded: they are shared by every font in a broad class, so they make
    "any two sans" look alike and erode the fine sub-style distinctions
    (glyphic vs grotesque, Didone vs Transitional, …).  Shared by the training
    collate (tag-weighted contrastive positives) and the eval panel (style-tag
    retrieval), so the two can never drift apart.
    """
    font_tags = font.tags()
    fine_tags = STYLE_CATEGORY_TAGS + THEME_TAGS
    return torch.tensor(
        [font_tags.get(tag, 0.0) / 100.0 for tag in fine_tags],
        dtype=torch.float32,
    )


class _FontDataset(TorchDataset):
    """Yields one font per item, optionally at a sampled variable-axis position."""

    def __init__(
        self,
        fonts: Sequence[GoogleFont],
        axes_by_path: dict | None = None,
        sample_axis: bool = False,
    ):
        self.fonts = list(fonts)
        self._axes_by_path = axes_by_path or {}
        self._sample_axis = sample_axis

    def __len__(self) -> int:
        return len(self.fonts)

    def __getitem__(self, idx: int) -> dict:
        font = self.fonts[idx]
        position = None
        if self._sample_axis:
            axes = self._axes_by_path.get(str(font.path))
            if axes:
                position = _sample_axis_position(axes, random)
        return {"font": font, "axis_position": position}


_AXIS_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def _sample_axis_position(axes: list[list], rng) -> list[float]:
    """Sample design coordinates in fvar order for a variable font.

    Half the time the font's default instance; otherwise one randomly chosen
    axis is moved to a grid point.  Perturbing a single axis avoids implausible
    multi-axis combinations, and the small grid (rather than continuous
    sampling) keeps the renderer's FreeType face cache effective.
    """
    default = [a[2] for a in axes]
    if rng.random() < 0.5:
        return default
    tag, lo, _default, hi = axes[rng.randrange(len(axes))]
    if hi <= lo:
        return default
    value = lo + rng.choice(_AXIS_GRID) * (hi - lo)
    return [value if a[0] == tag else a[2] for a in axes]


def _balance_key(name: str):
    """Batch-sampler grouping key: ``"family"``, ``"category"`` or neither."""
    if name == "family":
        return lambda font: font.family
    if name == "category":
        return lambda font: font.category()
    return None


class SiblingBatchSampler(BatchSampler):
    """Cap each family's per-batch contribution while keeping siblings together.

    ``ClassBalancedBatchSampler`` with ``key=family`` puts at most *one* font per
    family in each batch.  That makes a same-family soft-positive contrastive term
    identically zero, because a batch never contains a same-family,
    different-font pair.  This sampler instead draws up to ``siblings_per_family``
    fonts from each family per batch, so the family objective has siblings to pull
    together while large multi-weight families still cannot dominate a batch.
    """

    def __init__(
        self,
        items: Sequence[GoogleFont],
        *,
        key,
        batch_size: int,
        siblings_per_family: int,
        drop_last: bool,
        rng=None,
    ):
        if siblings_per_family < 1:
            raise ValueError(f"siblings_per_family must be >= 1, got {siblings_per_family}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if len(items) == 0:
            raise ValueError("Cannot build sibling sampler for empty dataset")

        self.batch_size = batch_size
        self.siblings_per_family = siblings_per_family
        self.drop_last = drop_last
        self._rng = rng if rng is not None else random

        self._groups: dict[str, list[int]] = {}
        for idx, item in enumerate(items):
            self._groups.setdefault(key(item), []).append(idx)
        self._families = list(self._groups.keys())
        self._total = len(items)

    def __len__(self) -> int:
        if self.drop_last:
            return self._total // self.batch_size
        return math.ceil(self._total / self.batch_size)

    def __iter__(self):
        # Partition each family's (shuffled) indices into chunks of size at most
        # ``siblings_per_family``, then pack the chunks into batches.  Every font
        # is emitted exactly once per epoch, and a family contributes at most one
        # chunk (i.e. ``siblings_per_family`` fonts) at a time, so siblings
        # coexist while large multi-weight families cannot dominate a batch.
        chunks: list[list[int]] = []
        for family in self._families:
            indices = self._groups[family][:]
            self._rng.shuffle(indices)
            for start in range(0, len(indices), self.siblings_per_family):
                chunks.append(indices[start : start + self.siblings_per_family])
        self._rng.shuffle(chunks)

        batch: list[int] = []
        for chunk in chunks:
            batch.extend(chunk)
            while len(batch) >= self.batch_size:
                yield batch[: self.batch_size]
                batch = batch[self.batch_size :]
        if batch and not self.drop_last:
            yield batch


class FontStyleDatasetMaker(DatasetMaker):
    """Dataset maker for font-level style embedding (glyph-set input)."""

    def __init__(
        self,
        repo_url: str | Path,
        batch_size: int,
        *,
        glyph_size: int = 128,
        input_codepoints: Sequence[int] | None = None,
        glyph_sample_size: int = 32,
        split_seed: int = 1234,
        canary_size: int | None = None,
        tag_names: list[str] | None = None,
        tag_num_classes: int = 0,
        balance_by: str = "family",
        siblings_per_family: int = 1,
        sample_axis: bool = False,
        text_encoder_name: str | None = None,
        text_embedding_dim: int = 384,
    ):
        self._text_encoder_name = text_encoder_name
        self._text_embedding_dim = text_embedding_dim
        self._text_embeddings: dict[str, torch.Tensor] = {}
        self._input_codepoints = (
            list(input_codepoints)
            if input_codepoints is not None
            else list(DEFAULT_INPUT_CODEPOINTS)
        )
        self._glyph_size = glyph_size
        self._glyph_sample_size = glyph_sample_size

        super().__init__(
            repo_url=str(repo_url),
            batch_size=batch_size,
            image_size=glyph_size,
            split_seed=split_seed,
            canary_size=canary_size,
            character_set=list(LATIN_CORE),
        )
        self._tag_names = tag_names or []
        self._tag_num_classes = tag_num_classes
        self._balance_by = balance_by
        self._siblings_per_family = siblings_per_family

        # Variable fonts contribute many instances: record their axes so each
        # item can be rendered at a sampled design position.
        self._axes_by_path: dict[str, list] = {}
        self._sample_axis = sample_axis
        if sample_axis:
            for font in (*self.train_fonts, *self.test_fonts):
                axes = fvar_axes(font.path)
                if axes:
                    self._axes_by_path[str(font.path)] = axes

        if self._text_encoder_name:
            self._precompute_text_embeddings()

    def filter_fonts(self) -> None:
        """Remove fonts that don't have every glyph in the input set."""
        needed = set(self._input_codepoints)
        self.googlefonts.fonts = [
            font for font in self.googlefonts.fonts if needed <= font.codepoints
        ]

    def _precompute_text_embeddings(self) -> None:
        """Encode ``description_with_tags()`` for every font family."""
        assert self._text_encoder_name is not None

        safe_name = self._text_encoder_name.replace("/", "_").replace("-", "_")
        repo = self.googlefonts.repo_path
        cache_path = repo / f"text_embeddings_{safe_name}.pt"
        if cache_path.exists():
            print(f"Loading cached text embeddings from {cache_path}")
            loaded = torch.load(cache_path, map_location="cpu", weights_only=True)
            if isinstance(loaded, dict):
                if loaded and all(
                    isinstance(v, torch.Tensor)
                    and not v.isnan().any()
                    and v.norm(p=2) > 0
                    for v in loaded.values()
                ):
                    self._text_embeddings = loaded
                    print(f"  → {len(self._text_embeddings)} embeddings loaded")
                    return
                else:
                    print("  → Cache invalid (NaN or zero vector); re-encoding.")

        from transformers import AutoModel, AutoTokenizer

        print(f"Encoding text descriptions with {self._text_encoder_name} …")
        tokenizer = AutoTokenizer.from_pretrained(self._text_encoder_name)
        model = AutoModel.from_pretrained(self._text_encoder_name).eval()

        family_descs: dict[str, str] = {}
        for font in self.googlefonts.fonts:
            if font.family not in family_descs:
                family_descs[font.family] = font.description_with_tags()

        family_embs: dict[str, torch.Tensor] = {}
        for family, desc in family_descs.items():
            if not desc.strip():
                v = torch.randn(self._text_embedding_dim)
                family_embs[family] = F.normalize(v, p=2, dim=-1)
                continue
            with torch.no_grad():
                tok = tokenizer(
                    desc,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                )
                out = model(**tok)
                mask = tok["attention_mask"].unsqueeze(-1)
                emb = (out.last_hidden_state * mask).sum(dim=1)
                emb = emb / mask.sum(dim=1).clamp(min=1)
                emb = F.normalize(emb, p=2, dim=-1)
                family_embs[family] = emb.squeeze(0).cpu()

        for font in self.googlefonts.fonts:
            emb = family_embs.get(font.family)
            if emb is None:
                v = torch.randn(self._text_embedding_dim)
                emb = F.normalize(v, p=2, dim=-1)
            self._text_embeddings[str(font.path)] = emb

        torch.save(self._text_embeddings, cache_path)
        print(
            f"Encoded {len(family_embs)} families "
            f"→ {len(self._text_embeddings)} font files; "
            f"cached to {cache_path}"
        )

    def train_set(self):
        return _FontDataset(self.train_fonts, self._axes_by_path, self._sample_axis)

    def test_set(self):
        # Validation renders on the default instance so the metric is stable
        # and comparable across runs; axis sampling is a training augmentation.
        return _FontDataset(self.test_fonts, self._axes_by_path, sample_axis=False)

    def train_loader(self):
        dataset = self.train_set()
        key = _balance_key(self._balance_by)
        if key is not None:
            if self._balance_by == "family" and self._siblings_per_family > 1:
                sampler: BatchSampler = SiblingBatchSampler(
                    self.train_fonts,
                    key=key,
                    batch_size=self.batch_size,
                    siblings_per_family=self._siblings_per_family,
                    drop_last=True,
                )
            else:
                sampler = ClassBalancedBatchSampler(
                    self.train_fonts,
                    key=key,
                    batch_size=self.batch_size,
                    drop_last=True,
                )
            return DataLoader(
                dataset,
                batch_sampler=sampler,
                collate_fn=self.collate_fn,
                num_workers=16,
                pin_memory=True,
            )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fn,
            num_workers=16,
            pin_memory=True,
        )

    def collate_fn(self, batch: list[dict]) -> dict:
        """Collate font-level items into a training batch.

        Returns:
            - ``images``: ``(2B, s, 1, H, W)`` — two disjoint masked views stacked.
            - ``glyph_mask``: ``(2B, s)`` boolean visibility mask.
            - tags / category / family / tag_vectors / text_embeddings for the
              B fonts (duplicated to 2B by the training loop).
        """
        fonts: list[GoogleFont] = [item["font"] for item in batch]
        positions = [item.get("axis_position") for item in batch]
        b = len(fonts)
        g = len(self._input_codepoints)  # full glyph vocabulary
        s = self._glyph_sample_size
        size = self._glyph_size

        glyphs = torch.zeros(b, s, 1, size, size, dtype=torch.float32)
        mask_a = torch.zeros(b, s, dtype=torch.bool)
        mask_b = torch.zeros(b, s, dtype=torch.bool)

        for i, font in enumerate(fonts):
            # Sample s glyphs (indices into the full vocabulary) without replacement.
            idx = random.sample(range(g), s)
            idx.sort()
            # Split positions into two disjoint halves for the two views.
            perm = random.sample(range(s), s)
            half = s // 2
            half_a = perm[:half]
            half_b = perm[half:]

            # A per-font tight frame over the whole input set, so the same
            # letter occupies comparable pixels in every font instead of
            # shrinking with the design's x-height.
            frame = frame_for_codepoints(font, self._input_codepoints)

            for j, cp_idx in enumerate(idx):
                glyph = render(
                    font,
                    codepoint=self._input_codepoints[cp_idx],
                    size=size,
                    axis_position=positions[i],
                    frame=frame,
                )[0]
                glyphs[i, j, 0] = glyph

            mask_a[i, half_a] = True
            mask_b[i, half_b] = True

        # Zero-out invisible glyphs so the reconstruction target is never leaked
        # into the encoder input.
        view_a = glyphs * mask_a[:, :, None, None, None]
        view_b = glyphs * mask_b[:, :, None, None, None]

        images = torch.cat([view_a, view_b], dim=0)  # (2B, s, 1, H, W)
        glyph_mask = torch.cat([mask_a, mask_b], dim=0)  # (2B, s)

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
        # Unique per font file — distinguishes different weights/styles of the
        # same family for the font-level (soft family grouping) contrastive.
        font_ids = [str(font.path) for font in fonts]

        # Per-font tag vectors for tag-weighted contrastive positives.
        tag_vectors = torch.stack([style_tag_vector(font) for font in fonts])  # (B, num_style_tags)

        result: dict = {
            "images": images,
            "glyph_mask": glyph_mask,
            "tags": tags,
            "tag_masks": tag_masks,
            "category": categories,
            "family": families,
            "font_ids": font_ids,
            "tag_vectors": tag_vectors,
        }

        if self._text_embeddings:
            text_embs = []
            for font in fonts:
                emb = self._text_embeddings.get(str(font.path))
                if emb is None:
                    v = torch.randn(self._text_embedding_dim)
                    emb = F.normalize(v, p=2, dim=-1)
                text_embs.append(emb)
            result["text_embeddings"] = torch.stack(text_embs)  # (B, D)

        return result
