"""Dataset and collation for font style embedding training (glyph-set input).

Each batch item is one font.  The collate renders the fixed glyph set for each
font and produces two masked views (random visible-glyph subsets) so the
contrastive objective learns invariance to *which* glyphs are shown.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader
from torch.utils.data import Dataset as TorchDataset

from hrothgar.dataset import DatasetMaker
from hrothgar.dataset_constants import LATIN_CORE
from hrothgar.googlefonts import GoogleFont, ALL_CATEGORIES
from hrothgar.style_embedding.config import DEFAULT_INPUT_CODEPOINTS
from hrothgar.style_embedding.render_utils import render_glyph


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


class _FontDataset(TorchDataset):
    """Yields one font per item (no per-glyph expansion)."""

    def __init__(self, fonts: Sequence[GoogleFont]):
        self.fonts = list(fonts)

    def __len__(self) -> int:
        return len(self.fonts)

    def __getitem__(self, idx: int) -> dict:
        return {"font": self.fonts[idx]}


class FontStyleDatasetMaker(DatasetMaker):
    """Dataset maker for font-level style embedding (glyph-set input)."""

    def __init__(
        self,
        repo_url: str | Path,
        batch_size: int,
        *,
        glyph_size: int = 64,
        input_codepoints: Optional[Sequence[int]] = None,
        glyph_sample_size: int = 32,
        split_seed: int = 1234,
        canary_size: Optional[int] = None,
        tag_names: Optional[list[str]] = None,
        tag_num_classes: int = 0,
        class_balanced: bool = True,
        text_encoder_name: Optional[str] = None,
        text_embedding_dim: int = 384,
    ):
        self._text_encoder_name = text_encoder_name
        self._text_embedding_dim = text_embedding_dim
        self._text_embeddings: dict[str, torch.Tensor] = {}
        self._input_codepoints = list(input_codepoints) if input_codepoints is not None else list(DEFAULT_INPUT_CODEPOINTS)
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
        self._class_balanced = class_balanced

        if self._text_encoder_name:
            self._precompute_text_embeddings()

    def filter_fonts(self) -> None:
        """Remove fonts that don't have every glyph in the input set."""
        needed = set(self._input_codepoints)
        self.googlefonts.fonts = [
            font
            for font in self.googlefonts.fonts
            if needed <= font.codepoints
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

        from transformers import AutoTokenizer, AutoModel

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
                    desc, return_tensors="pt",
                    truncation=True, max_length=512,
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
        return _FontDataset(self.train_fonts)

    def test_set(self):
        return _FontDataset(self.test_fonts)

    def train_loader(self):
        dataset = self.train_set()
        if self._class_balanced:
            return DataLoader(
                dataset,
                batch_sampler=_ClassBalancedBatchSampler(
                    self.train_fonts,
                    batch_size=self.batch_size,
                    drop_last=True,
                ),
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
            - ``codepoint_indices``: ``(B, s)`` glyph-vocabulary index per sample.
            - tags / category / family / tag_vectors / text_embeddings for the
              B fonts (duplicated to 2B by the training loop).
        """
        fonts: list[GoogleFont] = [item["font"] for item in batch]
        b = len(fonts)
        g = len(self._input_codepoints)  # full glyph vocabulary
        s = self._glyph_sample_size
        size = self._glyph_size

        glyphs = torch.zeros(b, s, 1, size, size, dtype=torch.float32)
        glyph_bboxes = torch.zeros(b, s, 4, dtype=torch.float32)
        shape_targets = torch.zeros(b, s, 1, size, size, dtype=torch.float32)
        codepoint_indices = torch.zeros(b, s, dtype=torch.long)
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

            for j, cp_idx in enumerate(idx):
                glyph = render_glyph(font, self._input_codepoints[cp_idx], size)
                glyphs[i, j, 0] = glyph
                glyph_bboxes[i, j] = _ink_bbox(glyph, size)
                shape_targets[i, j, 0] = _normalize_shape(glyph, size)
                codepoint_indices[i, j] = cp_idx

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

        # Per-font tag vectors for tag-weighted contrastive positives.
        style_vectors: list[torch.Tensor] = []
        for font in fonts:
            font_tags = font.tags()
            values: dict[str, float] = {}
            for tag in ALL_STYLE_TAGS:
                values[tag] = font_tags.get(tag, 0.0) / 100.0
            for parent in STYLE_PARENT_TAGS:
                child_max = 0.0
                for child in STYLE_CATEGORY_TAGS:
                    if child.startswith(parent + "/"):
                        child_max = max(child_max, values.get(child, 0.0))
                values[parent] = child_max
            vec = torch.tensor(
                [values[tag] for tag in ALL_STYLE_TAGS],
                dtype=torch.float32,
            )
            style_vectors.append(vec)
        tag_vectors = torch.stack(style_vectors)  # (B, num_style_tags)

        result: dict = {
            "images": images,
            "glyph_mask": glyph_mask,
            "codepoint_indices": codepoint_indices,
            "target_glyphs": glyphs,
            "glyph_bboxes": glyph_bboxes,
            "shape_targets": shape_targets,
            "tags": tags,
            "tag_masks": tag_masks,
            "category": categories,
            "family": families,
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


def _ink_bbox(glyph: torch.Tensor, size: int) -> torch.Tensor:
    """Return the glyph's ink bounding box as normalized ``(x0, y0, x1, y1)``.

    Ink is rendered near 0.0 on a white (1.0) background, so we threshold at
    0.5 to recover the glyph extent.  Blank glyphs return an all-zero box.
    """
    ink = glyph < 0.5
    if not ink.any():
        return torch.zeros(4)
    ys, xs = ink.nonzero(as_tuple=True)
    x0 = xs.min().float()
    x1 = xs.max().float()
    y0 = ys.min().float()
    y1 = ys.max().float()
    return torch.tensor([x0, y0, x1, y1], dtype=torch.float32) / size


def _normalize_shape(glyph: torch.Tensor, size: int) -> torch.Tensor:
    """Crop a glyph to its ink bbox and stretch it to a ``(size, size)`` square.

    This removes absolute placement/scale while keeping the within-bbox shape.
    Aspect ratio is intentionally *not* preserved — the layout head carries the
    width/height that this normalization discards.
    """
    ink = glyph < 0.5
    if not ink.any():
        return torch.ones(size, size)
    ys, xs = ink.nonzero(as_tuple=True)
    x0 = int(xs.min().item())
    x1 = int(xs.max().item())
    y0 = int(ys.min().item())
    y1 = int(ys.max().item())
    crop = glyph[y0 : y1 + 1, x0 : x1 + 1]
    if crop.numel() == 0:
        return torch.ones(size, size)
    crop = crop[None, None, :, :]  # (1, 1, h, w)
    out = F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False)
    return out[0, 0]  # (size, size)


class _ClassBalancedBatchSampler(BatchSampler):
    """Batch sampler that balances broad font categories within each batch."""

    def __init__(
        self,
        fonts: Sequence[GoogleFont],
        *,
        batch_size: int,
        drop_last: bool,
    ) -> None:
        self.fonts = list(fonts)
        self.batch_size = batch_size
        self.drop_last = drop_last

        class_to_indices: dict[str, list[int]] = {}
        for idx, font in enumerate(self.fonts):
            cls = font.category()
            class_to_indices.setdefault(cls, []).append(idx)

        if not class_to_indices:
            raise ValueError("No classes found for class-balanced sampling")

        self.class_to_indices = class_to_indices
        self.classes = sorted(class_to_indices.keys())

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.fonts) // self.batch_size
        return math.ceil(len(self.fonts) / self.batch_size)

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
