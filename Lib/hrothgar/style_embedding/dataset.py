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
import torch.nn.functional as F
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
        phrase_width: int = 768,
        phrase_height: int = 128,
        phrase_font_size: int = 72,
        split_seed: int = 1234,
        canary_size: Optional[int] = None,
        tag_names: Optional[list[str]] = None,
        tag_num_classes: int = 0,
        class_balanced: bool = True,
        text_encoder_name: Optional[str] = None,
        text_embedding_dim: int = 384,
    ):
        super().__init__(
            repo_url=str(repo_url),
            batch_size=batch_size,
            image_size=phrase_width,  # base class stores this; we use our own params
            split_seed=split_seed,
            canary_size=canary_size,
            character_set=list(LATIN_CORE),
        )
        self._tag_names = tag_names or []
        self._tag_num_classes = tag_num_classes
        self._phrase_width = phrase_width
        self._phrase_height = phrase_height
        self._phrase_font_size = phrase_font_size
        self._class_balanced = class_balanced
        self._text_encoder_name = text_encoder_name
        self._text_embedding_dim = text_embedding_dim
        self._text_embeddings: dict[str, torch.Tensor] = {}

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

        # Pre-compute text embeddings after filtering so we only encode
        # fonts that will actually be used.
        if self._text_encoder_name:
            self._precompute_text_embeddings()

    def _precompute_text_embeddings(self) -> None:
        """Encode ``description_with_tags()`` for every font family.

        Descriptions are shared across weights of the same static family, so
        we encode once per *family* and map each font file path to its
        family's embedding.  Results are cached to disk for reuse.
        """
        assert self._text_encoder_name is not None

        # Include model name in cache key so switching encoders
        # invalidates the cache.
        safe_name = self._text_encoder_name.replace("/", "_").replace("-", "_")
        repo = Path(self._repo_url)
        cache_path = repo / f"text_embeddings_{safe_name}.pt"
        if cache_path.exists():
            print(f"Loading cached text embeddings from {cache_path}")
            loaded = torch.load(cache_path, map_location="cpu", weights_only=True)
            if isinstance(loaded, dict):
                self._text_embeddings = loaded
                return

        from transformers import AutoTokenizer, AutoModel

        print(f"Encoding text descriptions with {self._text_encoder_name} …")
        tokenizer = AutoTokenizer.from_pretrained(self._text_encoder_name)
        model = AutoModel.from_pretrained(self._text_encoder_name).eval()

        # Collect unique descriptions per family.
        family_descs: dict[str, str] = {}
        for font in self.googlefonts.fonts:
            if font.family not in family_descs:
                family_descs[font.family] = font.description_with_tags()

        family_embs: dict[str, torch.Tensor] = {}
        zero = torch.zeros(self._text_embedding_dim)
        for family, desc in family_descs.items():
            if not desc:
                family_embs[family] = zero.clone()
                continue
            with torch.no_grad():
                tok = tokenizer(
                    desc, return_tensors="pt",
                    truncation=True, max_length=512,
                    padding=True,
                )
                out = model(**tok)
                # Mean pooling over tokens, excluding padding.
                mask = tok["attention_mask"].unsqueeze(-1)  # (1, L, 1)
                emb = (out.last_hidden_state * mask).sum(dim=1)  # (1, D)
                emb = emb / mask.sum(dim=1).clamp(min=1)
                emb = F.normalize(emb, p=2, dim=-1)
                family_embs[family] = emb.squeeze(0).cpu()

        # Map each font file path → its family's embedding.
        for font in self.googlefonts.fonts:
            emb = family_embs.get(font.family, zero.clone())
            self._text_embeddings[str(font.path)] = emb

        torch.save(self._text_embeddings, cache_path)
        print(
            f"Encoded {len(family_embs)} families "
            f"→ {len(self._text_embeddings)} font files; "
            f"cached to {cache_path}"
        )

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
                num_workers=16,
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
                self._phrase_width, self._phrase_height,
            )
            img2 = _render_and_resize(
                font, p2, self._phrase_font_size, axis2,
                self._phrase_width, self._phrase_height,
            )
            images_v1.append(img1)
            images_v2.append(img2)

        images = torch.cat([
            torch.stack(images_v1),
            torch.stack(images_v2),
        ], dim=0)  # (2B, 3, H, W)

        # Apply colour jitter for augmentation — prevents memorisation of
        # exact pixel values.
        images = _augment_batch(images)

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

        result: dict = {
            "images": images,
            "tags": tags,
            "tag_masks": tag_masks,
            "category": categories,
            "family": families,
        }

        # Attach pre-computed text embeddings keyed by font path.
        if self._text_embeddings:
            text_embs = [
                self._text_embeddings.get(
                    str(font.path),
                    torch.zeros(self._text_embedding_dim),
                )
                for font in fonts
            ]
            result["text_embeddings"] = torch.stack(text_embs)  # (B, D)

        return result


def _render_and_resize(
    font: GoogleFont,
    phrase: str,
    font_size: int,
    axis_position,
    phrase_width: int,
    phrase_height: int,
) -> torch.Tensor:
    """Render a phrase in a font and resize to a target (W, H) RGB tensor."""
    import torchvision.transforms.functional as TF

    arr = render_phrase(
        str(font.path), phrase,
        size=font_size,
        axis_position=axis_position if axis_position and len(axis_position) > 0 else None,
        width=phrase_width,
        height=phrase_height,
    )
    # arr is (H, W, 4) RGBA uint8 — strip alpha for RGB.
    img = torch.from_numpy(arr[..., :3].copy()).permute(2, 0, 1).float() / 255.0
    return img


def _augment_batch(images: torch.Tensor) -> torch.Tensor:
    """Apply colour jitter to prevent memorisation of exact pixel values.

    Each image in the batch gets independent jitter so the two views of the
    same font differ in brightness/contrast as well as phrase text.
    """
    import torchvision.transforms.functional as TF

    B = images.shape[0]
    out = []
    for i in range(B):
        img = images[i]
        # Random brightness/contrast jitter.
        b = float(torch.empty(1).uniform_(0.85, 1.15).item())
        c = float(torch.empty(1).uniform_(0.85, 1.15).item())
        img = TF.adjust_brightness(img, b)
        img = TF.adjust_contrast(img, c)
        out.append(img)
    return torch.stack(out)


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
