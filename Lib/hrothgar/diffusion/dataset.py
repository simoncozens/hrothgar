"""Class-conditional glyph dataset for Phase 1 of the diffusion generator.

Each item is a ``(grayscale glyph, class id)`` pair.  The class id folds
**codepoint x style-axis position** into a single label:

    class_id = codepoint_idx * num_rond + rond_idx

This is the fastest path to answer "is diffusion capable of rendering fine style
detail" — the model is *told* the style (ROND value) rather than having to infer
it from evidence glyphs.  Phase 2 will replace this label with an exemplar
feature map + cross-attention.

The dataset is intentionally small and render-on-demand; the canary precomputes
the whole thing into tensors (see :func:`materialize`) so epochs cost nothing and
the fixed target glyph is guaranteed to appear.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch.utils.data import Dataset as TorchDataset

RenderFn = Callable[[int, int], torch.Tensor]  # (codepoint, size) -> (H, W)


@dataclass(frozen=True)
class RONDVocab:
    """Maps ``(codepoint, rond value)`` <-> a single class id."""

    codepoints: tuple[int, ...]
    rond_values: tuple[int, ...]

    @property
    def cp_to_idx(self) -> dict[int, int]:
        return {cp: i for i, cp in enumerate(self.codepoints)}

    @property
    def rond_to_idx(self) -> dict[int, int]:
        return {r: i for i, r in enumerate(self.rond_values)}

    @property
    def num_classes(self) -> int:
        return len(self.codepoints) * len(self.rond_values)

    def encode(self, codepoint: int, rond_value: int) -> int:
        return self.cp_to_idx[codepoint] * len(self.rond_values) + self.rond_to_idx[rond_value]

    def decode(self, class_id: int) -> tuple[int, int]:
        """Return ``(codepoint, rond_value)``."""
        n = len(self.rond_values)
        cp = self.codepoints[class_id // n]
        rond = self.rond_values[class_id % n]
        return cp, rond


class ClassConditionalGlyphDataset(TorchDataset):
    """Yields ``(image (1, H, W), class_id)`` from a list of render samples."""

    def __init__(
        self,
        samples: Sequence[tuple[RenderFn, int, int]],
        image_size: int,
        channels: int = 1,
    ) -> None:
        self.samples = list(samples)
        self.image_size = image_size
        self.channels = channels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        render_fn, codepoint, class_id = self.samples[idx]
        img = render_fn(codepoint, self.image_size)  # (H, W)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        return img, class_id


def build_rond_dataset(
    render_fns: Sequence[RenderFn],
    rond_values: Sequence[int],
    glyphs: Sequence[int],
    image_size: int,
) -> tuple[ClassConditionalGlyphDataset, RONDVocab]:
    """Build a class-conditional dataset over every ``(glyph, ROND)`` pair.

    Args:
        render_fns: one renderer per ROND value (``len`` must match
            ``rond_values``).
        rond_values: the ROND values, one per renderer.
        glyphs: codepoints to render (all are trained).
        image_size: render side length.

    Returns:
        ``(dataset, vocab)`` where ``vocab`` maps codepoint/ROND <-> class id.
    """
    assert len(render_fns) == len(rond_values), "one renderer per ROND value"
    vocab = RONDVocab(tuple(sorted(glyphs)), tuple(sorted(rond_values)))
    samples: list[tuple[RenderFn, int, int]] = []
    for rond_value, render_fn in zip(sorted(rond_values), render_fns):
        for cp in sorted(glyphs):
            samples.append((render_fn, cp, vocab.encode(cp, rond_value)))
    return ClassConditionalGlyphDataset(samples, image_size), vocab


def materialize(dataset: ClassConditionalGlyphDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Render the whole dataset into ``(images, class_ids)`` tensors.

    ``images`` is ``(N, C, H, W)`` in ``[0, 1]``; ``class_ids`` is ``(N,)`` long.
    """
    images = []
    class_ids = []
    for img, class_id in dataset:
        images.append(img)
        class_ids.append(class_id)
    return torch.stack(images), torch.tensor(class_ids, dtype=torch.long)
