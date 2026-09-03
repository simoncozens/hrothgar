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


def build_exemplar_rond_data(
    render_fns: Sequence[RenderFn],
    rond_values: Sequence[int],
    glyphs: Sequence[int],
    num_evidence: int,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[int, int]]:
    """Build the exemplar (many-shot) canary dataset.

    For every ``(font, target codepoint)`` pair, render ``num_evidence``
    evidence glyphs (the style references, excluding the target) and the target
    glyph.  Returns:

    * ``evidence`` ``(T, N, 1, H, W)``
    * ``target_codepoint`` ``(T,)`` long (index into the sorted glyph list)
    * ``target_image`` ``(T, 1, H, W)``
    * ``cp_to_idx`` mapping codepoint -> embedding index
    """
    from hrothgar.style_extraction.render_utils import render_glyph

    sorted_glyphs = sorted(glyphs)
    cp_to_idx = {cp: i for i, cp in enumerate(sorted_glyphs)}

    evidence_list: list[torch.Tensor] = []
    cp_list: list[int] = []
    target_list: list[torch.Tensor] = []

    for render_fn in render_fns:
        for cp in sorted_glyphs:
            ev_cps = [g for g in sorted_glyphs if g != cp][:num_evidence]
            ev = torch.stack([render_fn(g, image_size) for g in ev_cps])  # (N, H, W)
            evidence_list.append(ev.unsqueeze(1))  # (N, 1, H, W)
            cp_list.append(cp_to_idx[cp])
            target_list.append(render_fn(cp, image_size).unsqueeze(0))  # (1, H, W)

    evidence = torch.stack(evidence_list)
    codepoints = torch.tensor(cp_list, dtype=torch.long)
    targets = torch.stack(target_list)
    return evidence, codepoints, targets, cp_to_idx


def build_fontid_rond_data(
    render_fns: Sequence[RenderFn],
    glyphs: Sequence[int],
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[int, int]]:
    """Build the factorized (codepoint, font-ID) canary dataset.

    For every ``(font, codepoint)`` pair, render the glyph.  Returns:

    * ``images`` ``(T, 1, H, W)``
    * ``codepoints`` ``(T,)`` long (index into the sorted glyph list)
    * ``font_ids`` ``(T,)`` long (index into ``render_fns``)
    * ``cp_to_idx`` mapping codepoint -> embedding index
    """
    from hrothgar.style_extraction.render_utils import render_glyph

    sorted_glyphs = sorted(glyphs)
    cp_to_idx = {cp: i for i, cp in enumerate(sorted_glyphs)}

    images: list[torch.Tensor] = []
    cp_list: list[int] = []
    fid_list: list[int] = []
    for fid, render_fn in enumerate(render_fns):
        for cp in sorted_glyphs:
            images.append(render_fn(cp, image_size).unsqueeze(0))  # (1, H, W)
            cp_list.append(cp_to_idx[cp])
            fid_list.append(fid)

    return (
        torch.stack(images),
        torch.tensor(cp_list, dtype=torch.long),
        torch.tensor(fid_list, dtype=torch.long),
        cp_to_idx,
    )
