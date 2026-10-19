"""Dataset for the exemplar-conditional diffusion model.

Each training item is a ``(target glyph, evidence glyph set, codepoint,
geometry)`` tuple:

* the **target** glyph is rendered crop-to-ink (:func:`render_glyph_with_geometry`),
  exactly as the factorized model consumes it, with its five geometry labels;
* the **evidence** glyphs are rendered in the uniform 2em frame
  (:func:`render_glyph_uniform`, aspect preserved — a style *reference*, not a
  diffusion I/O) and selected **content-aware** via
  :func:`~hrothgar.diffusion.exemplar_mix.draw_exemplars`: an uppercase target
  mostly sees uppercase exemplars with some lowercase, a digit sees digits plus
  letters, a symbol sees a broad mix.

Everything else — stratified subset selection, instance materialisation, the
codepoint-based train/val split, geometry-std normalisation — is inherited from
:class:`~hrothgar.diffusion.dataset_fontid.FontIdDatasetMaker`, so the two models
are trained on exactly comparable data.  The only difference is what a batch
carries: *evidence glyphs* instead of a ``(family_id, style_bucket)`` lookup.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from hrothgar.diffusion.dataset_fontid import NUM_WORKERS, FontIdDatasetMaker
from hrothgar.diffusion.exemplar_mix import DEFAULT_MIX, draw_exemplars
from hrothgar.render import (
    geometry_tensor,
    render_glyph_uniform,
    render_glyph_with_geometry,
)


class _ExemplarPairDataset(TorchDataset):
    """Yields ``(image, evidence, cp_idx, instance_id, geometry)`` for fixed pairs."""

    def __init__(
        self,
        pairs: list[tuple[int, int]],
        instances,
        cp_list,
        image_size: int,
        num_evidence_glyphs: int,
        mix: dict,
    ) -> None:
        self.pairs = pairs
        self.instances = instances
        self.cp_list = list(cp_list)
        self.image_size = image_size
        self.num_evidence_glyphs = num_evidence_glyphs
        self.mix = mix

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        instance_id, cp_idx = self.pairs[idx]
        instance = self.instances[instance_id]
        cp = self.cp_list[cp_idx]

        img, geometry = render_glyph_with_geometry(
            instance.font,
            cp,
            self.image_size,
            axis_position=instance.axis_position,
        )

        # Content-aware evidence: mostly the target's own class, with a tuned
        # admixture of the others (see exemplar_mix).  Re-drawn on every access
        # so the evidence set varies epoch to epoch (no frozen targets).
        ev_cps = draw_exemplars(
            cp,
            instance.codepoints,
            self.num_evidence_glyphs,
            mix=self.mix,
        )
        if not ev_cps:  # degenerate: a font whose only glyph is the target
            ev_cps = [cp] * self.num_evidence_glyphs

        evidence = []
        for ev_cp in ev_cps:
            ev_img, _ = render_glyph_uniform(
                instance.font,
                ev_cp,
                self.image_size,
                axis_position=instance.axis_position,
            )
            evidence.append(ev_img.unsqueeze(0))  # (1, H, W)

        return {
            "image": img.unsqueeze(0),  # (1, H, W)
            "evidence": torch.stack(evidence),  # (N, 1, H, W)
            "evidence_cps": ev_cps,  # (N,) codepoint ints, for logging
            "geometry": geometry_tensor(geometry),  # (5,)
            "cp_idx": cp_idx,
            "instance_id": instance_id,
        }


def _collate_exemplar(batch: list[dict]) -> dict:
    return {
        "images": torch.stack([b["image"] for b in batch]),
        "evidence": torch.stack([b["evidence"] for b in batch]),  # (B, N, 1, H, W)
        "evidence_cps": torch.tensor(
            [b["evidence_cps"] for b in batch], dtype=torch.long
        ),  # (B, N)
        "geometry": torch.stack([b["geometry"] for b in batch]),
        "codepoints": torch.tensor([b["cp_idx"] for b in batch], dtype=torch.long),
        "instance_ids": torch.tensor(
            [b["instance_id"] for b in batch], dtype=torch.long
        ),
    }


class ExemplarDatasetMaker(FontIdDatasetMaker):
    """Builds train/val loaders of ``(image, evidence, codepoint, geometry)``.

    Subclasses :class:`FontIdDatasetMaker` and reuses its stratified subset
    selection and codepoint-based split verbatim; only the per-item rendering and
    the batch collation differ (evidence glyphs replace the font-id lookup).
    """

    def __init__(
        self,
        repo: str | Path,
        batch_size: int,
        *,
        num_evidence_glyphs: int = 8,
        exemplar_mix: dict | None = None,
        **kwargs,
    ) -> None:
        self.num_evidence_glyphs = num_evidence_glyphs
        self.exemplar_mix = (
            exemplar_mix if exemplar_mix is not None else DEFAULT_MIX
        )
        super().__init__(repo, batch_size, **kwargs)

    def _loader(self, pairs: list[tuple[int, int]], shuffle: bool):
        dataset = _ExemplarPairDataset(
            pairs,
            self.instances,
            self.cp_list,
            self.image_size,
            self.num_evidence_glyphs,
            self.exemplar_mix,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=True,
            collate_fn=_collate_exemplar,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
        )

    def random_val_batch(self, n: int) -> dict:
        """A random (unseeded) batch of held-out pairs, for visualization only."""
        pairs = random.sample(self.val_pairs, k=min(n, len(self.val_pairs)))
        dataset = _ExemplarPairDataset(
            pairs,
            self.instances,
            self.cp_list,
            self.image_size,
            self.num_evidence_glyphs,
            self.exemplar_mix,
        )
        return _collate_exemplar([dataset[i] for i in range(len(pairs))])

    def instance_sidecar(self) -> list[dict]:
        """Instance records for inference: where to render each font's evidence.

        Unlike fontid there is no ``(family_id, style_bucket)`` lookup — the style
        is read from the evidence glyphs themselves — so the sidecar records only
        the source file and axis position.
        """
        return [
            {
                "path": inst.path,
                "family": inst.family,
                "weight": inst.weight,
                "style": inst.style,
                "variable": inst.variable,
                "axis_position": inst.axis_position,
            }
            for inst in self.instances
        ]
