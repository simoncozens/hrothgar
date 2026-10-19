"""Dataset for instance-membership training (glyph-level, episodic).

An *instance* is a font file at a specific axis position, so membership is
judged against a specific weight/width/style rather than a whole family.  Each
batch samples several families, several instances per family (same-family hard
negatives), and several codepoints per instance (content-invariant positives).
The collate renders uniform scale-and-pad glyphs plus geometry, matching the
framing the generator will denormalize to.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Sampler

from hrothgar.dataset import DatasetMaker, fvar_axes
from hrothgar.googlefonts import GoogleFont, GoogleFonts
from hrothgar.render import geometry_tensor, render_glyph_uniform
from hrothgar.set_membership.config import SetMembershipConfig


@dataclass
class Instance:
    id: int
    font: GoogleFont
    axis_position: list[float] | None
    family: str
    category: str


def build_instances(
    fonts: Sequence[GoogleFont], max_per_font: int = 4
) -> list[Instance]:
    """Enumerate instances.

    A static file contributes one default instance.  A variable file contributes
    its default plus one instance per axis moved to its far endpoint, capped at
    ``max_per_font`` — so a ``wght``/``wdth`` font yields weight- and width-
    shifted instances to learn against.
    """
    instances: list[Instance] = []
    for font in fonts:
        positions: list[list[float] | None] = [None]
        axes = fvar_axes(font.path)
        if axes:
            seen: set[tuple] = set()
            for tag, lo, default, hi in axes:
                extreme = hi if abs(hi - default) >= abs(lo - default) else lo
                if extreme == default:
                    continue
                position = tuple(
                    extreme if axis[0] == tag else axis[2] for axis in axes
                )
                if position not in seen:
                    seen.add(position)
                    positions.append(list(position))
            positions = positions[:max_per_font]
        for position in positions:
            instances.append(Instance(-1, font, position, font.family, font.category()))
    return instances


class _InstanceDataset(TorchDataset):
    def __init__(self, instances: Sequence[Instance]):
        self.instances = list(instances)

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> Instance:
        return self.instances[idx]


class InstanceBatchSampler(Sampler):
    """Sample instances so each batch contains same-family hard negatives.

    Each batch draws ``families_per_batch`` families and ``instances_per_family``
    instances from each, so e.g. a Bold and a Regular of the same family land in
    the same batch as negatives.
    """

    def __init__(
        self,
        instances,
        batch_size,
        instances_per_family,
        *,
        drop_last=True,
        rng=None,
    ):
        if instances_per_family <= 0 or batch_size % instances_per_family != 0:
            raise ValueError(
                "batch_size must be a positive multiple of instances_per_family "
                f"(got {batch_size}, {instances_per_family})"
            )
        self.batch_size = batch_size
        self.instances_per_family = instances_per_family
        self.drop_last = drop_last
        self._rng = rng if rng is not None else random
        self._family_to_idx: dict[str, list[int]] = defaultdict(list)
        for i, instance in enumerate(instances):
            self._family_to_idx[instance.family].append(i)
        self.families = sorted(self._family_to_idx)

    @property
    def families_per_batch(self) -> int:
        return self.batch_size // self.instances_per_family

    def __len__(self) -> int:
        n = len(self.families) // self.families_per_batch
        if self.drop_last or len(self.families) % self.families_per_batch == 0:
            return n
        return n + 1

    def __iter__(self):
        families = self.families[:]
        self._rng.shuffle(families)
        per_batch = self.families_per_batch
        for start in range(0, len(families), per_batch):
            chunk = families[start : start + per_batch]
            if len(chunk) < per_batch and self.drop_last:
                break
            batch: list[int] = []
            for family in chunk:
                members = self._family_to_idx[family]
                picks = self._rng.sample(
                    members, min(self.instances_per_family, len(members))
                )
                while len(picks) < self.instances_per_family:
                    picks.append(self._rng.choice(members))
                batch.extend(picks)
            self._rng.shuffle(batch)
            yield batch


class SetMembershipDatasetMaker:
    """Build train/test instance loaders for membership training."""

    def __init__(
        self,
        repo_url,
        batch_size,
        config: SetMembershipConfig,
        *,
        split_seed: int = 1234,
        canary_size: int | None = None,
    ):
        self.config = config
        self.batch_size = batch_size
        self._input_codepoints = list(config.input_codepoints)
        self._glyph_size = config.glyph_size
        self._codepoints_per_instance = config.codepoints_per_instance
        self._instances_per_family = config.instances_per_family

        self.googlefonts = GoogleFonts(str(repo_url), max_fonts=canary_size)
        self.filter_fonts()
        train_fonts, test_fonts = DatasetMaker._split_fonts_by_family(
            list(self.googlefonts.fonts), split_seed=split_seed
        )
        self.train_instances = self._assign_ids(
            build_instances(train_fonts, config.max_instances_per_font)
        )
        self.test_instances = self._assign_ids(
            build_instances(test_fonts, config.max_instances_per_font)
        )
        print(
            f"Instances: {len(self.train_instances)} train / {len(self.test_instances)} test "
            f"(from {len(train_fonts)} / {len(test_fonts)} files)"
        )

    def filter_fonts(self) -> None:
        needed = set(self._input_codepoints)
        self.googlefonts.fonts = [
            font for font in self.googlefonts.fonts if needed <= font.codepoints
        ]

    @staticmethod
    def _assign_ids(instances):
        for i, instance in enumerate(instances):
            instance.id = i
        return instances

    def train_loader(self):
        dataset = _InstanceDataset(self.train_instances)
        sampler = InstanceBatchSampler(
            self.train_instances, self.batch_size, self._instances_per_family
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=self.collate_fn,
            num_workers=8,
            pin_memory=True,
        )

    def test_loader(self):
        dataset = _InstanceDataset(self.test_instances)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=self.collate_fn,
            num_workers=8,
            pin_memory=True,
        )

    def collate_fn(self, batch: list[Instance]) -> dict:
        k = self._codepoints_per_instance
        size = self._glyph_size
        images: list[torch.Tensor] = []
        geometry: list[torch.Tensor] = []
        instance_ids: list[int] = []
        for instance in batch:
            available = [
                cp for cp in self._input_codepoints if cp in instance.font.codepoints
            ]
            chosen = random.sample(available, min(k, len(available)))
            while len(chosen) < k:
                chosen.append(random.choice(available))
            for codepoint in chosen:
                image, geom = render_glyph_uniform(
                    instance.font, codepoint, size, axis_position=instance.axis_position
                )
                images.append(image.unsqueeze(0))
                geometry.append(geometry_tensor(geom))
                instance_ids.append(instance.id)
        return {
            "images": torch.stack(images),  # (N, 1, H, W)
            "geometry": torch.stack(geometry),  # (N, 5)
            "instance_ids": torch.tensor(instance_ids, dtype=torch.long),  # (N,)
        }
