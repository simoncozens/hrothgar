"""Full-dataset maker for the factorized (codepoint, font-instance) diffusion model.

Each training item is a ``(glyph image, codepoint index, font instance)`` triple.
A font instance is a ``(file, weight, style, axis_position)`` row — for a static
family this is one weight file; for a variable family it may be a synthesised
``wght`` location on a single file.

The subset of font instances used for training is chosen by a **stratified
sampler** (see :class:`~hrothgar.dataset.StratifiedFontSampler`) rather than by
truncating the load order.  The :class:`RupeeFontSampler` subclass prefers
families that contain the acceptance glyph (₹), balancing text (sans/serif)
against fancy (display/script/handwriting) instances, keeping a spread of
in-family *contrast* (e.g. 100 + 900, never 400 + 500), and synthesising
variable-font locations.

The sampling unit is a ``(family, style)`` pair, so roman and italic are sampled
independently — an italic construction may legitimately differ from its roman
counterpart and gets its own conditioning.

The split is **codepoint-based**, not instance-based: for each codepoint we hold
out a fraction of the instances that contain it (the "fill in the missing glyph"
scenario).  Two invariants are enforced:

* every codepoint keeps at least ``min_train_fonts_per_codepoint`` training
  instances (so its content embedding is learned), and
* every instance keeps most of its codepoints (so its style embedding is
  learned).

The held-out ``(instance, codepoint)`` pairs form the validation set — generating
them correctly is the actual acceptance test.
"""

from __future__ import annotations

import json
import os
import random
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from hrothgar.dataset import (
    Instance,
    StratifiedFontSampler,
    Unit,
    _axis_position,
    _has_non_empty_outline,
    _hb_font_for_face,
    units_from_dicts,
    units_to_dicts,
)
from hrothgar.dataset_constants import LATIN_KERNEL
from hrothgar.googlefonts import GoogleFont, GoogleFonts, StandaloneFont
from hrothgar.render import GEOMETRY_SPEC, geometry_tensor, render_glyph_with_geometry

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))

RUPEE = ord("\u20b9")  # U+20B9 — the acceptance glyph


class RupeeFontSampler(StratifiedFontSampler):
    """Stratified sampler that prefers fonts containing an encoded rupee glyph."""

    def use_font(self, font: GoogleFont) -> bool:
        return RUPEE in font.codepoints


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _master_weights(inst: Instance) -> list[int]:
    """Weight master locations for a variable instance: the ``wght`` axis's
    min/default/max (the interpolation endpoints plus the default).  A variable
    instance without a ``wght`` axis yields just its declared weight."""
    wght = next((a for a in (inst.axes or []) if a[0] == "wght"), None)
    if wght is None:
        return [inst.weight]
    return sorted({int(wght[1]), int(wght[2]), int(wght[3])})


def _instance_record(
    unit: Unit, inst: Instance, family_id: int | None, weight: int, axis_position
) -> dict:
    """One inference-sidecar record, mirroring the training conditioning.

    ``family_id`` is ``None`` for families the model never sampled; ``weight``
    and ``axis_position`` describe a static file or a synthesized variable
    location."""
    return {
        "path": inst.path,
        "family": unit.family,
        "family_id": family_id,
        "weight": weight,
        "weight_norm": (weight - 400.0) / 400.0,
        "style": inst.style,
        "style_bucket": 0 if inst.style == "normal" else 1,
        "variable": inst.variable,
        "axis_position": axis_position,
    }


def inference_jobs(
    units: Sequence[Unit], family_to_id: dict[str, int], skip_re: str | None = None
) -> list[dict]:
    """Runtime generation jobs for every weight/style of a *known* family.

    Weight/style conditioning is computed here, not read from a precomputed
    sidecar: static files generate at their own weight, and variable files
    generate at their ``wght`` master locations (axis min/default/max).
    Families absent from ``family_to_id`` are skipped.
    """
    jobs: list[dict] = []
    for unit in units:
        if skip_re and re.match(skip_re, unit.family):
            continue

        family_id = family_to_id.get(unit.family)
        if family_id is None:
            continue
        statics = [i for i in unit.instances if not i.variable]
        variables = [i for i in unit.instances if i.variable]
        covered = {i.weight for i in statics}
        for inst in statics:
            jobs.append(
                _instance_record(unit, inst, family_id, inst.weight, None)
            )
        for inst in variables:
            for weight in _master_weights(inst):
                if weight in covered:
                    continue
                jobs.append(
                    _instance_record(
                        unit,
                        inst,
                        family_id,
                        weight,
                        _axis_position(inst.axes or [], weight),
                    )
                )
    return jobs


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class TrainInstance:
    """A unique training instance: a font file at a specific weight/style."""

    path: str  # repo-relative
    font: StandaloneFont
    axis_position: list[float] | None
    family: str
    family_id: int
    weight: int
    weight_norm: float
    style: str
    style_bucket: int
    variable: bool
    copies: int  # >1 when the sampler oversampled this instance
    codepoints: list[int]  # codepoints with a non-empty outline

    @property
    def font_meta(self) -> tuple[int, float, int]:
        return (self.family_id, self.weight_norm, self.style_bucket)


class _PairDataset(TorchDataset):
    """Yields ``(image, cp_idx, instance_id, font_meta)`` for fixed pairs."""

    def __init__(
        self,
        pairs: list[tuple[int, int]],
        instances: Sequence[TrainInstance],
        cp_list: Sequence[int],
        image_size: int,
    ) -> None:
        self.pairs = pairs
        self.instances = instances
        self.cp_list = list(cp_list)
        self.image_size = image_size

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
        return {
            "image": img.unsqueeze(0),  # (1, H, W)
            "geometry": geometry_tensor(geometry),  # (5,)
            "cp_idx": cp_idx,
            "instance_id": instance_id,
            "font_meta": torch.tensor(instance.font_meta, dtype=torch.float32),
        }


def _collate_fn(batch: list[dict]) -> dict:
    return {
        "images": torch.stack([b["image"] for b in batch]),
        "geometry": torch.stack([b["geometry"] for b in batch]),
        "codepoints": torch.tensor([b["cp_idx"] for b in batch], dtype=torch.long),
        "instance_ids": torch.tensor(
            [b["instance_id"] for b in batch], dtype=torch.long
        ),
        "font_meta": torch.stack([b["font_meta"] for b in batch]),  # (B, 3)
    }


class FontIdDatasetMaker:
    """Builds train/val loaders of ``(image, codepoint, instance)`` triples."""

    def __init__(
        self,
        repo: str | Path,
        batch_size: int,
        *,
        image_size: int = 128,
        character_set: Sequence[int] | None = None,
        extra_codepoints: Sequence[int] | None = None,
        remove_codepoints: Sequence[int] | None = None,
        oversample_codepoints: dict[int, int] | None = None,
        num_instances: int,
        strata: dict[str, float] | None = None,
        max_per_unit: int = 3,
        avg_instances: float = 1.6,
        target_frac: float = 0.7,
        prefer_multi_target: bool = True,
        replacement: bool = True,
        subset_seed: int = 1234,
        heldout_fraction: float = 0.25,
        min_train_fonts_per_codepoint: int = 20,
        split_seed: int = 1234,
        units: list[Unit] | None = None,
    ) -> None:
        self.repo = Path(repo)
        self.image_size = image_size
        self.batch_size = batch_size

        # Vocabulary = base set + extras - removals.
        base = set(character_set or LATIN_KERNEL)
        base |= set(extra_codepoints or [])
        base -= set(remove_codepoints or [])
        self.character_set = sorted(base)
        self.cp_to_idx = {cp: i for i, cp in enumerate(self.character_set)}
        self.cp_list = list(self.character_set)
        self.num_codepoints = len(self.character_set)

        # Codepoint -> oversample factor (rare/acceptance glyphs are fully
        # trained with no held-out split, then duplicated).
        self.oversample_codepoints = dict(oversample_codepoints or {})

        self.heldout_fraction = heldout_fraction
        self.min_train_fonts_per_codepoint = min_train_fonts_per_codepoint
        self.split_seed = split_seed

        # --- Build sampling units and select the training subset. -----------
        sampler = RupeeFontSampler()
        if units is None:
            gf = GoogleFonts(str(self.repo))
            units = sampler.build_units(gf, needed_codepoints=base)
        if num_instances <= 0:
            raise ValueError(
                "num_instances must be a positive integer (the stratified "
                "dataset size).  There is deliberately no 'all instances' "
                "mode — the raw library is unbalanced."
            )
        selected, self.subset_report = sampler.select_subset(
            units,
            num_instances,
            strata or None,  # empty dict -> the sampler's default strata mix
            max_per_unit=max_per_unit,
            avg_instances=avg_instances,
            target_frac=target_frac,
            prefer_multi_target=prefer_multi_target,
            replacement=replacement,
            min_coverage=min_train_fonts_per_codepoint + 1,
            seed=subset_seed,
        )

        # --- Materialise unique instances (collapse replacement copies). ----
        self.families = sorted({s.family for s in selected})
        self.family_to_id = {fam: i for i, fam in enumerate(self.families)}
        self.num_families = len(self.families)
        self.num_style_buckets = 2

        self.instances: list[TrainInstance] = []
        index_of: dict[tuple, int] = {}
        for s in selected:
            key = (s.path, s.weight, s.style, tuple(s.axis_position or ()))
            if key in index_of:
                self.instances[index_of[key]].copies += 1
                continue
            abspath = self.repo / s.path
            font = StandaloneFont(abspath)
            cps = self._available_codepoints(font)
            index_of[key] = len(self.instances)
            self.instances.append(
                TrainInstance(
                    path=s.path,
                    font=font,
                    axis_position=s.axis_position,
                    family=s.family,
                    family_id=self.family_to_id[s.family],
                    weight=s.weight,
                    weight_norm=s.weight_norm,
                    style=s.style,
                    style_bucket=s.style_bucket,
                    variable=s.variable,
                    copies=1,
                    codepoints=cps,
                )
            )

        self.font_meta = [inst.font_meta for inst in self.instances]
        self._build_pairs()
        self.geometry_std = self._compute_geometry_std()
        print(
            f"Instances: {len(self.instances)} unique "
            f"({sum(i.copies for i in self.instances)} rows); "
            f"families: {self.num_families}; codepoints: {self.num_codepoints}"
        )
        print(f"Pairs: {len(self.train_pairs)} train / {len(self.val_pairs)} held-out")

    # -- pair construction ---------------------------------------------------

    def _compute_geometry_std(self, n_samples: int = 2000) -> tuple[float, ...]:
        """Per-label standard deviation of the geometry labels, over a sample of
        training pairs (used to normalise the geometry loss)."""
        if not self.train_pairs:
            return tuple([1.0] * len(GEOMETRY_SPEC))
        rng = random.Random(self.split_seed)
        sample = rng.sample(self.train_pairs, min(n_samples, len(self.train_pairs)))
        geos = []
        for iid, cp_idx in sample:
            inst = self.instances[iid]
            _, geom = render_glyph_with_geometry(
                inst.font,
                self.cp_list[cp_idx],
                self.image_size,
                axis_position=inst.axis_position,
            )
            geos.append(geometry_tensor(geom))
        std = torch.stack(geos).std(dim=0).clamp_min(1e-3)
        return tuple(std.tolist())

    def _available_codepoints(self, font: StandaloneFont) -> list[int]:
        """Codepoints in the vocabulary that this font draws with a real outline."""
        hb_font = _hb_font_for_face(font.hb_face)
        out = []
        for cp in sorted(set(font.codepoints) & set(self.character_set)):
            gid = hb_font.get_nominal_glyph(cp)
            extents = hb_font.get_glyph_extents(gid)
            if _has_non_empty_outline(extents):
                out.append(cp)
        return out

    def _build_pairs(self) -> None:
        rng = random.Random(self.split_seed)
        cp_to_instances: dict[int, list[int]] = defaultdict(list)
        for iid, inst in enumerate(self.instances):
            for cp in inst.codepoints:
                cp_to_instances[self.cp_to_idx[cp]].append(iid)

        self.train_pairs: list[tuple[int, int]] = []
        self.val_pairs: list[tuple[int, int]] = []
        for cp_idx, inst_ids in cp_to_instances.items():
            cp = self.cp_list[cp_idx]
            if cp in self.oversample_codepoints:
                # Rare/acceptance codepoints: train on every instance that has
                # the glyph (no held-out), duplicated by the oversample factor.
                factor = self.oversample_codepoints[cp]
                for iid in inst_ids:
                    self.train_pairs.extend(
                        [(iid, cp_idx)] * (factor * self.instances[iid].copies)
                    )
                continue
            normal = list(inst_ids)
            n = len(normal)
            max_holdout = max(0, n - self.min_train_fonts_per_codepoint)
            n_holdout = min(int(self.heldout_fraction * n), max_holdout)
            holdout = set(rng.sample(normal, n_holdout)) if n_holdout else set()
            for iid in normal:
                if iid in holdout:
                    self.val_pairs.append((iid, cp_idx))
                else:
                    self.train_pairs.extend(
                        [(iid, cp_idx)] * self.instances[iid].copies
                    )

        random.Random(self.split_seed).shuffle(self.val_pairs)

    # -- loaders -------------------------------------------------------------

    def _loader(self, pairs: list[tuple[int, int]], shuffle: bool):
        dataset = _PairDataset(pairs, self.instances, self.cp_list, self.image_size)
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
        return self._loader(self.val_pairs, shuffle=False)

    def random_val_batch(self, n: int) -> dict:
        """A random (unseeded) batch of held-out pairs, for visualization only."""
        pairs = random.sample(self.val_pairs, k=min(n, len(self.val_pairs)))
        dataset = _PairDataset(pairs, self.instances, self.cp_list, self.image_size)
        return _collate_fn([dataset[i] for i in range(len(pairs))])

    # -- sidecars ------------------------------------------------------------

    def instance_sidecar(self) -> list[dict]:
        """Instance records for inference (mirrors the training conditioning)."""
        return [
            {
                "path": inst.path,
                "family": inst.family,
                "family_id": inst.family_id,
                "weight": inst.weight,
                "weight_norm": inst.weight_norm,
                "style": inst.style,
                "style_bucket": inst.style_bucket,
                "variable": inst.variable,
                "axis_position": inst.axis_position,
            }
            for inst in self.instances
        ]


def load_or_build_units(
    repo: str | Path,
    needed_codepoints: set[int],
    cache_path: Path | None = None,
    rebuild: bool = False,
) -> list[Unit]:
    """Load sampling units from a JSON cache, or build + cache them from the repo."""
    if cache_path is not None and not rebuild and cache_path.exists():
        data = json.loads(cache_path.read_text())
        if data.get("repo") == str(repo) and data.get("needed") == sorted(
            needed_codepoints
        ):
            print(
                f"Loaded {len(data['units'])} (family, style) units from {cache_path}"
            )
            return units_from_dicts(data["units"])
    gf = GoogleFonts(str(repo))
    units = RupeeFontSampler().build_units(gf, needed_codepoints=needed_codepoints)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "repo": str(repo),
                    "needed": sorted(needed_codepoints),
                    "units": units_to_dicts(units),
                }
            )
        )
        print(f"Cached {len(units)} (family, style) units -> {cache_path}")
    return units


def parse_strata(spec: str | None) -> dict[str, float]:
    """Parse a ``'sans:0.25,serif:0.25,...'`` fraction spec (empty = defaults)."""
    if not spec:
        return dict(RupeeFontSampler.DEFAULT_STRATA)
    out: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, frac = part.split(":")
        out[name.strip()] = float(frac)
    return out


def main() -> None:
    """CLI: report balance statistics for a given ``--num-instances``.

    Run ``python -m hrothgar.diffusion.dataset_fontid --num-instances N`` to see,
    before training, how a stratified subset of size ``N`` is distributed:
    instances per category, how many of the available families are sampled, and
    how many instances are oversampled (replacement fill).
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("GOOGLE_FONTS_REPO"))
    parser.add_argument(
        "--num-instances",
        type=int,
        required=True,
        help="target stratified instance count",
    )
    parser.add_argument(
        "--strata",
        default=None,
        help="override, e.g. 'sans:0.25,serif:0.25,display:0.2,"
        "script:0.2,handwriting:0.1'",
    )
    parser.add_argument("--max-per-unit", type=int, default=3)
    parser.add_argument("--avg-instances", type=float, default=1.6)
    parser.add_argument("--target-frac", type=float, default=0.7)
    parser.add_argument("--min-coverage", type=int, default=21)
    parser.add_argument(
        "--no-prefer-multi-target",
        dest="prefer_multi_target",
        action="store_false",
        default=True,
    )
    parser.add_argument("--no-replacement", dest="replacement", action="store_false")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(os.environ.get("FONT_DB_CACHE", "/tmp/hrothgar_units.json")),
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    if not args.repo:
        raise SystemExit("Provide --repo or set GOOGLE_FONTS_REPO")

    sampler = RupeeFontSampler()
    needed = set(LATIN_KERNEL) | {RUPEE}
    units = load_or_build_units(args.repo, needed, args.cache, args.rebuild_cache)
    strata_fracs = parse_strata(args.strata)
    _, rep = sampler.select_subset(
        units,
        args.num_instances,
        strata_fracs,
        max_per_unit=args.max_per_unit,
        avg_instances=args.avg_instances,
        target_frac=args.target_frac,
        prefer_multi_target=args.prefer_multi_target,
        replacement=args.replacement,
        min_coverage=args.min_coverage,
        seed=args.seed,
    )

    all_families = len({u.family for u in units})
    eligible_families = len(
        {u.family for u in units if u.max_coverage() >= args.min_coverage}
    )

    print(f"\nStratified subset report (n={args.num_instances}, seed={args.seed})")
    print(
        f"instances: {rep['n_instances']} total "
        f"({rep['distinct_instances']} distinct + "
        f"{rep['oversampled_instances']} oversampled)"
    )
    print("instances per category:")
    for st in sampler.STRATA:
        n = rep["instances_per_stratum"].get(st, 0)
        if n:
            print(f"  {st:<12}{n:>6}  ({n / max(rep['n_instances'], 1):>5.0%})")
    print(f"text / fancy: {rep['text_instances']} / {rep['fancy_instances']}")
    print(
        f"families sampled: {rep['families']} "
        f"(of {all_families} available, {eligible_families} eligible "
        f"at min-coverage {args.min_coverage})"
    )
    print(f"units sampled: {rep['units']}")
    print(f"oversampled instances: {rep['oversampled_instances']}")
    print(f"instances per family: {rep['family_k_histogram']}")
    print(
        f"styles: {rep['style_counts']}  | variable instances: "
        f"{rep['variable_instances']}"
    )
    print(
        f"families with target: {rep['families_with_target']}/{rep['families']} "
        f"({rep['target_family_frac']:.0%})"
    )


if __name__ == "__main__":
    main()
