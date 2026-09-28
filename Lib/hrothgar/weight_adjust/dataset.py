"""Dataset for the weight-adjustment model.

Each training item is a target glyph plus ``num_exemplars`` exemplar pairs, all
from a single font's regular→bold relationship.  That relationship comes from
one of two sources:

* **variable font** — a single file with a ``wght`` axis; the regular and bold
  are synthesized by setting the axis coordinate.
* **static pair** — a ``-Regular`` file and a ``-Bold`` file (same family and
  style, matched via ``METADATA.pb``); the regular and bold are *different
  files*, so the model learns from the font's actual bold design rather than a
  dialed coordinate.

Training **resamples on the fly**: each ``__getitem__`` draws a fresh random
(source, target, exemplars, weight) combination, so the model never re-reads
the same samples twice and sees an unbounded variety over training.  Validation
is fixed and deterministic (every held-out codepoint of every held-out source),
so the metric stays comparable across checkpoints.

A random held-out subset of the ``character_set`` (split by ``split_seed``) is
never a training target or exemplar, so validation measures generalization to
constructions the model never trained on.  Sources are split into train/val by
family, so validation also exercises the style encoder on unseen fonts.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from hrothgar.dataset import (
    _axis_position,
    _has_non_empty_outline,
    _hb_font_for_face,
    fvar_axes,
    parse_metadata_weights,
)
from hrothgar.dataset_constants import LATIN_KERNEL
from hrothgar.googlefonts import GoogleFonts
from hrothgar.weight_adjust.render import render_gid_shared_frame


# ── Shared item-building / rendering helpers ────────────────────────────────


def _gid(font, cp: int) -> int:
    return _hb_font_for_face(font.hb_face).get_nominal_glyph(cp)


def _codepoints_in(font, allowed: set[int]) -> list[int]:
    """Codepoints in ``allowed`` that this font draws with a non-empty outline."""
    hb = _hb_font_for_face(font.hb_face)
    out = []
    for cp in sorted(font.codepoints & allowed):
        gid = hb.get_nominal_glyph(cp)
        if _has_non_empty_outline(hb.get_glyph_extents(gid)):
            out.append(cp)
    return out


def _codepoints_in_pair(regular_font, bold_font, allowed: set[int]) -> list[int]:
    """Codepoints drawable in *both* files of a static pair."""
    reg_hb = _hb_font_for_face(regular_font.hb_face)
    bold_hb = _hb_font_for_face(bold_font.hb_face)
    out = []
    for cp in sorted(regular_font.codepoints & bold_font.codepoints & allowed):
        if _has_non_empty_outline(
            reg_hb.get_glyph_extents(reg_hb.get_nominal_glyph(cp))
        ) and _has_non_empty_outline(
            bold_hb.get_glyph_extents(bold_hb.get_nominal_glyph(cp))
        ):
            out.append(cp)
    return out


def _source_codepoints(source: dict, allowed: set[int]) -> list[int]:
    if source["kind"] == "variable":
        return _codepoints_in(source["font"], allowed)
    return _codepoints_in_pair(source["regular"], source["bold"], allowed)


def _wght_axis(axes: dict, font_path) -> list | None:
    """The font's ``wght`` axis ``[tag, min, default, max]``, or ``None``."""
    return next((a for a in axes[font_path] if a[0] == "wght"), None)


def _effective_wght(axes: dict, font_path, requested: int) -> float:
    """The weight actually rendered after clamping to the font's wght range."""
    wght = _wght_axis(axes, font_path)
    if wght is None:
        return float(requested)
    return float(min(max(requested, wght[1]), wght[3]))


def _weight_scalar(axes: dict, font_path, regular_weight: int, target_weight: int) -> float:
    """Normalized target weight, computed from the *clamped* rendered weight so
    the scalar always matches the actual raster thickness."""
    eff_regular = _effective_wght(axes, font_path, regular_weight)
    eff_target = _effective_wght(axes, font_path, target_weight)
    return (eff_target - eff_regular) / max(eff_regular, 1e-6)


def _variable_item(
    axes: dict,
    font,
    regular_weight: int,
    bold_weight: int,
    target_cp: int,
    exemplar_cps: Sequence[int],
    target_weight: int,
) -> dict:
    fa = axes[font.path]
    return {
        "src": (font, _axis_position(fa, regular_weight)),
        "tgt": (font, _axis_position(fa, target_weight)),
        "ref": (font, _axis_position(fa, bold_weight)),
        "target_cp": target_cp,
        "exemplar_cps": list(exemplar_cps),
        "weight": _weight_scalar(axes, font.path, regular_weight, target_weight),
    }


def _static_item(
    regular_font,
    bold_font,
    regular_weight: int,
    bold_weight: int,
    target_cp: int,
    exemplar_cps: Sequence[int],
) -> dict:
    return {
        "src": (regular_font, None),
        "tgt": (bold_font, None),
        "ref": (bold_font, None),
        "target_cp": target_cp,
        "exemplar_cps": list(exemplar_cps),
        "weight": (bold_weight - regular_weight) / regular_weight,
    }


def _source_item(
    source: dict,
    axes: dict,
    regular_weight: int,
    bold_weight: int,
    target_cp: int,
    exemplar_cps: Sequence[int],
    target_weight: int,
) -> dict:
    if source["kind"] == "variable":
        return _variable_item(
            axes, source["font"], regular_weight, bold_weight,
            target_cp, exemplar_cps, target_weight,
        )
    return _static_item(
        source["regular"], source["bold"], regular_weight, bold_weight,
        target_cp, exemplar_cps,
    )


def _render_item(item: dict, image_size: int) -> dict:
    """Render an item dict into the tensors the model consumes."""
    src_font, src_axis = item["src"]
    tgt_font, tgt_axis = item["tgt"]
    ref_font, ref_axis = item["ref"]
    target = item["target_cp"]

    reg_img, reg_geo = render_gid_shared_frame(
        src_font, _gid(src_font, target), image_size, src_axis
    )
    bold_img, bold_geo = render_gid_shared_frame(
        tgt_font, _gid(tgt_font, target), image_size, tgt_axis
    )

    pairs = []
    for cp in item["exemplar_cps"]:
        er, _ = render_gid_shared_frame(
            src_font, _gid(src_font, cp), image_size, src_axis
        )
        eb, _ = render_gid_shared_frame(
            ref_font, _gid(ref_font, cp), image_size, ref_axis
        )
        pairs.append(torch.stack([er, eb], dim=0))  # (2, H, W)
    exemplars = torch.stack(pairs, dim=0)  # (K, 2, H, W)

    return {
        "regular": reg_img.unsqueeze(0),
        "bold": bold_img.unsqueeze(0),
        "font": src_font,
        "regular_advance": reg_geo["advance"],
        "advance_delta": bold_geo["advance"] - reg_geo["advance"],
        "exemplars": exemplars,
        "weight": item["weight"],
    }


def _collate_fn(batch: list[dict]) -> dict:
    return {
        "regular": torch.stack([b["regular"] for b in batch]),
        "bold": torch.stack([b["bold"] for b in batch]),
        "font": [b["font"] for b in batch],
        "regular_advance": torch.tensor(
            [b["regular_advance"] for b in batch], dtype=torch.float32
        ),
        "advance_delta": torch.tensor(
            [b["advance_delta"] for b in batch], dtype=torch.float32
        ),
        "exemplars": torch.stack([b["exemplars"] for b in batch]),
        "weight": torch.tensor([b["weight"] for b in batch], dtype=torch.float32),
    }


# ── Datasets ────────────────────────────────────────────────────────────────


class WeightAdjustDataset(TorchDataset):
    """Fixed-item dataset (validation): renders precomputed items."""

    def __init__(self, items: list[dict], image_size: int) -> None:
        self.items = items
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return _render_item(self.items[idx], self.image_size)


class ResamplingWeightAdjustDataset(TorchDataset):
    """On-the-fly training dataset: a fresh random draw on every access.

    The item (source, target, exemplars, weight) is resampled from a persistent
    seeded RNG each ``__getitem__`` call, so consecutive epochs never repeat a
    sample and the model sees an unbounded variety.  ``len`` only sets the epoch
    size (how many draws per epoch).
    """

    def __init__(
        self,
        sources: list[dict],
        axes: dict,
        *,
        image_size: int,
        num_exemplars: int,
        regular_weight: int,
        bold_weight: int,
        weight_min: int,
        weight_max: int,
        codepoints: set[int],
        samples_per_font: int,
        seed: int,
    ) -> None:
        self.image_size = image_size
        self.num_exemplars = num_exemplars
        self.regular_weight = regular_weight
        self.bold_weight = bold_weight
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.axes = axes
        self.rng = random.Random(seed)

        # Precompute the valid codepoint pool per source so the hot path doesn't
        # redo hb lookups; drop sources that can't yield enough exemplars.
        self.sources = []
        self.avail = []
        for source in sources:
            pool = _source_codepoints(source, codepoints)
            if len(pool) < num_exemplars + 1:
                continue
            self.sources.append(source)
            self.avail.append(pool)
        self.epoch_size = len(self.sources) * samples_per_font

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, idx: int) -> dict:
        del idx  # ignored: the RNG (not the index) picks the sample
        si = self.rng.randrange(len(self.sources))
        source = self.sources[si]
        avail = self.avail[si]

        target = self.rng.choice(avail)
        pool = [c for c in avail if c != target]
        exemplars = self.rng.sample(pool, self.num_exemplars)
        target_weight = self.rng.randint(self.weight_min, self.weight_max)

        item = _source_item(
            source, self.axes, self.regular_weight, self.bold_weight,
            target, exemplars, target_weight,
        )
        return _render_item(item, self.image_size)


# ── Maker ───────────────────────────────────────────────────────────────────


class WeightAdjustDatasetMaker:
    """Builds train/val loaders over variable fonts and static pairs."""

    def __init__(
        self,
        repo: str | Path,
        batch_size: int,
        *,
        image_size: int = 128,
        num_exemplars: int = 5,
        regular_weight: int = 400,
        bold_weight: int = 700,
        canary_size: int | None = None,
        samples_per_font: int = 8,
        val_fraction: float = 0.2,
        character_set: Sequence[int] = LATIN_KERNEL,
        val_codepoint_fraction: float = 0.25,
        weight_min: int = 500,
        weight_max: int = 800,
        include_static: bool = True,
        split_seed: int = 1234,
    ) -> None:
        self.repo = Path(repo)
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_exemplars = num_exemplars
        self.regular_weight = regular_weight
        self.bold_weight = bold_weight
        self.samples_per_font = samples_per_font
        self.weight_min = weight_min
        self.weight_max = weight_max

        # Held-out codepoints: a random subset of the character set, mirroring
        # the base ``DatasetMaker`` character-set split.
        train_cp, val_cp = train_test_split(
            sorted(set(character_set)),
            test_size=val_codepoint_fraction,
            random_state=split_seed,
        )
        self.train_codepoints = set(train_cp)
        self.val_codepoints = set(val_cp)

        gf = GoogleFonts(str(self.repo), max_fonts=canary_size)

        # Variable sources: one file, weights dialed via the wght axis.  Skip
        # fonts whose wght axis cannot go heavier than the regular weight
        # (e.g. thin-only families like Playwrite) — they produce a target
        # identical to the input and teach the model the identity map.
        variable = [f for f in gf.fonts if self._is_variable_with_wght(f)]
        self.axes = {f.path: fvar_axes(f.path) for f in variable}
        variable = [f for f in variable if self._can_bold(f)]

        # Unified sources: each is a (family, style) regular→bold relationship.
        sources = []
        for f in variable:
            sources.append({"family": f.family, "kind": "variable", "font": f})
        n_static = 0
        if include_static:
            for pair in self._find_static_pairs(gf):
                sources.append({"family": pair["family"], "kind": "static", **pair})
                n_static += 1

        train_sources, val_sources = self._split_sources_by_family(
            sources, val_fraction, split_seed
        )
        rng = random.Random(split_seed)

        self.val_items = self._build_val_items(val_sources, rng)
        self.train_dataset = ResamplingWeightAdjustDataset(
            train_sources,
            self.axes,
            image_size=image_size,
            num_exemplars=num_exemplars,
            regular_weight=regular_weight,
            bold_weight=bold_weight,
            weight_min=weight_min,
            weight_max=weight_max,
            codepoints=self.train_codepoints,
            samples_per_font=samples_per_font,
            seed=split_seed + 3,
        )
        print(
            f"WeightAdjust: {len(sources)} sources "
            f"({len(variable)} variable / {n_static} static); "
            f"{len(train_sources)} train / {len(val_sources)} val; "
            f"{len(self.train_codepoints)} train cps / {len(self.val_codepoints)} held-out cps; "
            f"{self.train_dataset.epoch_size} train draws/epoch / {len(self.val_items)} val items"
        )

    # -- font selection -----------------------------------------------------

    @staticmethod
    def _is_variable_with_wght(font) -> bool:
        axes = fvar_axes(font.path)
        return axes is not None and any(a[0] == "wght" for a in axes)

    def _can_bold(self, font) -> bool:
        """True when the font can render a weight heavier than ``regular_weight``."""
        wght = _wght_axis(self.axes, font.path)
        return wght is not None and wght[3] > self.regular_weight

    def _find_static_pairs(self, gf) -> list[dict]:
        """Match ``-Regular`` / ``-Bold`` files within each (family, style)."""
        static = [f for f in gf.fonts if not self._is_variable_with_wght(f)]
        by_meta = defaultdict(list)
        for f in static:
            by_meta[f.metadata_pb].append(f)

        pairs = []
        for meta_pb, fonts in by_meta.items():
            weights = parse_metadata_weights(meta_pb)
            by_style = defaultdict(dict)  # style -> {weight: font}
            for f in fonts:
                wt, st = weights.get(f.path.name, (None, "normal"))
                if wt is None:
                    continue
                by_style[st][wt] = f
            for style, w2f in by_style.items():
                regular = w2f.get(self.regular_weight)
                bold = w2f.get(self.bold_weight)
                if regular is not None and bold is not None:
                    pairs.append(
                        {
                            "family": regular.family,
                            "style": style,
                            "regular": regular,
                            "bold": bold,
                        }
                    )
        return pairs

    @staticmethod
    def _split_sources_by_family(
        sources: list[dict], val_fraction: float, split_seed: int
    ) -> tuple[list, list]:
        family_to_sources = defaultdict(list)
        for s in sources:
            family_to_sources[s["family"]].append(s)
        families = sorted(family_to_sources)
        rng = random.Random(split_seed)
        rng.shuffle(families)
        n_val = max(1, int(len(families) * val_fraction))
        val_families = set(families[:n_val])
        train = [
            s for fam in families if fam not in val_families for s in family_to_sources[fam]
        ]
        val = [s for fam in val_families for s in family_to_sources[fam]]
        return train, val

    # -- item construction ---------------------------------------------------

    def _build_val_items(self, sources: list[dict], rng: random.Random) -> list[dict]:
        items = []
        for source in sources:
            targets = _source_codepoints(source, self.val_codepoints)
            if not targets:
                continue
            exemplar_pool = _source_codepoints(source, self.train_codepoints)
            if len(exemplar_pool) < self.num_exemplars:
                continue
            exemplars = rng.sample(exemplar_pool, self.num_exemplars)
            for cp in targets:
                items.append(
                    _source_item(
                        source, self.axes, self.regular_weight, self.bold_weight,
                        cp, exemplars, self.bold_weight,
                    )
                )
        return items

    # -- loaders -------------------------------------------------------------

    def train_loader(self):
        # The resampling dataset draws randomly via its own RNG, so no shuffle
        # is needed (and ``num_workers`` must stay 0 for the RNG to be shared).
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=_collate_fn,
        )

    def val_loader(self):
        # Keep the last (partial) validation batch so a small canary val set is
        # never silently empty.
        return DataLoader(
            WeightAdjustDataset(self.val_items, self.image_size),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=_collate_fn,
        )

    def random_val_batch(self, n: int) -> dict:
        """A random batch of validation items, for visualization only."""
        items = random.sample(self.val_items, k=min(n, len(self.val_items)))
        dataset = WeightAdjustDataset(items, self.image_size)
        return _collate_fn([dataset[i] for i in range(len(items))])
