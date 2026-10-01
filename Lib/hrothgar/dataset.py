from __future__ import annotations

import math
import os
import random
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, ClassVar, Generic, TypeVar

import torch
import uharfbuzz as hb
from sklearn.model_selection import train_test_split
from torch.utils.data import BatchSampler, DataLoader
from torch.utils.data import Dataset as TorchDataset

from hrothgar.dataset_constants import CAPS_ONLY  # pyright: ignore[reportUnusedImport]
from hrothgar.dataset_constants import (
    LATIN_KERNEL,
)  # pyright: ignore[reportUnusedImport]
from hrothgar.dataset_constants import LATIN_CORE  # noqa: F401
from hrothgar.googlefonts import GoogleFont, GoogleFonts

_T = TypeVar("_T")


def _has_non_empty_outline(extents) -> bool:
    """Return True when HarfBuzz extents indicate drawable geometry."""
    if extents is None:
        return False
    _x_bearing, _y_bearing, width, height = extents
    return not (width == 0 and height == 0)


def _hb_font_for_face(face):
    """Construct a HarfBuzz Font object for a face."""
    return hb.Font(face)


class ClassBalancedBatchSampler(BatchSampler, Generic[_T]):
    """Batch sampler that balances a class label within each emitted batch.

    Items are grouped by a caller-supplied ``key`` (for example a font's
    category or classification), and each batch draws an even quota of items
    from every class.  This stops majority classes (e.g. sans-serif fonts)
    from dominating training batches.
    """

    def __init__(
        self,
        items: Sequence[_T],
        *,
        key: Callable[[_T], str],
        batch_size: int,
        drop_last: bool,
        rng=None,
    ) -> None:
        """
        ``rng`` is the RNG used for font selection; defaults to the global
        ``random`` module.  Passing a dedicated ``random.Random`` makes batch
        composition reproducible (canary mode).
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if len(items) == 0:
            raise ValueError("Cannot build class-balanced sampler for empty dataset")

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.dataset_size = len(items)
        self._rng = rng if rng is not None else random

        class_to_indices: dict[str, list[int]] = {}
        for idx, item in enumerate(items):
            class_to_indices.setdefault(key(item), []).append(idx)

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

        class_cursor = self._rng.randrange(num_classes)

        for _ in range(num_batches):
            batch_indices: list[int] = []

            if num_classes <= self.batch_size:
                base = self.batch_size // num_classes
                remainder = self.batch_size % num_classes
                class_order = self.classes[:]
                self._rng.shuffle(class_order)

                for cls in class_order:
                    indices = self.class_to_indices[cls]
                    for _ in range(base):
                        batch_indices.append(self._rng.choice(indices))

                for cls in class_order[:remainder]:
                    batch_indices.append(self._rng.choice(self.class_to_indices[cls]))
            else:
                selected_classes = [
                    self.classes[(class_cursor + i) % num_classes]
                    for i in range(self.batch_size)
                ]
                class_cursor = (class_cursor + self.batch_size) % num_classes
                for cls in selected_classes:
                    batch_indices.append(self._rng.choice(self.class_to_indices[cls]))

            self._rng.shuffle(batch_indices)
            yield batch_indices


class DatasetMaker:
    """Create train/test splits and loaders over glyph rendering items."""

    def __init__(
        self,
        repo_url: str,
        batch_size: int,
        having: set[int] | None = None,
        target_codepoints: set[int] | None = None,
        canary_size: int | None = None,
        image_size: int = 128,
        split_seed: int = 1234,
        character_set: Sequence[int] | None = None,
    ):
        self.target_codepoints = set(target_codepoints) if target_codepoints else None
        if character_set is None:
            character_set = LATIN_CORE
        self._character_set: list[int] = sorted(set(character_set))
        having_filter: set[int] | None = None
        if having is not None:
            having_filter = set(having)

        self.googlefonts = GoogleFonts(
            repo_url,
            having=having_filter,
            # Canary mode only needs the first few fonts; don't scan the whole
            # repository to get them.
            max_fonts=canary_size,
        )
        self.filter_fonts()
        self.batch_size = batch_size
        self.image_size = image_size
        self.split_seed = split_seed
        # Keep data-order randomization reproducible without forcing fixed batches.
        self._train_loader_generator = torch.Generator()
        self._train_loader_generator.manual_seed(self.split_seed + 1)
        self._test_loader_generator = torch.Generator()
        self._test_loader_generator.manual_seed(self.split_seed + 2)

        # Test chars are a random split from the character set.
        _, self.test_codepoints = train_test_split(
            self._character_set,
            random_state=self.split_seed,
        )

        self.canary_size = canary_size

        if canary_size is not None:
            fonts = self.googlefonts.fonts[:canary_size]
        else:
            fonts = self.googlefonts.fonts

        self.train_fonts, self.test_fonts = self._split_fonts_by_family(
            fonts,
            split_seed=self.split_seed,
        )
        print("Train fonts:", len(self.train_fonts))
        print("Test fonts:", len(self.test_fonts))

    @staticmethod
    def _split_fonts_by_family(fonts, *, split_seed: int):
        """Split fonts into train/test by family to avoid cross-style leakage."""
        if len(fonts) < 2:
            return list(fonts), []

        family_to_fonts = defaultdict(list)
        for font in fonts:
            family_to_fonts[font.family].append(font)

        families = sorted(family_to_fonts.keys())
        if len(families) < 2:
            # If only one family is available, keep current behaviour and avoid empty train.
            return list(fonts), []

        train_families, test_families = train_test_split(
            families,
            random_state=split_seed,
        )

        train_family_set = set(train_families)
        test_family_set = set(test_families)

        train_fonts = [
            font for family in train_family_set for font in family_to_fonts[family]
        ]
        test_fonts = [
            font for family in test_family_set for font in family_to_fonts[family]
        ]
        return train_fonts, test_fonts

    def filter_fonts(self):
        pass

    def train_set(self):
        return Dataset(
            self.train_fonts, codepoint_filter_fn=self.train_codepoint_filter
        )

    def test_set(self):
        return Dataset(self.test_fonts, codepoint_filter_fn=self.test_codepoint_filter)

    def train_codepoint_filter(self, font_codepoints: set[int]) -> set[int]:
        if self.target_codepoints is not None:
            return set(font_codepoints) & self.target_codepoints
        return set(font_codepoints) - set(self.test_codepoints)

    def test_codepoint_filter(self, font_codepoints: set[int]) -> set[int]:
        if self.target_codepoints is not None:
            return set(font_codepoints) & self.target_codepoints
        return set(font_codepoints) & set(self.test_codepoints)

    def train_loader(self):
        return DataLoader(
            self.train_set(),
            batch_size=self.batch_size,
            shuffle=True,
            generator=self._train_loader_generator,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def test_loader(self):
        return DataLoader(
            self.test_set(),
            batch_size=self.batch_size,
            shuffle=True,
            generator=self._test_loader_generator,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        raise NotImplementedError("Base DatasetMaker does not implement collate_fn")


class Dataset(TorchDataset):
    def __init__(self, fonts, codepoint_filter_fn: Callable[[set[int]], set[int]]):
        self.fonts = fonts
        self.codepoint_filter_fn = codepoint_filter_fn
        self.order = []
        for font in self.fonts:
            hb_font = _hb_font_for_face(font.hb_face)
            chars = self.codepoint_filter_fn(set(font.codepoints))
            for char in chars:
                # Skip empty glyphs; they can destabilize training targets.
                gid = hb_font.get_nominal_glyph(char)
                extents = hb_font.get_glyph_extents(gid)
                if _has_non_empty_outline(extents):
                    self.order.append((font, char))

    def __len__(self):
        return len(self.order)

    def __getitem__(self, idx):
        font, char = self.order[idx]
        return {
            "char": char,
            "font": font,
        }


class AllGidsDataset(TorchDataset):
    def __init__(self, fonts):
        self.fonts = fonts
        self.order = []
        for font in self.fonts:
            hb_font = _hb_font_for_face(font.hb_face)
            for gid in range(1, font.hb_face.glyph_count):
                # Skip empty glyphs; they can destabilize training targets.
                extents = hb_font.get_glyph_extents(gid)
                if _has_non_empty_outline(extents):
                    self.order.append((font, gid))

    def __len__(self):
        return len(self.order)

    def __getitem__(self, idx):
        font, gid = self.order[idx]
        return {
            "gid": gid,
            "font": font,
        }


# ---------------------------------------------------------------------------
# Stratified font-instance sampling
# ---------------------------------------------------------------------------
#
# Turns a Google Fonts checkout into a balanced, stratified list of font
# instances (a ``(file, weight, style, axis_position)`` row).  The selection
# balances text (sans/serif) against fancy (display/script/handwriting)
# instances, keeps a spread of in-family *contrast* (e.g. 100 + 900, never
# 400 + 500), synthesises variable-font locations, and optionally prefers
# families for which :meth:`StratifiedFontSampler.use_font` returns True.
#
# The sampling unit is a ``(family, style)`` pair, so roman and italic are
# sampled independently.


@dataclass
class Instance:
    """One ``(font file, weight, style)`` row within a sampling unit."""

    path: str  # repo-relative path
    weight: int  # weight class (100..900)
    style: str  # "normal" | "italic"
    variable: bool
    axes: list[list] | None  # fvar axes if variable, else None
    axis_position: list[float] | None  # set only for synthesised locations
    has_target: bool  # ``use_font`` returned True for this font
    coverage: int = 0  # number of needed codepoints present


@dataclass
class Unit:
    """A ``(family, style)`` sampling unit."""

    family: str
    style: str
    bucket: str
    instances: list[Instance] = field(default_factory=list)
    classification: str = ""
    tag_category: str = ""

    def stratum(self) -> str:
        if self.bucket != "fancy":
            return self.bucket
        if self.tag_category == "Script":
            return "script"
        if "HANDWRITING" in self.classification:
            return "handwriting"
        return "display"

    def static_weights(self) -> list[int]:
        return sorted({i.weight for i in self.instances if not i.variable})

    def weight_range(self) -> tuple[int, int] | None:
        ws = [i.weight for i in self.instances]
        for i in self.instances:
            wght = next((a for a in (i.axes or []) if a[0] == "wght"), None)
            if wght:
                ws += [int(wght[1]), int(wght[3])]
        if not ws:
            return None
        return min(ws), max(ws)

    def has_target(self) -> bool:
        return any(i.has_target for i in self.instances)

    def max_coverage(self) -> int:
        return max((i.coverage for i in self.instances), default=0)

    def key(self) -> str:
        return f"{self.family}::{self.style}"


@dataclass
class SelectedInstance:
    """A chosen training instance (one row per copy from replacement fill)."""

    unit: str
    family: str
    bucket: str
    stratum: str
    path: str
    weight: int
    style: str
    style_bucket: int
    variable: bool
    axis_position: list[float] | None
    has_target: bool
    copy: int = 0


# -- Font metadata helpers ----------------------------------------------------


def parse_metadata_weights(metadata_pb: Path) -> dict[str, tuple[int, str]]:
    """Return ``filename -> (weight, style)`` from a family METADATA.pb."""
    if not metadata_pb.exists():
        return {}
    txt = metadata_pb.read_text(encoding="utf-8", errors="replace")
    out: dict[str, tuple[int, str]] = {}
    for block in re.findall(r"fonts\s*\{(.*?)\n\s*\}", txt, re.DOTALL):
        fn = re.search(r'filename:\s*"([^"]+)"', block)
        wt = re.search(r"weight:\s*(\d+)", block)
        st = re.search(r'style:\s*"([^"]+)"', block)
        if fn:
            out[fn.group(1)] = (
                int(wt.group(1)) if wt else 400,
                st.group(1) if st else "normal",
            )
    return out


def fvar_axes(path: Path) -> list[list] | None:
    """Return ``[[tag, min, default, max], ...]`` if the font is variable."""
    if "[" not in path.name:
        return None
    from fontTools.ttLib import TTFont

    try:
        return [
            [a.axisTag, a.minValue, a.defaultValue, a.maxValue]
            for a in TTFont(path, lazy=True)["fvar"].axes
        ]
    except Exception:
        return None


def style_bucket(font: GoogleFont) -> str:
    """Coarse style bucket from ``METADATA.pb``'s top-level classification.

    The tag-based ``category()`` cannot return Handwriting and treats Display as
    a default, so we use ``classification()`` here.
    """
    cls = font.classification()
    if "MONOSPACE" in cls:
        return "mono"
    if "HANDWRITING" in cls or "DISPLAY" in cls:
        return "fancy"
    if "SANS_SERIF" in cls:
        return "sans"
    if "SERIF" in cls:
        return "serif"
    return "other"


def units_to_dicts(units: Sequence[Unit]) -> list[dict]:
    """Serialise units (for an on-disk cache)."""
    return [
        {
            "family": u.family,
            "style": u.style,
            "bucket": u.bucket,
            "classification": u.classification,
            "tag_category": u.tag_category,
            "instances": [asdict(i) for i in u.instances],
        }
        for u in units
    ]


def units_from_dicts(data: Sequence[dict]) -> list[Unit]:
    """Deserialise units previously written by :func:`units_to_dicts`."""
    units = []
    for u in data:
        unit = Unit(
            family=u["family"],
            style=u["style"],
            bucket=u["bucket"],
            classification=u.get("classification", ""),
            tag_category=u.get("tag_category", ""),
        )
        unit.instances = [Instance(**i) for i in u["instances"]]
        units.append(unit)
    return units


# -- Contrast selection within a unit ------------------------------------------


def _axis_position(axes: list[list], weight: int) -> list[float]:
    """Full design-coordinate list in fvar order, defaulting non-wght axes."""
    return [
        float(min(max(weight, lo), hi)) if tag == "wght" else float(default)
        for tag, lo, default, hi in axes
    ]


def _match_instance(unit: Unit, target: int) -> Instance | None:
    for i in unit.instances:
        if not i.variable and i.weight == target:
            return i
    for i in unit.instances:
        axes = i.axes or []
        wght = next((a for a in axes if a[0] == "wght"), None)
        if wght and wght[1] <= target <= wght[3]:
            return Instance(
                path=i.path,
                weight=target,
                style=i.style,
                variable=True,
                axes=axes,
                axis_position=_axis_position(axes, target),
                has_target=i.has_target,
                coverage=i.coverage,
            )
    statics = [i for i in unit.instances if not i.variable]
    if statics:
        return min(statics, key=lambda i: abs(i.weight - target))
    return None


def _regular_instance(unit: Unit) -> Instance | None:
    """The instance a unit contributes to the Regular-only dataset.

    A single-static-font family (exactly one static file, no variable file)
    contributes that file as-is — its declared weight is the family's only
    weight, so that weight is the output we generate.  Every other unit
    contributes its weight-400 (Regular) instance: a static file declared at
    400, or a variable file instantiated at a synthesized weight-400 location
    (explicitly, regardless of the ``wght`` axis origin/default).
    """
    statics = [i for i in unit.instances if not i.variable]
    variables = [i for i in unit.instances if i.variable]
    if len(statics) == 1 and not variables:
        return statics[0]
    for i in statics:
        if i.weight == 400:
            return i
    for i in variables:
        axes = i.axes or []
        wght = next((a for a in axes if a[0] == "wght"), None)
        if wght and wght[1] <= 400 <= wght[3]:
            return Instance(
                path=i.path,
                weight=400,
                style=i.style,
                variable=True,
                axes=axes,
                axis_position=_axis_position(axes, 400),
                has_target=i.has_target,
                coverage=i.coverage,
            )
    return None


def contrast_instances(unit: Unit, k: int) -> list[Instance]:
    """Choose ``k`` instances of maximal weight contrast within one unit."""
    rng = unit.weight_range()
    if rng is None:
        return []
    wmin, wmax = rng
    if wmin == wmax:
        return unit.instances[:1]

    k = max(1, k)
    if k == 1:
        pool = [i for i in unit.instances if i.has_target] or unit.instances
        return [min(pool, key=lambda i: (abs(i.weight - 400), i.variable))]

    targets = [round(wmin + (wmax - wmin) * t / (k - 1)) for t in range(k)]
    chosen: list[Instance] = []
    used: set[tuple] = set()
    for target in targets:
        inst = _match_instance(unit, target)
        if inst is None:
            continue
        key = (inst.path, inst.weight, inst.style, tuple(inst.axis_position or ()))
        if key in used:
            continue
        used.add(key)
        chosen.append(inst)
    return chosen


def _has_continuous_wght(unit: Unit) -> bool:
    """True if any variable instance has a non-degenerate ``wght`` axis.

    A continuous ``wght`` range means the family can be synthesised at arbitrary
    weights, so it can supply a full contrast ladder up to ``max_per_unit``
    (not just the two endpoints).
    """
    for i in unit.instances:
        wght = next((a for a in (i.axes or []) if a[0] == "wght"), None)
        if wght and wght[1] < wght[3]:
            return True
    return False


def contrast_capacity(unit: Unit, max_per_unit: int) -> int:
    """How many contrasting instances this unit can supply."""
    rng = unit.weight_range()
    if rng is None:
        return 0
    wmin, wmax = rng
    if wmin == wmax:
        return 1
    if _has_continuous_wght(unit):
        return max_per_unit
    return min(len(unit.static_weights()), max_per_unit)


def target_contrast_capacity(unit: Unit, max_per_unit: int) -> int:
    """Contrasting instances that all contain the target glyph."""
    if not unit.has_target():
        return 0
    rng = unit.weight_range()
    if rng is None:
        return 0
    wmin, wmax = rng
    if wmin == wmax:
        return 1
    # A variable font whose file contains the target glyph carries it at every
    # synthesised weight, so it can supply a full contrast ladder.
    for i in unit.instances:
        if not i.has_target:
            continue
        wght = next((a for a in (i.axes or []) if a[0] == "wght"), None)
        if wght and wght[1] < wght[3]:
            return max_per_unit
    return min(
        len({i.weight for i in unit.instances if not i.variable and i.has_target}),
        max_per_unit,
    )


# -- Stratified allocation ----------------------------------------------------


def _allocate_stratum(
    pool: list[Unit],
    target: int,
    *,
    max_per_unit: int,
    avg_instances: float,
    rng: random.Random,
    prefer_multi_target: bool,
) -> list[tuple[Unit, int]]:
    if target <= 0 or not pool:
        return []

    def sort_key(unit: Unit):
        tcap = target_contrast_capacity(unit, max_per_unit)
        cap = contrast_capacity(unit, max_per_unit)
        pref = (tcap >= 2, cap >= 2) if prefer_multi_target else (cap >= 2,)
        return (tuple(-int(b) for b in pref), rng.random())

    ordered = sorted(pool, key=sort_key)
    n_units = min(len(ordered), max(1, round(target / max(avg_instances, 1.0))))
    chosen = ordered[:n_units]
    counts = {id(u): 1 for u in chosen}
    total = len(chosen)
    idx = n_units
    while total < target and idx < len(ordered):
        unit = ordered[idx]
        counts[id(unit)] = 1
        chosen.append(unit)
        total += 1
        idx += 1
    progress = True
    while total < target and progress:
        progress = False
        for unit in chosen:
            if total >= target:
                break
            cap = min(max_per_unit, max(contrast_capacity(unit, max_per_unit), 1))
            if counts[id(unit)] < cap:
                counts[id(unit)] += 1
                total += 1
                progress = True
    return [(unit, counts[id(unit)]) for unit in chosen]


def _stratum_capacity(pool: list[Unit], max_per_unit: int, target_frac: float) -> int:
    with_target = [u for u in pool if u.has_target()]
    if not with_target or target_frac <= 0:
        eligible = pool
    elif target_frac >= 1.0:
        eligible = with_target
    else:
        without = [u for u in pool if not u.has_target()]
        n_other = round(len(with_target) * (1 - target_frac) / target_frac)
        eligible = with_target + without[:n_other]
    return sum(
        min(max_per_unit, max(contrast_capacity(u, max_per_unit), 1)) for u in eligible
    )


def _normalise_targets(desired: dict[str, int], n: int) -> dict[str, int]:
    out = dict(desired)
    keys = list(out)
    diff = n - sum(out.values())
    i = 0
    while diff != 0 and keys:
        st = keys[i % len(keys)]
        if diff > 0:
            out[st] += 1
            diff -= 1
        elif out[st] > 0:
            out[st] -= 1
            diff += 1
        i += 1
    return out


class StratifiedFontSampler:
    """Select a balanced, stratified set of font instances from a font library.

    Subclass and override :meth:`use_font` to prefer certain fonts (for example
    fonts that contain an encoded rupee glyph).  The strata mix and the
    text/fancy split are class attributes and may also be overridden.
    """

    STRATA = ("sans", "serif", "display", "script", "handwriting", "mono", "other")
    TEXT_STRATA = ("sans", "serif")
    FANCY_STRATA = ("display", "script", "handwriting")

    # Default stratum mix: text (sans/serif) 50%, fancy 50% split display/script/
    # handwriting.  Handwriting is capacity-limited because few families with the
    # target glyph exist; the allocator redistributes or oversamples as configured.
    DEFAULT_STRATA: ClassVar[dict[str, float]] = {
        "sans": 0.25,
        "serif": 0.25,
        "display": 0.20,
        "script": 0.20,
        "handwriting": 0.10,
    }

    def use_font(self, font: GoogleFont) -> bool:
        """Return True for fonts that should be *preferred* during selection.

        The default implementation prefers nothing.  Subclasses override this to
        express a target predicate (e.g. ``return RUPEE in font.codepoints``).
        """
        return False

    def build_units(
        self, gf: GoogleFonts, needed_codepoints: set[int] | None = None
    ) -> list[Unit]:
        """Build ``(family, style)`` sampling units from a Google Fonts checkout."""
        needed = needed_codepoints or set()
        grouped: dict[str, list[GoogleFont]] = defaultdict(list)
        for f in gf.fonts:
            grouped[f.family].append(f)

        units: list[Unit] = []
        for fam, fs in grouped.items():
            meta = parse_metadata_weights(fs[0].path.parent / "METADATA.pb")
            by_style: dict[str, list[tuple[GoogleFont, int]]] = defaultdict(list)
            for f in fs:
                weight, style = meta.get(f.path.name, (400, "normal"))
                by_style[style].append((f, weight))
            for style, entries in by_style.items():
                unit = Unit(
                    family=fam,
                    style=style,
                    bucket=style_bucket(entries[0][0]),
                    classification=entries[0][0].classification(),
                    tag_category=entries[0][0].category(),
                )
                for f, weight in entries:
                    cps = f.codepoints
                    axes = fvar_axes(f.path)
                    unit.instances.append(
                        Instance(
                            path=str(f.path.relative_to(gf.repo_path)).replace(
                                os.sep, "/"
                            ),
                            weight=weight,
                            style=style,
                            variable=axes is not None,
                            axes=axes,
                            axis_position=None,
                            has_target=self.use_font(f),
                            coverage=len(cps & needed) if needed else 0,
                        )
                    )
                units.append(unit)
        return units

    def select_subset(
        self,
        units: list[Unit],
        n: int,
        strata_fracs: dict[str, float] | None = None,
        *,
        max_per_unit: int = 3,
        avg_instances: float = 1.6,
        target_frac: float = 0.0,
        prefer_multi_target: bool = False,
        replacement: bool = True,
        min_coverage: int = 0,
        regular_only: bool = False,
        seed: int = 1234,
    ) -> tuple[list[SelectedInstance], dict]:
        """Select ``n`` stratified training instances.

        Returns ``(instances, report)``.  ``min_coverage`` drops units whose
        fonts cover fewer than that many of the needed codepoints (so subset
        budget is not spent on fonts that cannot train the vocabulary).  When
        ``regular_only`` is set, each ``(family, style)`` unit contributes only
        its Regular instance (weight 400 for multi-weight/variable families; the
        family's sole weight for single-weight static families).
        """
        strata_fracs = dict(
            strata_fracs if strata_fracs is not None else self.DEFAULT_STRATA
        )
        rng = random.Random(seed)
        eligible = [
            u for u in units if min_coverage <= 0 or u.max_coverage() >= min_coverage
        ]
        if regular_only:
            trimmed: list[Unit] = []
            for u in eligible:
                inst = _regular_instance(u)
                if inst is None:
                    continue
                trimmed.append(
                    Unit(
                        family=u.family,
                        style=u.style,
                        bucket=u.bucket,
                        classification=u.classification,
                        tag_category=u.tag_category,
                        instances=[inst],
                    )
                )
            eligible = trimmed
        by_stratum: dict[str, list[Unit]] = defaultdict(list)
        for unit in eligible:
            by_stratum[unit.stratum()].append(unit)

        desired = {st: round(n * frac) for st, frac in strata_fracs.items()}
        capacity = {
            st: _stratum_capacity(by_stratum.get(st, []), max_per_unit, target_frac)
            for st in desired
        }
        if replacement:
            final = _normalise_targets(desired, n)
        else:
            final = {st: min(desired[st], capacity[st]) for st in desired}
            deficit = n - sum(final.values())
            if deficit > 0:
                spare = {
                    st: capacity[st] - final[st]
                    for st in final
                    if desired[st] > 0 and capacity[st] - final[st] > 0
                }
                total_spare = sum(spare.values())
                if total_spare > 0:
                    shares = {st: deficit * spare[st] / total_spare for st in spare}
                    base = {st: int(shares[st]) for st in spare}
                    remainder = deficit - sum(base.values())
                    for st in sorted(
                        spare, key=lambda s: shares[s] - base[s], reverse=True
                    ):
                        if remainder <= 0:
                            break
                        if base[st] < spare[st]:
                            base[st] += 1
                            remainder -= 1
                    for st, add in base.items():
                        final[st] += min(add, spare[st])

        selected: list[SelectedInstance] = []
        for st, target in final.items():
            if target <= 0:
                continue
            pool = by_stratum.get(st, [])
            n_units_needed = max(1, round(target / max(avg_instances, 1.0)))
            with_target = [u for u in pool if u.has_target()]
            without_target = [u for u in pool if not u.has_target()]
            rng.shuffle(with_target)
            rng.shuffle(without_target)
            n_other = min(
                len(without_target), max(0, round((1 - target_frac) * n_units_needed))
            )
            n_with = min(len(with_target), max(0, n_units_needed - n_other))
            allowed = with_target[:n_with] + without_target[:n_other]
            without_rem = without_target[n_other:] if target_frac < 1.0 else []
            extra = with_target[n_with:] + without_rem
            cap_sum = sum(
                min(max_per_unit, max(contrast_capacity(u, max_per_unit), 1))
                for u in allowed
            )
            for unit in extra:
                if cap_sum >= target:
                    break
                allowed.append(unit)
                cap_sum += min(
                    max_per_unit, max(contrast_capacity(unit, max_per_unit), 1)
                )
            if not allowed:
                continue
            picks = _allocate_stratum(
                allowed,
                target,
                max_per_unit=max_per_unit,
                avg_instances=avg_instances,
                rng=rng,
                prefer_multi_target=prefer_multi_target,
            )
            pairs: list[tuple[Unit, Instance]] = []
            for unit, k in picks:
                pairs += [(unit, inst) for inst in contrast_instances(unit, k)]

            # Replacement fill: resample units (weighted toward multi-contrast
            # ones) and duplicate their instances when distinct capacity is short.
            if replacement and len(pairs) < target:
                cands = [u for u in allowed if contrast_capacity(u, max_per_unit) >= 1]
                weights = [
                    3 if target_contrast_capacity(u, max_per_unit) >= 2 else 1
                    for u in cands
                ]
                while len(pairs) < target and cands:
                    unit = rng.choices(cands, weights=weights, k=1)[0]
                    k = min(
                        max_per_unit,
                        max(contrast_capacity(unit, max_per_unit), 1),
                        target - len(pairs),
                    )
                    cs = contrast_instances(unit, k)
                    if not cs:
                        break
                    pairs += [(unit, inst) for inst in cs]

            seen: Counter = Counter()
            for unit, inst in pairs:
                key = (
                    unit.family,
                    inst.path,
                    inst.weight,
                    inst.style,
                    tuple(inst.axis_position or ()),
                )
                copy_index = seen[key]
                seen[key] += 1
                selected.append(
                    SelectedInstance(
                        unit=unit.key(),
                        family=unit.family,
                        bucket=unit.bucket,
                        stratum=st,
                        path=inst.path,
                        weight=inst.weight,
                        style=inst.style,
                        style_bucket=0 if inst.style == "normal" else 1,
                        variable=inst.variable,
                        axis_position=inst.axis_position,
                        has_target=inst.has_target,
                        copy=copy_index,
                    )
                )

        return selected, self.summarise_subset(selected, n, seed)

    def summarise_subset(
        self, selected: list[SelectedInstance], n: int, seed: int
    ) -> dict:
        """Aggregate report over a selection (used by the CLI and for logging)."""
        distinct = [s for s in selected if s.copy == 0]
        inst_by_st = Counter(s.stratum for s in selected)
        per_unit = Counter(s.unit for s in distinct)
        per_family = Counter(s.family for s in distinct)
        fam_names = set(per_family)
        target_fams = {
            f
            for f in fam_names
            if any(s.has_target for s in distinct if s.family == f)
        }
        multi_units = {u: c for u, c in per_unit.items() if c >= 2}
        unit_family = {s.unit: s.family for s in distinct}
        fam_units = Counter(unit_family[u] for u in set(per_unit))
        spans = [
            max(x.weight for x in distinct if x.unit == u)
            - min(x.weight for x in distinct if x.unit == u)
            for u in multi_units
        ]
        style_counts = Counter(s.style for s in distinct)
        return {
            "seed": seed,
            "requested": n,
            "n_instances": len(selected),
            "distinct_instances": len(distinct),
            "oversampled_instances": len(selected) - len(distinct),
            "families": len(fam_names),
            "units": len(per_unit),
            "instances_per_stratum": dict(inst_by_st),
            "text_instances": sum(inst_by_st[st] for st in self.TEXT_STRATA),
            "fancy_instances": sum(inst_by_st[st] for st in self.FANCY_STRATA),
            "unit_k_histogram": dict(sorted(Counter(per_unit.values()).items())),
            "family_k_histogram": dict(sorted(Counter(per_family.values()).items())),
            "families_with_target": len(target_fams),
            "target_family_frac": len(target_fams) / len(fam_names)
            if fam_names
            else 0.0,
            "variable_instances": sum(s.variable for s in distinct),
            "style_counts": dict(style_counts),
            "multi_instance_units": len(multi_units),
            "families_with_both_styles": sum(1 for c in fam_units.values() if c >= 2),
            "mean_multi_unit_weight_span": (
                sum(spans) / len(spans) if spans else 0.0
            ),
        }
