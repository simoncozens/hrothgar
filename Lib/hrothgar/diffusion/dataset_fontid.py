"""Full-dataset maker for the factorized (codepoint, font-instance) diffusion model.

Each training item is a ``(glyph image, codepoint index, font instance)`` triple.
A font instance is a ``(file, weight, style, axis_position)`` row — for a static
family this is one weight file; for a variable family it may be a synthesised
``wght`` location on a single file.

The subset of font instances used for training is chosen by a **stratified
sampler** (see :func:`select_subset`) rather than by truncating the load order.
The sampler balances text (sans/serif) against fancy (display/script/handwriting)
instances, keeps a spread of in-family *contrast* (e.g. 100 + 900, never
400 + 500), synthesises variable-font locations, and targets families that
contain the acceptance glyph (₹).

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

import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from hrothgar.dataset import (
    _has_non_empty_outline,
    _hb_font_for_face,
)
from hrothgar.dataset_constants import LATIN_KERNEL
from hrothgar.glyph_rendering import geometry_tensor
from hrothgar.googlefonts import GoogleFont, GoogleFonts, StandaloneFont
from hrothgar.render_utils import render_glyph_with_geometry

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))

RUPEE = ord("\u20b9")  # U+20B9 — the acceptance glyph
STRATA = ("sans", "serif", "display", "script", "handwriting", "mono", "other")
TEXT_STRATA = ("sans", "serif")
FANCY_STRATA = ("display", "script", "handwriting")

# Default stratum mix: text (sans/serif) 50%, fancy 50% split display/script/
# handwriting.  Handwriting is capacity-limited because few families with ₹
# exist; the allocator redistributes or oversamples as configured.
DEFAULT_STRATA = {
    "sans": 0.25,
    "serif": 0.25,
    "display": 0.20,
    "script": 0.20,
    "handwriting": 0.10,
}


# ---------------------------------------------------------------------------
# Font metadata helpers
# ---------------------------------------------------------------------------


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


def fvar_axes(path: Path) -> Optional[list[list]]:
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


# ---------------------------------------------------------------------------
# Sampling units
# ---------------------------------------------------------------------------


@dataclass
class Instance:
    """One ``(font file, weight, style)`` row within a sampling unit."""

    path: str  # repo-relative path
    weight: int  # weight class (100..900)
    style: str  # "normal" | "italic"
    variable: bool
    axes: Optional[list[list]]  # fvar axes if variable, else None
    axis_position: Optional[list[float]]  # set only for synthesised locations
    has_target: bool  # font contains U+20B9
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

    def weight_range(self) -> Optional[tuple[int, int]]:
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
    weight_norm: float
    style: str
    style_bucket: int
    variable: bool
    axis_position: Optional[list[float]]
    has_target: bool
    copy: int = 0


def build_units(
    gf: GoogleFonts, needed_codepoints: Optional[set[int]] = None
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
                        path=str(f.path.relative_to(gf.repo_path)).replace(os.sep, "/"),
                        weight=weight,
                        style=style,
                        variable=axes is not None,
                        axes=axes,
                        axis_position=None,
                        has_target=RUPEE in cps,
                        coverage=len(cps & needed) if needed else 0,
                    )
                )
            units.append(unit)
    return units


def units_to_dicts(units: Sequence[Unit]) -> list[dict]:
    """Serialise units (for the CLI's on-disk cache)."""
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
            family=u["family"], style=u["style"], bucket=u["bucket"],
            classification=u.get("classification", ""),
            tag_category=u.get("tag_category", ""),
        )
        unit.instances = [Instance(**i) for i in u["instances"]]
        units.append(unit)
    return units


# ---------------------------------------------------------------------------
# Contrast selection within a unit
# ---------------------------------------------------------------------------


def _axis_position(axes: list[list], weight: int) -> list[float]:
    """Full design-coordinate list in fvar order, defaulting non-wght axes."""
    return [
        float(min(max(weight, lo), hi)) if tag == "wght" else float(default)
        for tag, lo, default, hi in axes
    ]


def _match_instance(unit: Unit, target: int) -> Optional[Instance]:
    for i in unit.instances:
        if not i.variable and i.weight == target:
            return i
    for i in unit.instances:
        axes = i.axes or []
        wght = next((a for a in axes if a[0] == "wght"), None)
        if wght and wght[1] <= target <= wght[3]:
            return Instance(
                path=i.path, weight=target, style=i.style, variable=True,
                axes=axes, axis_position=_axis_position(axes, target),
                has_target=i.has_target, coverage=i.coverage,
            )
    statics = [i for i in unit.instances if not i.variable]
    if statics:
        return min(statics, key=lambda i: abs(i.weight - target))
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


def contrast_capacity(unit: Unit, max_per_unit: int) -> int:
    """How many contrasting instances this unit can supply."""
    rng = unit.weight_range()
    if rng is None:
        return 0
    wmin, wmax = rng
    if wmin == wmax:
        return 1
    points = set(unit.static_weights())
    for i in unit.instances:
        wght = next((a for a in (i.axes or []) if a[0] == "wght"), None)
        if wght:
            points |= {int(wght[1]), int(wght[3])}
    return min(len(points), max_per_unit)


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
    pts = {i.weight for i in unit.instances if not i.variable and i.has_target}
    for i in unit.instances:
        if not i.has_target:
            continue
        wght = next((a for a in (i.axes or []) if a[0] == "wght"), None)
        if wght:
            pts |= {int(wght[1]), int(wght[3])}
    return min(len(pts), max_per_unit)


# ---------------------------------------------------------------------------
# Stratified allocation
# ---------------------------------------------------------------------------


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
    if target_frac >= 1.0:
        eligible = with_target
    else:
        without = [u for u in pool if not u.has_target()]
        n_other = round(len(with_target) * (1 - target_frac) / max(target_frac, 1e-9))
        eligible = with_target + without[:n_other]
    return sum(min(max_per_unit, max(contrast_capacity(u, max_per_unit), 1))
               for u in eligible)


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


def select_subset(
    units: list[Unit],
    n: int,
    strata_fracs: dict[str, float],
    *,
    max_per_unit: int = 3,
    avg_instances: float = 1.6,
    target_frac: float = 0.7,
    prefer_multi_target: bool = True,
    replacement: bool = True,
    min_coverage: int = 0,
    seed: int = 1234,
) -> tuple[list[SelectedInstance], dict]:
    """Select ``n`` stratified training instances.

    Returns ``(instances, report)``.  ``min_coverage`` drops units whose fonts
    cover fewer than that many of the needed codepoints (so subset budget is not
    spent on fonts that cannot train the vocabulary).
    """
    rng = random.Random(seed)
    eligible = [u for u in units
                if min_coverage <= 0 or u.max_coverage() >= min_coverage]
    by_stratum: dict[str, list[Unit]] = defaultdict(list)
    for unit in eligible:
        by_stratum[unit.stratum()].append(unit)

    desired = {st: round(n * frac) for st, frac in strata_fracs.items()}
    capacity = {st: _stratum_capacity(by_stratum.get(st, []), max_per_unit,
                                      target_frac)
                for st in desired}
    if replacement:
        final = _normalise_targets(desired, n)
    else:
        final = {st: min(desired[st], capacity[st]) for st in desired}
        deficit = n - sum(final.values())
        if deficit > 0:
            spare = {st: capacity[st] - final[st] for st in final
                     if desired[st] > 0 and capacity[st] - final[st] > 0}
            total_spare = sum(spare.values())
            if total_spare > 0:
                shares = {st: deficit * spare[st] / total_spare for st in spare}
                base = {st: int(shares[st]) for st in spare}
                remainder = deficit - sum(base.values())
                for st in sorted(spare, key=lambda s: shares[s] - base[s],
                                 reverse=True):
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
        n_other = min(len(without_target),
                      max(0, round((1 - target_frac) * n_units_needed)))
        n_with = min(len(with_target), max(0, n_units_needed - n_other))
        allowed = with_target[:n_with] + without_target[:n_other]
        without_rem = without_target[n_other:] if target_frac < 1.0 else []
        extra = with_target[n_with:] + without_rem
        cap_sum = sum(min(max_per_unit, max(contrast_capacity(u, max_per_unit), 1))
                      for u in allowed)
        for unit in extra:
            if cap_sum >= target:
                break
            allowed.append(unit)
            cap_sum += min(max_per_unit, max(contrast_capacity(unit, max_per_unit), 1))
        if not allowed:
            continue
        picks = _allocate_stratum(
            allowed, target, max_per_unit=max_per_unit,
            avg_instances=avg_instances, rng=rng,
            prefer_multi_target=prefer_multi_target,
        )
        pairs: list[tuple[Unit, Instance]] = []
        for unit, k in picks:
            pairs += [(unit, inst) for inst in contrast_instances(unit, k)]

        # Replacement fill: resample units (weighted toward multi-contrast ones)
        # and duplicate their instances when distinct capacity is short.
        if replacement and len(pairs) < target:
            cands = [u for u in allowed if contrast_capacity(u, max_per_unit) >= 1]
            weights = [
                3 if target_contrast_capacity(u, max_per_unit) >= 2 else 1
                for u in cands
            ]
            while len(pairs) < target and cands:
                unit = rng.choices(cands, weights=weights, k=1)[0]
                k = min(max_per_unit,
                        max(contrast_capacity(unit, max_per_unit), 1),
                        target - len(pairs))
                cs = contrast_instances(unit, k)
                if not cs:
                    break
                pairs += [(unit, inst) for inst in cs]

        seen: Counter = Counter()
        for unit, inst in pairs:
            key = (unit.family, inst.path, inst.weight, inst.style,
                   tuple(inst.axis_position or ()))
            copy_index = seen[key]
            seen[key] += 1
            selected.append(SelectedInstance(
                unit=unit.key(),
                family=unit.family,
                bucket=unit.bucket,
                stratum=st,
                path=inst.path,
                weight=inst.weight,
                weight_norm=(inst.weight - 400.0) / 400.0,
                style=inst.style,
                style_bucket=0 if inst.style == "normal" else 1,
                variable=inst.variable,
                axis_position=inst.axis_position,
                has_target=inst.has_target,
                copy=copy_index,
            ))

    return selected, summarise_subset(selected, n, seed)


def summarise_subset(selected: list[SelectedInstance], n: int, seed: int) -> dict:
    """Aggregate report over a selection (used by the CLI and for logging)."""
    distinct = [s for s in selected if s.copy == 0]
    inst_by_st = Counter(s.stratum for s in selected)
    per_unit = Counter(s.unit for s in distinct)
    per_family = Counter(s.family for s in distinct)
    fam_names = set(per_family)
    target_fams = {f for f in fam_names
                   if any(s.has_target for s in distinct if s.family == f)}
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
        "text_instances": sum(inst_by_st[st] for st in TEXT_STRATA),
        "fancy_instances": sum(inst_by_st[st] for st in FANCY_STRATA),
        "unit_k_histogram": dict(sorted(Counter(per_unit.values()).items())),
        "family_k_histogram": dict(sorted(Counter(per_family.values()).items())),
        "families_with_target": len(target_fams),
        "target_family_frac": len(target_fams) / len(fam_names) if fam_names else 0.0,
        "variable_instances": sum(s.variable for s in distinct),
        "style_counts": dict(style_counts),
        "multi_instance_units": len(multi_units),
        "families_with_both_styles": sum(1 for c in fam_units.values() if c >= 2),
        "mean_multi_unit_weight_span": (sum(spans) / len(spans) if spans else 0.0),
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class TrainInstance:
    """A unique training instance: a font file at a specific weight/style."""

    path: str  # repo-relative
    font: StandaloneFont
    axis_position: Optional[list[float]]
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
            instance.font, cp, self.image_size,
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
        character_set: Optional[Sequence[int]] = None,
        extra_codepoints: Optional[Sequence[int]] = None,
        remove_codepoints: Optional[Sequence[int]] = None,
        oversample_codepoints: Optional[dict[int, int]] = None,
        subset_n: Optional[int] = 200,
        strata: Optional[dict[str, float]] = None,
        max_per_unit: int = 3,
        avg_instances: float = 1.6,
        target_frac: float = 0.7,
        prefer_multi_target: bool = True,
        replacement: bool = True,
        subset_seed: int = 1234,
        heldout_fraction: float = 0.25,
        min_train_fonts_per_codepoint: int = 20,
        split_seed: int = 1234,
        units: Optional[list[Unit]] = None,
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
        if units is None:
            gf = GoogleFonts(str(self.repo))
            units = build_units(gf, needed_codepoints=base)
        strata_fracs = dict(strata or DEFAULT_STRATA)
        if subset_n is None or subset_n <= 0:
            # Use every instance of every eligible unit (full-library run).
            selected = _all_instances(units, min_train_fonts_per_codepoint + 1)
            self.subset_report = {"mode": "full", "n_instances": len(selected)}
        else:
            selected, self.subset_report = select_subset(
                units, subset_n, strata_fracs,
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
            self.instances.append(TrainInstance(
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
            ))

        self.font_meta = [inst.font_meta for inst in self.instances]
        self._build_pairs()
        print(
            f"Instances: {len(self.instances)} unique "
            f"({sum(i.copies for i in self.instances)} rows); "
            f"families: {self.num_families}; codepoints: {self.num_codepoints}"
        )
        print(f"Pairs: {len(self.train_pairs)} train / {len(self.val_pairs)} held-out")

    # -- pair construction ---------------------------------------------------

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
        dataset = _PairDataset(
            pairs, self.instances, self.cp_list, self.image_size
        )
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
        dataset = _PairDataset(
            pairs, self.instances, self.cp_list, self.image_size
        )
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


def _all_instances(units: list[Unit], min_coverage: int) -> list[SelectedInstance]:
    """Every instance of every eligible unit (full-library mode)."""
    out: list[SelectedInstance] = []
    for unit in units:
        if unit.max_coverage() < min_coverage:
            continue
        for inst in unit.instances:
            out.append(SelectedInstance(
                unit=unit.key(), family=unit.family, bucket=unit.bucket,
                stratum=unit.stratum(), path=inst.path, weight=inst.weight,
                weight_norm=(inst.weight - 400.0) / 400.0, style=inst.style,
                style_bucket=0 if inst.style == "normal" else 1,
                variable=inst.variable, axis_position=inst.axis_position,
                has_target=inst.has_target,
            ))
    return out
