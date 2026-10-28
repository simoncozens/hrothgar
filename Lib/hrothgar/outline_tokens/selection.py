"""Select unambiguous, deduplicated designs for the n-way style probe.

A *design* is a cluster of families sharing an identical ``e`` outline (same
normalized-outline fingerprint).  Families are grouped by fingerprint, tags are
merged across the group (per-tag max, so an unassessed duplicate of an assessed
design inherits the assessed tag), and a design is kept iff exactly one target
style-category tag is above ``threshold`` and no *other* style-category tag is.

This is the leave-one-design-out unit: a whole design (all its width/script
variants) is held out atomically, so the probe cannot coast on a near-identical
outline left in the training fold.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import uharfbuzz as hb

from hrothgar.googlefonts import GoogleFonts
from hrothgar.outline_tokens import outline_fingerprint
from hrothgar.similarity.dataset import STYLE_CATEGORY_TAGS

# The fine styles we split apart.  Index == label id.
TARGET_TAGS = [
    "/Sans/Geometric",
    "/Sans/Superellipse",
    "/Sans/Humanist",
    "/Serif/Transitional",
]

CLASSES = list(TARGET_TAGS)


@dataclass
class Design:
    name: str
    members: list[str] = field(default_factory=list)
    font: object = None
    label: int = -1
    tags: dict[str, float] = field(default_factory=dict)


def _regular_font(fonts: list) -> object:
    for f in fonts:
        if "regular" in f.path.name.lower():
            return f
    non_italic = [f for f in fonts if "italic" not in f.path.name.lower()]
    if non_italic:
        return sorted(non_italic, key=lambda f: str(f.path))[0]
    return fonts[0]


def _by_family(gf: GoogleFonts) -> dict[str, list]:
    by_family: dict[str, list] = defaultdict(list)
    for font in gf.fonts:
        by_family[font.family].append(font)
    return by_family


def _family_tags(gf: GoogleFonts) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for font in gf.fonts:
        if font.family not in out:
            out[font.family] = font.tags()
    return out


def _active(tags: dict[str, float], threshold: float) -> set[str]:
    return {t for t in STYLE_CATEGORY_TAGS if tags.get(t, 0.0) > threshold}


def _merged_tags(members: list[str], fam_tags: dict) -> dict[str, float]:
    merged: dict[str, float] = {}
    for m in members:
        for tag, v in fam_tags[m].items():
            if v > merged.get(tag, 0.0):
                merged[tag] = v
    return merged


def _rep(members: list[str], fam_tags: dict) -> str:
    assessed = [m for m in members if fam_tags.get(m)]
    return min(assessed or members, key=lambda n: (len(n), n))


def select_designs(
    gf: GoogleFonts,
    glyph: str = "e",
    threshold: float = 50.0,
    precision: int = 4,
) -> list[Design]:
    """Return the clean (unambiguous) designs, one per dedupe group, with labels."""
    by_family = _by_family(gf)
    fam_tags = _family_tags(gf)
    families = list(fam_tags)

    groups: dict[tuple, list[str]] = defaultdict(list)
    cp = ord(glyph)
    for family in families:
        font = _regular_font(by_family[family])
        gid = hb.Font(font.hb_face).get_nominal_glyph(cp)
        if gid is None:
            groups[("__nogid__", family)].append(family)
            continue
        groups[outline_fingerprint(font.hb_face, gid, precision=precision)].append(family)

    target_set = set(TARGET_TAGS)
    designs: list[Design] = []
    for members in groups.values():
        tags = _merged_tags(members, fam_tags)
        active = _active(tags, threshold)
        active_targets = active & target_set
        if active == active_targets and len(active_targets) == 1:
            label = TARGET_TAGS.index(next(iter(active_targets)))
            rep = _rep(members, fam_tags)
            designs.append(
                Design(rep, list(members), _regular_font(by_family[rep]), label, tags)
            )

    designs.sort(key=lambda d: d.name)
    return designs
