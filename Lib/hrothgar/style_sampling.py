"""Style reference sampling — no torch dependency.

Separated from ``hrothgar.ar.dataset`` so that inference-only code
(Core ML, Glyphs.app plugin) can import these without pulling in torch.
"""

from __future__ import annotations

import math
import random
from typing import Optional, Sequence

import uharfbuzz as hb

from hrothgar.dataset_constants import LATIN_KERNEL


def _has_non_empty_glyph(font, codepoint: int) -> bool:
    """Return True if the font has a non-empty outline for codepoint."""
    if not hasattr(font, "hb_face"):
        return True
    hb_font = hb.Font(font.hb_face)  # type: ignore
    gid = hb_font.get_nominal_glyph(codepoint)
    extents = hb_font.get_glyph_extents(gid)
    if extents is None:
        return False
    return not all(x == 0 for x in extents)


def _font_has_codepoint(font, codepoint: int) -> bool:
    if hasattr(font, "has_codepoint"):
        return bool(font.has_codepoint(codepoint))
    return codepoint in getattr(font, "codepoints", set())


def _is_blank_rendering(rendered) -> bool:
    max_val = float(rendered.max())
    min_val = float(rendered.min())
    return max_val == min_val and (max_val == 1.0 or max_val == 0.0)


def _sample_style_codepoints(
    *,
    font,
    target_char: int,
    style_glyph_count: int,
    common_style_codepoints: Optional[Sequence[int]],
) -> list[int]:
    """Select style codepoints for one target item.

    By default this samples per-item codepoints from the same font, excluding
    the target codepoint. If a common list is provided, it is used first and
    then padded with per-font sampling when needed.
    """
    if style_glyph_count <= 0:
        raise ValueError(f"style_glyph_count must be positive, got {style_glyph_count}")

    available = [
        cp
        for cp in font.codepoints
        if cp != target_char and _has_non_empty_glyph(font, cp)
    ]
    available = [cp for cp in available if cp in LATIN_KERNEL]

    if not available:
        return [target_char] * style_glyph_count

    selected: list[int] = []
    if common_style_codepoints is not None:
        selected = [
            cp
            for cp in common_style_codepoints
            if cp != target_char
            and cp in font.codepoints
            and _has_non_empty_glyph(font, cp)
        ]
        random.shuffle(selected)

        if len(selected) >= style_glyph_count:
            return selected[:style_glyph_count]
        if selected:
            selected.extend(
                random.choices(selected, k=style_glyph_count - len(selected))
            )
        else:
            selected.extend([target_char] * style_glyph_count)
        return selected

    if len(selected) >= style_glyph_count:
        return selected[:style_glyph_count]

    remaining_count = style_glyph_count - len(selected)
    remaining_candidates = [cp for cp in available if cp not in selected]
    if len(remaining_candidates) >= remaining_count:
        selected.extend(random.sample(remaining_candidates, remaining_count))
    else:
        selected.extend(remaining_candidates)
        if selected:
            selected.extend(
                random.choices(selected, k=style_glyph_count - len(selected))
            )
        else:
            selected.extend([target_char] * (style_glyph_count - len(selected)))

    return selected


__all__ = [
    "_has_non_empty_glyph",
    "_font_has_codepoint",
    "_is_blank_rendering",
    "_sample_style_codepoints",
]
