"""Dataset constants — no heavy dependencies (no torch, sklearn).

These are separated from ``hrothgar.dataset`` so that inference-only
code (Core ML, Glyphs.app plugin) can import them without pulling in
the full ML stack.
"""

from collections.abc import Iterable

from glyphsets import GlyphSet, unicodes_per_glyphset


def _without_combining(codepoints: Iterable[int]) -> list[int]:
    return [x for x in codepoints if not (0x0300 <= x <= 0x036F)]


def _glyphset_codepoints(glyphset_name: str) -> list[int]:
    """Codepoints in a named glyphset, minus space.

    ``unicodes_per_glyphset`` returns ``None`` for an unknown glyphset name
    rather than raising, so make that failure explicit here.
    """
    codepoints = unicodes_per_glyphset(glyphset_name)
    if codepoints is None:
        raise ValueError(f"Unknown glyphset: {glyphset_name}")
    return [x for x in codepoints if x != 32]


LATIN_CORE = [
    x
    for x in GlyphSet("GF_Latin_Core").get_characters()
    if x != 32 and not (0x0300 <= x <= 0x036F)
]
LATIN_CORE.append(0x20B9)  # Rupee

LATIN_KERNEL = _without_combining(_glyphset_codepoints("GF_Latin_Kernel"))

LGC_ALL = _without_combining(
    set(
        _glyphset_codepoints("GF_Latin_Core")
        + _glyphset_codepoints("GF_Latin_Plus")
        + _glyphset_codepoints("GF_Latin_African")
        + _glyphset_codepoints("GF_Cyrillic_Core")
        + _glyphset_codepoints("GF_Greek_Core")
    )
)

CAPS_ONLY = [ord(x) for x in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789$₹"]

__all__ = ["CAPS_ONLY", "LATIN_CORE", "LATIN_KERNEL", "LGC_ALL"]
