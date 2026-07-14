"""Dataset constants — no heavy dependencies (no torch, sklearn).

These are separated from ``hrothgar.dataset`` so that inference-only
code (Core ML, Glyphs.app plugin) can import them without pulling in
the full ML stack.
"""

from glyphsets import GlyphSet, unicodes_per_glyphset

LATIN_CORE = [x for x in GlyphSet("GF_Latin_Core").get_characters() if x != 32]
LATIN_CORE = [x for x in LATIN_CORE if not (0x0300 <= x <= 0x036F)]
LATIN_CORE.append(0x20B9)  # Rupee

LATIN_KERNEL = [x for x in unicodes_per_glyphset("GF_Latin_Kernel") if x != 32]
LATIN_KERNEL = [x for x in LATIN_KERNEL if not (0x0300 <= x <= 0x036F)]

LGC_ALL = set(
    [x for x in unicodes_per_glyphset("GF_Latin_Core") if x != 32]
    + [x for x in unicodes_per_glyphset("GF_Latin_Plus") if x != 32]
    + [x for x in unicodes_per_glyphset("GF_Latin_African") if x != 32]
    + [x for x in unicodes_per_glyphset("GF_Cyrillic_Core") if x != 32]
    + [x for x in unicodes_per_glyphset("GF_Greek_Core") if x != 32]
)
LGC_ALL = [x for x in LGC_ALL if not (0x0300 <= x <= 0x036F)]

CAPS_ONLY = [ord(x) for x in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789$₹"]

__all__ = ["LATIN_CORE", "LATIN_KERNEL", "LGC_ALL", "CAPS_ONLY"]
