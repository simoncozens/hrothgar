"""Codepoint constants for the content detector (importable without torch).

The content detector learns one *anchor* (a "slot") per codepoint and must
separate that codepoint from everything that looks like it.  Two curated sets
drive the pilot:

* :data:`PILOT_ANCHORS` — the codepoints that get a learned slot;
* :data:`PILOT_NEGATIVES` — the broad pool random negatives are drawn from
  (diacritic variants and punctuation included).

:data:`CONFUSABLES` names, per anchor, the glyphs a careless reader could
mistake it for: its case sibling, Latin near-neighbours and digits.  Diacritic
variants (``é`` for ``e``, ``à`` for ``á``) are derived programmatically from the
pool rather than listed here.  These curated glyphs are the *hard* negatives —
the whole point of the detector is to get them right.

Cross-script homoglyphs (Cyrillic ``а е о с ѕ``, Greek ``α ε ο ρ``) are **not** in
the negative pool.  Fonts build those codepoints from the Latin outlines, so in
many fonts they render *bit-identically* to the Latin anchor (mean pixel
difference 0.0, verified across Roboto / Playfair / Source Sans 3) while carrying
a different codepoint.  They are therefore not hard negatives but *impossible*
ones — the same picture labelled both positive and negative — and training on
them damages the embedding and caps ``recall_at_far``.  They are retained in
:data:`OFF_SCRIPT_CODEPOINTS` only as documentation/future off-sample use.

The codepoints are in em-relative *content* space, not display space: a glyph
with a diacritic is a different codepoint from its base, and an uppercase glyph
is a different codepoint from its lowercase sibling.
"""

from __future__ import annotations

import unicodedata

# The anchor set is grown in tiers, so the effect of each tier can be watched in
# isolation.  Tier 1 is the lowercase pilot; tier 2 adds the uppercase of the same
# letters, which turns on the *case* axis — a case pair differs mainly by height
# and is the hardest content distinction in the set.
PILOT_LOWERCASE: list[int] = [ord(c) for c in "aegofcs"]
PILOT_UPPERCASE: list[int] = [ord(c) for c in "AEGOFCS"]
PILOT_ACCENTED: list[int] = [
    0xE1,  # á
    0xE0,  # à
]
PILOT_ANCHORS: list[int] = PILOT_LOWERCASE + PILOT_ACCENTED + PILOT_UPPERCASE


def nfd_base_codepoint(codepoint: int) -> int:
    """The base letter of a precomposed codepoint (``é`` -> ``e``).

    Cyrillic/Greek letters are not canonically decomposable, so their base is
    themselves — which is what we want: ``а`` (Cyrillic) is *not* a diacritic
    variant of ``a`` (Latin) but a homoglyph, excluded from the negative pool in
    any case (see the module docstring).
    """
    decomposed = unicodedata.normalize("NFD", chr(codepoint))
    return ord(decomposed[0]) if decomposed else codepoint


# Per-anchor hard negatives.  Latin only (see the module docstring): a
# cross-script homoglyph is frequently the *same outline* as the anchor, which
# makes it an impossible negative rather than a hard one.  Order is irrelevant
# (they are sampled), but keep the strongest confusions first for readability.
# A case sibling is listed explicitly even where it is itself an anchor: anchors
# are legitimately each other's hard negatives, which is what teaches case.
CONFUSABLES: dict[int, list[int]] = {
    # Lowercase tier.
    ord("a"): [ord("A"), 0x0251, ord("d"), ord("q"), ord("o")],
    ord("e"): [ord("E"), ord("c"), ord("o"), ord("s")],
    ord("g"): [ord("G"), ord("q"), ord("y"), ord("j"), ord("9")],
    ord("o"): [ord("O"), ord("0"), ord("c"), ord("e")],
    ord("f"): [ord("F"), ord("t"), ord("l"), ord("i")],
    ord("c"): [ord("C"), ord("e"), ord("o"), ord("s")],
    ord("s"): [ord("S"), ord("z"), ord("c"), ord("e"), ord("5")],
    # Accented lowercase: the case sibling is the other accented form.
    0xE1: [0xC1],  # á -> Á
    0xE0: [0xC0],  # à -> À
    # Uppercase tier: the case sibling first, then the classic capital confusions.
    ord("A"): [ord("a"), ord("4")],
    ord("E"): [ord("e"), ord("F")],
    ord("G"): [ord("g"), ord("C"), ord("6")],
    ord("O"): [ord("o"), ord("0"), ord("Q"), ord("D")],
    ord("F"): [ord("f"), ord("E"), ord("P")],
    ord("C"): [ord("c"), ord("G"), ord("O")],
    ord("S"): [ord("s"), ord("5"), ord("$"), ord("Z")],
}

# Cross-script homoglyphs, deliberately *not* used as negatives (see the module
# docstring).  Kept here so the exclusion is visible and so they are available
# for a future script-rejection experiment.
OFF_SCRIPT_CODEPOINTS: list[int] = sorted(
    [0x0430, 0x0435, 0x043E, 0x0441, 0x0440, 0x0445, 0x0455, 0x0456]  # Cyrillic
    + [0x03B1, 0x03B5, 0x03BF, 0x03C1]  # Greek
)

# The broad pool random negatives come from: all Latin letters (both cases),
# digits, common precomposed diacritics and a few punctuation marks.  Anchors are
# not excluded here — a glyph of ``a`` is a perfectly good negative for the ``e``
# anchor.  Cross-script homoglyphs are excluded (see the module docstring).
PILOT_NEGATIVES: list[int] = sorted(
    set(
        [ord(c) for c in "abcdefghijklmnopqrstuvwxyz"]
        + [ord(c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
        + [ord(c) for c in "0123456789"]
        # precomposed lowercase Latin diacritics (à á â ä ç è é ê ë ì ï ñ ò ó ö ù ü)
        + [
            0xE0,
            0xE1,
            0xE2,
            0xE4,
            0xE7,
            0xE8,
            0xE9,
            0xEA,
            0xEB,
            0xEC,
            0xEF,
            0xF1,
            0xF2,
            0xF3,
            0xF6,
            0xF9,
            0xFC,
        ]
        # precomposed uppercase Latin diacritics (À Á Â Ä Ç È É Ê Ë Ì Ï Ñ Ò Ó Ö Ù Ü),
        # so the uppercase anchors get their diacritic variants too
        + [
            0xC0,
            0xC1,
            0xC2,
            0xC4,
            0xC7,
            0xC8,
            0xC9,
            0xCA,
            0xCB,
            0xCC,
            0xCF,
            0xD1,
            0xD2,
            0xD3,
            0xD6,
            0xD9,
            0xDC,
        ]
        + [0x0251]  # Latin small letter alpha
        + [ord("."), ord(","), ord("'"), ord("-"), ord("!")]
    )
)

# Families held out of the dataset entirely (training *and* the eval panel).
# Laila encodes ``á`` and ``à`` the wrong way round, so as training data it is a
# wrong *label* — it would teach the á/à anchors each other's glyph.  It is kept
# aside deliberately as the practical regression test for the detector.
EXCLUDED_FAMILIES: frozenset[str] = frozenset({"Laila"})

__all__ = [
    "CONFUSABLES",
    "EXCLUDED_FAMILIES",
    "OFF_SCRIPT_CODEPOINTS",
    "PILOT_ACCENTED",
    "PILOT_ANCHORS",
    "PILOT_LOWERCASE",
    "PILOT_NEGATIVES",
    "PILOT_UPPERCASE",
    "nfd_base_codepoint",
]
