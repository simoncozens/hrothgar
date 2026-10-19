"""Content-aware exemplar selection for the exemplar-conditional diffusion model.

When the denoiser is conditioned on a font's *evidence glyphs* (the style
references it cross-attends to), which glyphs those are matters.  A lowercase
``a`` carries almost no information about how a font draws an uppercase ``Q`` —
the uppercase and lowercase skeletons can be stylistically near-orthogonal.  So
the evidence set should be *mostly* the same broad character class as the target
codepoint, with a controlled admixture of the other classes.

This module is the single place that encodes that policy.  The core is
:func:`exemplar_mix`, which maps a target codepoint to a probability
distribution over four glyph categories — ``UPPER``, ``LOWER``, ``DIGIT``,
``SYMBOL`` — and :func:`draw_exemplars`, which turns that distribution (plus a
font's available codepoints) into an actual list of ``k`` exemplar codepoints.

The defaults encode the intuition:

* uppercase target  → mostly uppercase, some lowercase, a little digit/symbol;
* lowercase target  → mostly lowercase, some uppercase, a little digit/symbol;
* digit target      → mostly digits, some upper/lowercase;
* symbol target     → a broad mix of everything (symbols are the least
  self-similar class, so they lean on the rest of the font).

Tune either by editing :data:`DEFAULT_MIX`, by passing a custom matrix to
``mix=``, or by sweeping the single ``same_frac`` knob via :func:`scaled_mix`.
"""

from __future__ import annotations

import random
import unicodedata
from collections import defaultdict

UPPER = "upper"
LOWER = "lower"
DIGIT = "digit"
SYMBOL = "symbol"
CATEGORIES = (UPPER, LOWER, DIGIT, SYMBOL)

# target category -> {source category: weight}.  Rows are normalised at use, so
# only relative proportions matter.
DEFAULT_MIX: dict[str, dict[str, float]] = {
    UPPER: {UPPER: 0.60, LOWER: 0.30, DIGIT: 0.05, SYMBOL: 0.05},
    LOWER: {UPPER: 0.30, LOWER: 0.60, DIGIT: 0.05, SYMBOL: 0.05},
    DIGIT: {UPPER: 0.20, LOWER: 0.20, DIGIT: 0.50, SYMBOL: 0.10},
    SYMBOL: {UPPER: 0.25, LOWER: 0.25, DIGIT: 0.20, SYMBOL: 0.30},
}


def glyph_category(codepoint: int) -> str:
    """Classify a codepoint into one of :data:`CATEGORIES`.

    Letters are split on case (``Lu``/``Lt`` → UPPER, ``Ll`` → LOWER), decimal
    digits (``Nd``) are DIGIT, and everything else (symbols, punctuation,
    spacing) is SYMBOL.
    """
    try:
        cat = unicodedata.category(chr(codepoint))
    except (ValueError, OverflowError):
        return SYMBOL
    if cat in ("Lu", "Lt"):
        return UPPER
    if cat == "Ll":
        return LOWER
    if cat == "Nd":
        return DIGIT
    return SYMBOL


def scaled_mix(same_frac: float, base: dict = DEFAULT_MIX) -> dict:
    """Build a mix matrix whose per-row same-category weight is ``same_frac``.

    Cross-category weights are rescaled proportionally to ``base``, so a single
    knob controls "how self-similar should the evidence be" without losing the
    relative cross-category preferences baked into ``base``.
    """
    out: dict[str, dict[str, float]] = {}
    for target, row in base.items():
        diag = row.get(target, 0.0)
        cross_total = sum(w for c, w in row.items() if c != target)
        scale = (1.0 - same_frac) / cross_total if cross_total > 0 else 0.0
        new = {c: (same_frac if c == target else w * scale) for c, w in row.items()}
        out[target] = new
    return out


def exemplar_mix(
    codepoint: int, mix: dict | None = None
) -> dict[str, float]:
    """Return the normalised ``{category: ratio}`` evidence mix for ``codepoint``.

    ``mix`` is a ``target -> {source: weight}`` matrix; defaults to
    :data:`DEFAULT_MIX`.
    """
    mix = mix if mix is not None else DEFAULT_MIX
    target = glyph_category(codepoint)
    row = mix.get(target)
    if not row:
        # Unknown target category: fall back to a uniform mix.
        return {c: 1.0 / len(CATEGORIES) for c in CATEGORIES}
    total = sum(row.values())
    if total <= 0:
        return {c: 1.0 / len(CATEGORIES) for c in CATEGORIES}
    return {c: row.get(c, 0.0) / total for c in CATEGORIES}


def draw_exemplars(
    target: int,
    pool,
    k: int,
    mix: dict | None = None,
    rng: random.Random | None = None,
    exclude_target: bool = True,
) -> list[int]:
    """Draw ``k`` exemplar codepoints for ``target`` from ``pool``.

    ``pool`` is the set of codepoints the font can draw.  The target itself is
    excluded by default (an exemplar set must not leak the answer for the
    "incomplete font" task).  Counts are allocated across categories by the
    largest-remainder method from :func:`exemplar_mix`, then sampled within each
    category *without replacement* so a single category never contributes the
    same glyph twice (only a category that is genuinely under-supplied relative
    to its allocation is topped up with replacement).  The result is shuffled so
    the mix is interleaved rather than grouped by class.

    ``rng`` may be a ``random.Random`` instance (so callers can seed it); by
    default the module-level ``random`` is used.
    """
    rng = rng if rng is not None else random
    available = [cp for cp in pool if cp != target] if exclude_target else list(pool)
    if not available or k <= 0:
        return []

    ratios = exemplar_mix(target, mix)
    by_cat: dict[str, list[int]] = defaultdict(list)
    for cp in available:
        by_cat[glyph_category(cp)].append(cp)

    # Only categories that actually have glyphs participate.
    cats = [c for c in CATEGORIES if by_cat.get(c)]
    weights = [ratios.get(c, 0.0) for c in cats]
    wsum = sum(weights)
    if wsum <= 0:
        weights = [1.0] * len(cats)
        wsum = float(len(cats))

    # Largest-remainder allocation so the per-category counts sum exactly to k.
    raw = [w / wsum * k for w in weights]
    counts = [int(r) for r in raw]
    rem = k - sum(counts)
    order = sorted(range(len(cats)), key=lambda i: raw[i] - int(raw[i]), reverse=True)
    for i in range(rem):
        counts[order[i % len(order)]] += 1

    out: list[int] = []
    for i, cat in enumerate(cats):
        if counts[i] <= 0:
            continue
        cat_pool = by_cat[cat]
        if counts[i] <= len(cat_pool):
            out.extend(rng.sample(cat_pool, k=counts[i]))
        else:
            # Category under-supplied relative to its allocation: take every
            # glyph once, then top up with replacement.  Duplicates are only
            # possible here, not when the category has enough distinct glyphs.
            out.extend(cat_pool)
            out.extend(rng.choices(cat_pool, k=counts[i] - len(cat_pool)))
    rng.shuffle(out)
    return out
