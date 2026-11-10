"""Per-axis similarity combiner.

The flat-MLP baseline showed that concatenating heterogeneous metadata into one
vector and training a head to predict font identity *overfits* (own-cosine drops
when axes are combined).  This module is the fix: compute a **per-axis similarity
score** between every pair of fonts — numeric axes by standardized proximity,
categorical axes by match/mismatch with ``N/A`` neutral, tag axes by masked
cosine — and learn a small **combiner** over those per-axis scores.

Two combiner forms (``--combiner``):

* ``linear``  — ``score = sum_k w_k * s_k + b`` (unconstrained weights; the
  signed ``w_k`` is the interpretable axis-importance table).
* ``softmax`` — ``score = sum_k softmax(w)_k * s_k`` (non-negative convex
  combination; reads as "fraction of similarity signal per axis").

Both are trained with a circle loss over the named positives/negatives
(``two_channel_circle_loss`` semantics, but on a provided similarity matrix),
so positives are attracted, named negatives repelled, and unassigned pairs stay
neutral.  There are no per-font parameters, so "generalization" means the axis
weights transfer across held-out clusters — the leave-cluster-out protocol here
masks the held cluster's pairs out of the combiner training.

Usage::

    python -m hrothgar.similarity.axis_combiner --combiner linear
    python -m hrothgar.similarity.axis_combiner --combiner softmax
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from hrothgar.cedars import load_cedars
from hrothgar.fontquant import _to_float, quantify_row
from hrothgar.similarity.dataset import STYLE_CATEGORY_TAGS, THEME_TAGS
from hrothgar.similarity.metrics import masked_pairwise_cosine, pairwise_auc
from hrothgar.similarity.supervision import (
    JUDGEMENTS_PATH,
    SIMILARITIES_PATH,
    build_supervision,
    display_scores,
    text_half_supervision,
)
from hrothgar.utils import device

# ── Expressive tags (the subjective-response axis, from families.csv) ─────────
EXPRESSIVE_TAGS = [
    "/Expressive/Active",
    "/Expressive/Artistic",
    "/Expressive/Awkward",
    "/Expressive/Business",
    "/Expressive/Calm",
    "/Expressive/Childlike",
    "/Expressive/Competent",
    "/Expressive/Cute",
    "/Expressive/Excited",
    "/Expressive/Fancy",
    "/Expressive/Futuristic",
    "/Expressive/Happy",
    "/Expressive/Innovative",
    "/Expressive/Loud",
    "/Expressive/Playful",
    "/Expressive/Rugged",
    "/Expressive/Sincere",
    "/Expressive/Sophisticated",
    "/Expressive/Stiff",
    "/Expressive/Vintage",
]

# ── Remaining tag families (used by the display arm, which is tag-only) ──────
MONOSPACE_TAGS = ["/Monospace/Monospace"]
PURPOSE_TAGS = ["/Purpose/Easy Reading", "/Purpose/Learn To Write"]
QUALITY_TAGS = [
    "/Quality/Concept",
    "/Quality/Drawing",
    "/Quality/Spacing",
    "/Quality/Wordspace",
]
SEASONAL_TAGS = [
    "/Seasonal/Christmas",
    "/Seasonal/Diwali",
    "/Seasonal/Halloween",
    "/Seasonal/Hanukkah",
    "/Seasonal/Holi",
    "/Seasonal/Kwanzaa",
    "/Seasonal/Lunar New Year",
    "/Seasonal/Valentine's Day",
]
SPECIAL_USE_TAGS = [
    "/Special use/Barcode",
    "/Special use/Redaction",
    "/Special use/Symbols",
]

# ── Axis specification ────────────────────────────────────────────────────────
# Numeric axes: fontquant column names (scalar) or ("VECTOR", col) for the
# 26-vector proportions.  All values are standardized per-column over the
# universe; missing values contribute neither distance nor presence.
_NUMERIC_AXES: dict[str, list] = {
    "weight": ["appearance/weight", "appearance/weight_perceptual", "parametric/XOPQ", "opentype/os2_weight_class"],
    "width": ["appearance/width", "parametric/XTRA", "opentype/os2_width_class"],
    "slant": ["appearance/slant"],
    "contrast": [
        "stroke_contrast/medial",
        "stroke_contrast/antiqua",
        "stroke_contrast/raycaster",
        "stroke_contrast/medial_angle",
        "stroke_contrast/antiqua_angle",
    ],
    "parametric": [
        "parametric/XCLR", "parametric/XCLS", "parametric/XOFI", "parametric/XOLC",
        "parametric/XTFI", "parametric/XTLC", "parametric/YOFI", "parametric/YOLC",
        "parametric/YOPQ", "parametric/YTAS", "parametric/YTDE",
    ],
    "roughness": ["appearance/roughness"],
    "proportion-lower": [("VECTOR", "proportion/lowercase")],
    "proportion-upper": [("VECTOR", "proportion/uppercase")],
}

# Categorical axes: (source, column, vocabulary).  source "fq" = fontquant,
# "cedars" = cedars.csv.  A value outside the vocab, or a missing/N-A value, is
# the all-zero "not comparable" state.
_CATEGORICAL_AXES: list[tuple[str, str, str, list[str]]] = [
    ("a-storey", "fq", "appearance/lowercase_a_style", ["single_story", "double_story"]),
    ("a-shape", "fq", "structure/double_storey_a", ["flat", "angled", "convex", "concave"]),
    ("g-storey", "fq", "appearance/lowercase_g_style", ["single_story", "double_story"]),
    ("serif_shape", "fq", "structure/serif_shape", ["slab", "hairline", "wedge", "bracketed"]),
    ("stem_profile", "fq", "structure/stem_profile", ["straight", "flared", "tapered"]),
    ("stem_terminals", "fq", "structure/stem_terminals", ["blunted", "soft", "angled", "rounded"]),
    ("e-crossbar", "cedars", "e-crossbar", ["flat", "rounded", "angled"]),
    ("finials", "cedars", "finials", ["mixed", "flourished", "restrained"]),
    ("c-overhang", "cedars", "c-overhang", ["hook", "diagonal", "circular", "horizontal", "rounded", "vertical", "serif", "other"]),
    ("loops", "fq", "structure/loops", ["circular", "oval", "rectangular", "round triangular", "rounded rectangle", "superellipse", "triangular"]),
    ("waist", "fq", "structure/waist", ["high", "low", "middle"]),
    ("casing", "fq", "casing/lowercase_shapes", ["lowercase", "smallcaps", "uppercase"]),
]

# Boolean axes (fontquant columns, ``true``/``false``).  "both true" = +1
# (distinctive), "differ" = -1, "both false" = 0 (the uninformative majority),
# missing = 0.
_BOOLEAN_AXES: list[tuple[str, str]] = [
    ("monospaced", "appearance/monospaced"),
    ("stencil", "appearance/stencil"),
    ("serif_asymmetric", "structure/serif_asymmetric"),
    ("serif_bifurcated", "structure/serif_bifurcated"),
]

# Tag axes: masked cosine over these tag families.
_TAG_AXES: dict[str, list[str]] = {
    "tags-style": STYLE_CATEGORY_TAGS,
    "tags-theme": THEME_TAGS,
    "tags-expressive": EXPRESSIVE_TAGS,
}

# The display arm is tag-only: masked cosine over *every* tag family (all 74
# tags).  The style categories keep the shared "is categorized" mask (absent =
# genuine 0); the sparse families (theme/expressive/quality/purpose/seasonal/
# special/monospace) are masked per-tag (absent = "not assessed").
_DISPLAY_TAG_AXES: dict[str, list[str]] = {
    "tags-style": STYLE_CATEGORY_TAGS,
    "tags-theme": THEME_TAGS,
    "tags-expressive": EXPRESSIVE_TAGS,
    "tags-quality": QUALITY_TAGS,
    "tags-purpose": PURPOSE_TAGS,
    "tags-seasonal": SEASONAL_TAGS,
    "tags-special": SPECIAL_USE_TAGS,
    "tags-monospace": MONOSPACE_TAGS,
}


# ── Feature assembly ──────────────────────────────────────────────────────────
def _column_stats(raw: list[list[float]]) -> tuple[list[float], list[float]]:
    """Per-column mean and standard deviation over observed (non-NaN) values."""
    n = len(raw)
    d = len(raw[0]) if n else 0
    means, stds = [], []
    for j in range(d):
        vals = [raw[i][j] for i in range(n) if not math.isnan(raw[i][j])]
        mu = sum(vals) / len(vals) if vals else 0.0
        var = sum((v - mu) ** 2 for v in vals) / len(vals) if vals else 1.0
        means.append(mu)
        stds.append(math.sqrt(var))
    return means, stds


def _standardize_tensor(
    values: torch.Tensor, means: torch.Tensor, stds: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standardize columns from frozen ``means``/``stds``; return ``(values, presence)``.

    Missing entries (NaN) are imputed to 0 (the column mean after
    standardization) with ``presence`` 0; observed entries are clipped to
    ±5 standard deviations.  ``means``/``stds`` are passed in rather
    than fitted so a *single* new font can be placed in a standardization space
    fitted on the library (see :mod:`hrothgar.similarity.text_index`).
    """
    present = ~torch.isnan(values)
    sd = torch.where(stds > 1e-8, stds, torch.ones_like(stds))
    z = ((values - means) / sd).clamp(-5.0, 5.0)
    V = torch.where(present, z, torch.zeros_like(z))
    return V, present.float()


def _standardize_with_mask(raw: list[list[float]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Standardize per-column (NaN-tolerant); return ``(values, presence)``.

    ``values`` has missing entries imputed to 0 (the column mean after
    standardization); ``presence`` is 1.0 where the entry was observed.
    """
    n = len(raw)
    d = len(raw[0]) if n else 0
    if n == 0 or d == 0:
        return torch.zeros(n, d), torch.zeros(n, d)
    means, stds = _column_stats(raw)
    return _standardize_tensor(
        torch.tensor(raw, dtype=torch.float32),
        torch.tensor(means, dtype=torch.float32),
        torch.tensor(stds, dtype=torch.float32),
    )


def _parse_vector(value: str) -> list[float]:
    if not value:
        return [float("nan")] * 26
    parts = [p.strip() for p in value.split(",") if p.strip() != ""]
    out = [_to_float(p) if p else float("nan") for p in parts]
    while len(out) < 26:
        out.append(float("nan"))
    return out[:26]


def _expanded_columns(cols: list) -> list[str]:
    """Flatten an axis spec into per-dimension column names (26 for a vector)."""
    out: list[str] = []
    for col in cols:
        if isinstance(col, tuple) and col[0] == "VECTOR":
            out.extend(f"{col[1]}[{i}]" for i in range(26))
        else:
            out.append(col)
    return out


def numeric_rows(fonts, row_for=None) -> dict[str, list[list[float]]]:
    """Raw per-font values (``NaN`` = missing) for each numeric axis.

    ``row_for(font) -> {column: value}`` supplies the fontquant measurements;
    it defaults to :func:`hrothgar.fontquant.quantify_row`, which measures the
    font with the ``fontquant`` package (and memoizes it).  This is the
    *unstandardized* half of :func:`_numeric_raw`, split out so the values can be
    frozen in an index and standardized later (see
    :func:`hrothgar.similarity.text_index.build_index`).
    """
    row_for = row_for or quantify_row
    measured = [row_for(font) for font in fonts]

    def scalar(row, col):
        if not row:
            return float("nan")
        v = row.get(col, "")
        return _to_float(v) if v != "" else float("nan")

    # Derived proportions (ratios are more style-bearing than raw em units).
    def ratio(row, num_col, den_col):
        if not row:
            return float("nan")
        num = _to_float(row.get(num_col, ""))
        den = _to_float(row.get(den_col, ""))
        return (num / den) if (den and den != 0 and not math.isnan(num) and not math.isnan(den)) else float("nan")

    out: dict[str, list[list[float]]] = {}
    for name, cols in _NUMERIC_AXES.items():
        raw: list[list[float]] = []
        for row in measured:
            values: list[float] = []
            for col in cols:
                if isinstance(col, tuple) and col[0] == "VECTOR":
                    values.extend(_parse_vector(row.get(col[1], "") if row else ""))
                else:
                    values.append(scalar(row, col))
            raw.append(values)
        out[name] = raw

    # Proportions axis: ascender/cap_height/descender/x_height + two ratios.
    raw = []
    for row in measured:
        raw.append([
            scalar(row, "appearance/ascender"),
            scalar(row, "appearance/cap_height"),
            scalar(row, "appearance/descender"),
            scalar(row, "appearance/x_height"),
            ratio(row, "appearance/x_height", "appearance/ascender"),
            ratio(row, "appearance/cap_height", "appearance/ascender"),
        ])
    out["proportions"] = raw
    return out


def _numeric_raw(sup, row_for=None) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Per-font standardized values + presence masks for each numeric axis."""
    return {
        name: _standardize_with_mask(rows)
        for name, rows in numeric_rows(sup.fonts, row_for).items()
    }


def categorical_code_rows(fonts, cedars, row_for=None) -> dict[str, list[int]]:
    """Integer codes per categorical axis: vocab index, or -1 for missing/N-A."""
    row_for = row_for or quantify_row
    measured = [row_for(font) for font in fonts]
    out: dict[str, list[int]] = {}
    for name, source, col, vocab in _CATEGORICAL_AXES:
        codes: list[int] = []
        for i, font in enumerate(fonts):
            if source == "fq":
                row = measured[i]
                val = (row.get(col, "") if row else "").strip()
            else:  # cedars
                row = cedars.get(font.family.strip().lower())
                val = (row[col] if row else "").strip()
                val = "" if val in ("", "na") else val
            codes.append(vocab.index(val) if val in vocab else -1)
        out[name] = codes
    for name, col in _BOOLEAN_AXES:
        codes = []
        for row in measured:
            val = (row.get(col, "") if row else "").strip().lower()
            codes.append(1 if val == "true" else (0 if val == "false" else -1))
        out[name] = codes
    return out


def _categorical_codes(sup, cedars, row_for=None) -> dict[str, torch.Tensor]:
    """Tensor form of :func:`categorical_code_rows`."""
    return {
        name: torch.tensor(codes, dtype=torch.long)
        for name, codes in categorical_code_rows(sup.fonts, cedars, row_for).items()
    }


def tag_rows(
    fonts, tag_axes: dict[str, list[str]] | None = None
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Per-font tag vectors + presence masks for each tag axis."""
    tag_axes = _TAG_AXES if tag_axes is None else tag_axes
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, tags in tag_axes.items():
        V = torch.zeros(len(fonts), len(tags))
        M = torch.zeros(len(fonts), len(tags))
        for i, font in enumerate(fonts):
            ftags = font.tags()
            for j, tag in enumerate(tags):
                if tag in ftags:
                    V[i, j] = ftags[tag] / 100.0
                    M[i, j] = 1.0
        # Style-category tags are exhaustive/mutually-exclusive: once a font is
        # categorized, every style dim is a real value (absent = genuine 0).
        if name == "tags-style":
            has = M.sum(dim=1) > 0
            M = has.float().unsqueeze(1).expand_as(M)
        out[name] = (V, M)
    return out


def _tag_mats(sup, tag_axes: dict[str, list[str]] | None = None) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Per-font tag vectors + presence masks for each tag axis."""
    return tag_rows(sup.fonts, tag_axes)


# ── Per-axis similarity tensor ────────────────────────────────────────────────
def _numeric_sim(V: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """RMSE-based proximity ``1 - dist`` in [-1, 1]; incomparable -> 0."""
    n = V.shape[0]
    comp = P[:, None, :] * P[None, :, :]  # (N, N, d)
    delta = V[:, None, :] - V[None, :, :]  # (N, N, d)
    sq = (delta * delta) * comp
    cnt = comp.sum(dim=-1)  # (N, N)
    dist = torch.sqrt(sq.sum(dim=-1) / cnt.clamp(min=1))
    sim = (1.0 - dist).clamp(-1.0, 1.0)
    return sim.masked_fill(cnt == 0, 0.0)


def _categorical_sim(codes: torch.Tensor, boolean: bool) -> torch.Tensor:
    """Match/mismatch/neutral similarity from integer codes (-1 = missing)."""
    n = codes.shape[0]
    ci = codes[:, None]
    cj = codes[None, :]
    present = (ci >= 0) & (cj >= 0)
    same = (ci == cj) & present
    differ = present & (ci != cj)
    if boolean:
        both_true = (ci == 1) & (cj == 1)
        return both_true.float() - differ.float()
    return same.float() - differ.float()


# ── Single-anchor row similarity ──────────────────────────────────────────────
# These are the ``(1, N)`` counterparts of the pairwise functions above, used to
# score one query font against a frozen library without materializing ``N x N``.
# They must agree with the pairwise versions numerically; ``tests/test_text_index.py``
# checks that on random data.
def numeric_sim_row(
    value: torch.Tensor, presence: torch.Tensor, V: torch.Tensor, P: torch.Tensor
) -> torch.Tensor:
    """RMSE proximity of one font's values against an ``(N, d)`` library axis."""
    comp = presence.unsqueeze(0) * P  # (N, d)
    delta = value.unsqueeze(0) - V  # (N, d)
    sq = (delta * delta) * comp
    cnt = comp.sum(dim=-1)  # (N,)
    dist = torch.sqrt(sq.sum(dim=-1) / cnt.clamp(min=1))
    sim = (1.0 - dist).clamp(-1.0, 1.0)
    return sim.masked_fill(cnt == 0, 0.0)


def categorical_sim_row(code: int, codes: torch.Tensor, boolean: bool) -> torch.Tensor:
    """Match/mismatch/neutral similarity of one code against an ``(N,)`` axis."""
    present = (codes >= 0) & (code >= 0)
    same = (codes == code) & present
    differ = present & (codes != code)
    if boolean:
        both_true = codes.eq(1) if code == 1 else torch.zeros_like(codes, dtype=torch.bool)
        return both_true.float() - differ.float()
    return same.float() - differ.float()


def tag_cosine_row(
    value: torch.Tensor,
    mask: torch.Tensor,
    V: torch.Tensor,
    M: torch.Tensor,
) -> torch.Tensor:
    """Masked cosine of one tag vector against an ``(N, T)`` library axis."""
    overlap = mask.unsqueeze(0) * M  # (N, T)
    num = ((value * mask).unsqueeze(0) * (V * M)).sum(dim=-1)
    norm_a = (((value * value) * mask).unsqueeze(0) * M).sum(dim=-1)
    norm_l = ((V * V) * M * mask.unsqueeze(0)).sum(dim=-1)
    denom = torch.sqrt(norm_a * norm_l).clamp(min=1e-12)
    sim = num / denom
    return sim.masked_fill(overlap.sum(dim=-1) == 0, 0.0)



def _recogniser_models_dir() -> Path:
    """Directory the feature-recogniser checkpoints are loaded from.

    Resolution order:

    1. ``$HROTHGAR_MODELS_DIR`` (explicit override),
    2. the repository's ``models/`` (a checkout / editable install),
    3. ``hrothgar/features/checkpoints``, the copy shipped in the wheel.

    Step 3 is what lets a plain ``pip install hrothgar`` run the text-arm
    similarity check without a trained model directory.
    """
    override = os.environ.get("HROTHGAR_MODELS_DIR")
    if override:
        return Path(override)
    repo_models = Path(__file__).resolve().parents[3] / "models"
    if (repo_models / "a_storey.pth").exists():
        return repo_models
    return Path(__file__).resolve().parents[1] / "features" / "checkpoints"


def _load_recognisers() -> dict[str, object]:
    """Load the recogniser models still used by the combiner.

    Returns ``{"a_storey": ..., "e_crossbar": ..., "finials": ..., "c_overhang": ...}``.
    These cover ``a-storey``/``casing`` (from ``a_storey``), ``e-crossbar``,
    ``finials`` and ``c-overhang``; every other glyph-feature axis now comes from
    fontquant's geometry quantifiers.
    """
    from hrothgar.features import (
        AStoreyRecognizer,
        COverhangRecognizer,
        ECrossbarRecognizer,
        FinialsRecognizer,
    )

    models_dir = _recogniser_models_dir()

    def load(cls, stem):
        path = models_dir / f"{stem}.pth"
        if not path.exists():
            raise FileNotFoundError(
                f"feature recogniser checkpoint {path} not found; set "
                "HROTHGAR_MODELS_DIR to a directory containing a_storey.pth, "
                "e_crossbar.pth, finials.pth and c_overhang.pth"
            )
        return cls.from_checkpoint(path)

    return {
        "a_storey": load(AStoreyRecognizer, "a_storey"),
        "e_crossbar": load(ECrossbarRecognizer, "e_crossbar"),
        "finials": load(FinialsRecognizer, "finials"),
        "c_overhang": load(COverhangRecognizer, "c_overhang"),
    }


def recogniser_code_rows(fonts, recognisers: dict | None = None) -> dict[str, list[int]]:
    """Categorical codes for the recogniser-covered axes, from model verdicts.

    Each recogniser's verdict is mapped back onto the *same* vocabulary
    ``_CATEGORICAL_AXES`` uses, with ``-1`` for "unrecognisable"/"blank"/out-of-
    domain.  ``a_storey`` feeds both ``a-storey`` (single/double) and ``casing``
    (lowercase vs uppercase — small-caps is folded into uppercase, a known lossy
    approximation).  ``a-shape``/``serif``/``terminals`` are *not* covered here:
    they now come from fontquant's geometry-based quantifiers.
    """
    recognisers = recognisers or _load_recognisers()
    a_storey = recognisers["a_storey"]
    e_crossbar = recognisers["e_crossbar"]
    finials = recognisers["finials"]
    c_overhang = recognisers["c_overhang"]

    vocab = {name: v for name, _s, _c, v in _CATEGORICAL_AXES}
    covered = {"a-storey", "e-crossbar", "finials", "casing", "c-overhang"}
    out: dict[str, list[int]] = {name: [-1] * len(fonts) for name in covered}

    def code(axis: str, value: str) -> int:
        v = vocab[axis]
        return v.index(value) if value in v else -1

    for i, font in enumerate(fonts):
        # a_storey -> a-storey (single/double) + casing (lowercase/uppercase)
        a = a_storey.recognise_font(font)["verdict"]
        if a == "single":
            out["a-storey"][i] = code("a-storey", "single_story")
            out["casing"][i] = code("casing", "lowercase")
        elif a == "double":
            out["a-storey"][i] = code("a-storey", "double_story")
            out["casing"][i] = code("casing", "lowercase")
        elif a == "uppercase":
            out["casing"][i] = code("casing", "uppercase")

        out["e-crossbar"][i] = code("e-crossbar", e_crossbar.recognise_font(font)["verdict"])
        out["finials"][i] = code("finials", finials.recognise_font(font)["verdict"])
        out["c-overhang"][i] = code("c-overhang", c_overhang.recognise_font(font)["verdict"])

    return out


def _recogniser_codes(sup, recognisers: dict | None = None) -> dict[str, torch.Tensor]:
    """Tensor form of :func:`recogniser_code_rows`."""
    return {
        name: torch.tensor(codes, dtype=torch.long)
        for name, codes in recogniser_code_rows(sup.fonts, recognisers).items()
    }


def build_feature_bank(
    sup,
    feature_source: str = "hand",
    cedars: dict | None = None,
    stats_rows: list[int] | None = None,
    row_for=None,
) -> dict:
    """Per-font feature data for every axis, *without* pairwise computation.

    This is the freezable half of :func:`build_similarity_tensor`: a
    self-describing bank of the raw data each axis is built from, keyed so that
    :func:`hrothgar.similarity.text_index.TextIndex` can save it, reload it, and
    score a single query font against it.

    ``stats_rows`` optionally restricts which fonts the numeric standardization
    means/stds are fitted on (default: all).  The frozen index uses it to store
    positions for the *whole* library while standardizing in the text-arm
    universe's space — exactly the space a query font is standardized into.

    Fontquant measurements come from ``row_for(font)`` (default:
    :func:`hrothgar.fontquant.quantify_row`).

    Returns::

        {
          "families": [...],                       # (N,)
          "axis_names": [...],                     # (K,) scoring order
          "numeric":     {name: {"columns", "values", "means", "stds"}},
          "categorical": {name: {"vocab", "boolean", "codes"}},
          "tags":        {name: {"tags", "values", "mask"}},
        }

    ``values``/``means``/``stds`` are tensors: ``values`` is ``(N, d)`` raw
    fontquant quantities with ``NaN`` for missing, and the statistics are what a
    single new font is standardized with (so it lands in the same space as the
    library).
    """
    row_for = row_for or quantify_row
    if cedars is None:
        # cedars.csv is hand-labelled ground truth. Only the text/demo arms that
        # ask for ``feature_source="hand"`` need it; the recogniser path replaces
        # every cedars-backed axis with a model verdict, so a run without
        # cedars.csv still works.
        cedars = load_cedars() if feature_source != "recogniser" else {}

    fonts = sup.fonts
    raw_numeric = numeric_rows(fonts, row_for)
    codes = categorical_code_rows(fonts, cedars, row_for)
    if feature_source == "recogniser":
        codes.update(recogniser_code_rows(fonts))
    tag_mats = tag_rows(fonts)

    def fit_rows(rows: list[list[float]]) -> list[list[float]]:
        return rows if stats_rows is None else [rows[i] for i in stats_rows]

    numeric: dict[str, dict] = {}
    for name, cols in _NUMERIC_AXES.items():
        means, stds = _column_stats(fit_rows(raw_numeric[name]))
        numeric[name] = {
            "columns": _expanded_columns(cols),
            "values": torch.tensor(raw_numeric[name], dtype=torch.float32),
            "means": torch.tensor(means, dtype=torch.float32),
            "stds": torch.tensor(stds, dtype=torch.float32),
        }
    # Proportions axis: ascender/cap_height/descender/x_height + two ratios.
    means, stds = _column_stats(fit_rows(raw_numeric["proportions"]))
    numeric["proportions"] = {
        "columns": [
            "appearance/ascender",
            "appearance/cap_height",
            "appearance/descender",
            "appearance/x_height",
            "x_height/ascender",
            "cap_height/ascender",
        ],
        "values": torch.tensor(raw_numeric["proportions"], dtype=torch.float32),
        "means": torch.tensor(means, dtype=torch.float32),
        "stds": torch.tensor(stds, dtype=torch.float32),
    }

    categorical: dict[str, dict] = {}
    for name, _src, _col, vocab in _CATEGORICAL_AXES:
        categorical[name] = {
            "vocab": list(vocab),
            "boolean": False,
            "codes": torch.tensor(codes[name], dtype=torch.long),
        }
    for name, _col in _BOOLEAN_AXES:
        categorical[name] = {
            "vocab": ["false", "true"],
            "boolean": True,
            "codes": torch.tensor(codes[name], dtype=torch.long),
        }

    tags: dict[str, dict] = {}
    for name, (V, M) in tag_mats.items():
        # Materialize the style mask: ``tag_rows`` builds it with ``expand`` (a
        # stride-0 view), which is fine in memory but should not be frozen as-is.
        tags[name] = {
            "tags": list(_TAG_AXES[name]),
            "values": V.contiguous(),
            "mask": M.contiguous(),
        }

    return {
        "families": [f.family for f in fonts],
        "axis_names": axis_names(),
        "numeric": numeric,
        "categorical": categorical,
        "tags": tags,
    }


def axis_names() -> list[str]:
    """The scoring-axis order: numeric, categorical, boolean, then tag axes."""
    return (
        sorted(_NUMERIC_AXES)
        + ["proportions"]
        + [name for name, _src, _col, _vocab in _CATEGORICAL_AXES]
        + [name for name, _col in _BOOLEAN_AXES]
        + list(_TAG_AXES)
    )


def bank_similarity_tensor(bank: dict) -> tuple[torch.Tensor, list[str]]:
    """``(S, axis_names)`` with ``S`` of shape ``(N, N, K)`` from a feature bank."""
    names: list[str] = []
    mats: list[torch.Tensor] = []

    for name in bank["axis_names"]:
        if name in bank["numeric"]:
            ax = bank["numeric"][name]
            V, P = _standardize_tensor(ax["values"], ax["means"], ax["stds"])
            names.append(name)
            mats.append(_numeric_sim(V, P))
        elif name in bank["categorical"]:
            ax = bank["categorical"][name]
            names.append(name)
            mats.append(_categorical_sim(ax["codes"], boolean=ax["boolean"]))
        else:
            ax = bank["tags"][name]
            sim, _ = masked_pairwise_cosine(ax["values"], ax["mask"])
            names.append(name)
            mats.append(sim)

    S = torch.stack(mats, dim=-1)  # (N, N, K)
    return S, names


def build_similarity_tensor(sup, feature_source: str = "hand", row_for=None) -> tuple[torch.Tensor, list[str]]:
    """Return ``(S, axis_names)`` with ``S`` of shape ``(N, N, K)``.

    ``feature_source`` selects where the categorical glyph-feature axes come
    from: ``"hand"`` (cedars.csv + fontquant, the ground truth) or
    ``"recogniser"`` (the small feature-recogniser models).  The stem-terminal
    signal is split into two fontquant axes — ``stem_profile`` (stem shape) and
    ``stem_terminals`` (terminal cut/shape) — in both sources.

    ``row_for(font)`` supplies fontquant measurements (default: measure the font
    with the ``fontquant`` package), which lets a query TTF join the universe.
    """
    bank = build_feature_bank(sup, feature_source=feature_source, row_for=row_for)
    return bank_similarity_tensor(bank)


def build_display_tensor(sup) -> tuple[torch.Tensor, list[str]]:
    """Tag-only similarity tensor for the display arm: ``(S, names)``.

    Unlike :func:`build_similarity_tensor`, this feeds *no* fontquant/CEDARS/
    recogniser data — every axis is a masked cosine over a tag family (all 74
    tags, via :data:`_DISPLAY_TAG_AXES`).  The combiner then learns how much each
    tag family's agreement should contribute to "fits the vibe".
    """
    tag_mats = _tag_mats(sup, _DISPLAY_TAG_AXES)
    names: list[str] = []
    mats: list[torch.Tensor] = []
    for name in _DISPLAY_TAG_AXES:
        V, M = tag_mats[name]
        sim, _ = masked_pairwise_cosine(V, M)
        names.append(name)
        mats.append(sim)
    S = torch.stack(mats, dim=-1)  # (N, N, K)
    return S, names


def tag_axes_summary(font, tag_axes: dict[str, list[str]] | None = None) -> dict[str, str]:
    """Human-readable tag values for one font, grouped by tag family.

    Tag-only counterpart to :func:`font_feature_summary`, for the display arm's
    explain dump.  Returns ``{axis_name: "tag=centile, ..."}`` with only the
    assessed (non-zero) tags shown.
    """
    tag_axes = _DISPLAY_TAG_AXES if tag_axes is None else tag_axes
    ftags = font.tags()
    out: dict[str, str] = {}
    for name, tags in tag_axes.items():
        present = {t: v for t, v in ftags.items() if t in tags and v > 0}
        out[name] = ", ".join(f"{t}={v:.0f}" for t, v in present.items()) if present else "none"
    return out


# ── Debugging / explanation helpers ────────────────────────────────────────────
def _recogniser_verdicts(font, recognisers: dict | None) -> dict[str, str]:
    """Run the live recognisers on one font; return their verdicts.

    Mirrors the mapping :func:`_recogniser_codes` applies, so a debug dump shows
    exactly the label the model saw (``a_storey`` also drives ``casing``).  The
    ``a-storey``/``casing`` verdicts are translated onto the same vocabulary the
    categorical axes use (``single_story``/``double_story``; ``lowercase``/
    ``uppercase``), so a dump lines up with the axis codes directly.
    """
    recognisers = recognisers or _load_recognisers()
    a = recognisers["a_storey"].recognise_font(font)["verdict"]  # single/double/uppercase/blank/unrecognisable
    e = recognisers["e_crossbar"].recognise_font(font)["verdict"]  # flat/rounded/angled
    f = recognisers["finials"].recognise_font(font)["verdict"]  # mixed/flourished/restrained
    c = recognisers["c_overhang"].recognise_font(font)["verdict"]  # flat/blunted/ball/.../other
    a_storey = {"single": "single_story", "double": "double_story"}.get(a, a)
    casing = {"single": "lowercase", "double": "lowercase", "uppercase": "uppercase"}.get(a, a)
    return {"a-storey": a_storey, "casing": casing, "e-crossbar": e, "finials": f, "c-overhang": c}


def font_feature_summary(font, cedars, recognisers: dict | None = None) -> dict[str, str]:
    """Return a human-readable raw-feature dict for one font.

    Keys are the axis names the combiner consumes; values are compact strings
    showing the *raw* (unstandardized) inputs — fontquant columns, recogniser
    verdicts and tag centiles — so two fonts can be diffed by eye.
    """
    row = quantify_row(font.path)
    cedar = cedars.get(font.family.strip().lower(), {})
    verdicts = _recogniser_verdicts(font, recognisers)
    out: dict[str, str] = {}

    def raw(col: str) -> str:
        v = row.get(col, "").strip()
        return v if v != "" else "—"

    for name, cols in _NUMERIC_AXES.items():
        parts = []
        for col in cols:
            if isinstance(col, tuple) and col[0] == "VECTOR":
                vec = _parse_vector(row.get(col[1], ""))
                parts.append(f"{col[1]}=[{','.join(f'{x:.2f}' for x in vec if not math.isnan(x))}]")
            else:
                parts.append(f"{col.split('/')[-1]}={raw(col)}")
        out[name] = " ".join(parts)

    # proportions is derived from four scalar columns + two ratios.
    asc = _to_float(row.get("appearance/ascender", ""))
    cap = _to_float(row.get("appearance/cap_height", ""))
    dsc = _to_float(row.get("appearance/descender", ""))
    xh = _to_float(row.get("appearance/x_height", ""))
    def _r(n, d):
        return f"{n / d:.2f}" if (d and not math.isnan(n) and not math.isnan(d)) else "—"
    out["proportions"] = (
        f"asc={raw('appearance/ascender')} cap={raw('appearance/cap_height')} "
        f"desc={raw('appearance/descender')} x={raw('appearance/x_height')} "
        f"x/asc={_r(xh, asc)} cap/asc={_r(cap, asc)}"
    )

    for name, source, col, _vocab in _CATEGORICAL_AXES:
        if name in verdicts:
            v = verdicts[name]
            src = "recogniser"
        elif source == "fq":
            v = raw(col)
            src = "fq"
        else:
            v = (cedar.get(col, "") or "").strip()
            v = "—" if v in ("", "na") else v
            src = "cedars"
        out[name] = f"{v} ({src})"

    for name, col in _BOOLEAN_AXES:
        out[name] = raw(col)

    for name, tags in _TAG_AXES.items():
        ftags = font.tags()
        present = {t: v for t, v in ftags.items() if t in tags and v > 0}
        out[name] = ", ".join(f"{t}={v:.0f}" for t, v in present.items()) if present else "none"

    return out


def pair_breakdown(S: torch.Tensor, names: list[str], w: torch.Tensor, b: torch.Tensor, i: int, j: int) -> list[tuple[str, float, float, float]]:
    """Per-axis ``(name, similarity, weight, contribution)`` for pair ``(i, j)``.

    Sorted by descending ``|contribution|`` so the axes doing the most work come
    first; ``contribution = similarity * weight``, and ``sum(contrib) + b`` is the
    model's raw score for the pair.
    """
    s = S[i, j]
    contrib = s * w
    rows = [(names[k], float(s[k]), float(w[k]), float(contrib[k])) for k in range(len(names))]
    rows.sort(key=lambda r: -abs(r[3]))
    return rows


# ── Combiner + circle loss ────────────────────────────────────────────────────
def circle_loss_from_sim(
    sim: torch.Tensor,
    pos_weights: torch.Tensor,
    neg_weights: torch.Tensor,
    margin: float = 0.25,
    gamma: float = 128.0,
) -> torch.Tensor:
    """Circle loss on a provided ``(N, N)`` similarity matrix."""
    n = sim.shape[0]
    diag = torch.eye(n, dtype=torch.bool, device=sim.device)
    positives = (pos_weights > 0) & ~diag
    negatives = (neg_weights > 0) & ~diag
    negatives = negatives & ~positives

    delta_p = 1.0 - margin
    delta_n = margin
    alpha_p = torch.clamp_min(1.0 + margin - sim, min=0.0).detach()
    alpha_n = torch.clamp_min(sim + margin, min=0.0).detach()

    logit_p = -gamma * pos_weights * alpha_p * (sim - delta_p)
    logit_n = gamma * neg_weights * alpha_n * (sim - delta_n)
    logit_p = logit_p.masked_fill(~positives, float("-inf"))
    logit_n = logit_n.masked_fill(~negatives, float("-inf"))

    lp = torch.logsumexp(logit_p, dim=1)
    ln = torch.logsumexp(logit_n, dim=1)
    loss = F.softplus(lp + ln)

    has_pos = positives.any(dim=1)
    has_neg = negatives.any(dim=1)
    neg_only = has_neg & ~has_pos
    if neg_only.any():
        rows = neg_only
        s = sim[rows]
        m = negatives[rows].float()
        w = neg_weights[rows]
        hinge = (w * m * F.relu(s + margin)).sum(dim=1)
        denom = m.sum(dim=1).clamp(min=1)
        loss = loss.clone()
        loss[rows] = loss[rows] + hinge / denom

    active = has_pos | has_neg
    return loss[active].mean() if active.any() else torch.tensor(0.0, device=sim.device)


class SimilarityMLP(nn.Module):
    """Shallow MLP over the per-axis similarity vector ``s(a, b)``.

    Input is ``(N, N, K)``; the network is applied identically to every pair's
    ``K``-dim similarity vector and the output is passed through ``tanh`` to keep
    it in ``[-1, 1]`` (the same range the circle-loss margins assume).  Because
    the input is the (already-symmetric) similarity vector, the score is
    symmetric in ``(a, b)`` by construction.
    """

    def __init__(self, in_dim: int, hidden: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """``S`` is ``(N, N, K)``; returns ``(N, N)`` scores in ``[-1, 1]``."""
        n = S.shape[0]
        flat = S.reshape(-1, S.shape[-1])
        return torch.tanh(self.net(flat)).reshape(n, n)


class ResidualSimilarityMLP(nn.Module):
    """Linear base + a near-zero-initialized interaction correction.

    The linear head carries the robust per-axis signal; the small MLP correction
    starts at zero and only adds conjunctive structure where the supervised pairs
    support it, so it cannot blow away the linear solution on limited data.
    """

    def __init__(self, in_dim: int, hidden: int = 8, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)
        self.correction = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        with torch.no_grad():
            self.correction[-1].weight.zero_()
            self.correction[-1].bias.zero_()

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        n = S.shape[0]
        flat = S.reshape(-1, S.shape[-1])
        out = self.linear(flat) + self.correction(flat)
        return torch.tanh(out).reshape(n, n)


def _mlp_importance(S: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Mean absolute gradient of the score w.r.t. each axis (a sensitivity table)."""
    n = S.shape[0]
    K = S.shape[-1]
    S_flat = S.reshape(-1, K).detach().requires_grad_(True)
    out = model(S_flat.view(n, n, K))
    grad = torch.autograd.grad(out.sum(), S_flat)[0]  # (N*N, K)
    return grad.abs().mean(dim=0)  # (K,)


def _train_mlp_combiner(
    S: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    *,
    hidden: int,
    dropout: float,
    residual: bool,
    steps: int,
    lr: float,
    margin: float,
    gamma: float,
    seed: int,
    weight_decay: float,
) -> nn.Module:
    """Train an MLP combiner; returns the model (``score = model(S)``)."""
    K = S.shape[-1]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model_cls = ResidualSimilarityMLP if residual else SimilarityMLP
    model = model_cls(K, hidden=hidden, dropout=dropout).to(S.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        sim = model(S)
        loss = circle_loss_from_sim(sim, pos, neg, margin, gamma)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


def _pair_weight_matrices(sup, dev, exclude: set[int] | None = None):
    n = len(sup.fonts)
    pos = torch.zeros(n, n, device=dev)
    neg = torch.zeros(n, n, device=dev)
    for a, b, w in sup.pos_pairs:
        if exclude and (a in exclude or b in exclude):
            continue
        pos[a, b] = w
    for a, b, w in sup.neg_pairs:
        if exclude and (a in exclude or b in exclude):
            continue
        neg[a, b] = w
    eye = torch.eye(n, dtype=torch.bool, device=dev)
    pos = pos.masked_fill(eye, 0.0)
    neg = neg.masked_fill(eye, 0.0)
    return pos, neg


def _train_combiner(
    S: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    *,
    combiner: str,
    steps: int,
    lr: float,
    margin: float,
    gamma: float,
    seed: int,
    weight_decay: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train the combiner; returns ``(w, b)`` with ``score = S @ w + b``."""
    K = S.shape[1] if S.dim() == 2 else S.shape[2]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    w_raw = nn.Parameter(torch.zeros(K, device=S.device))
    b = nn.Parameter(torch.zeros(1, device=S.device))
    params = [w_raw, b]
    opt = torch.optim.Adam(
        [{"params": [w_raw], "weight_decay": weight_decay}, {"params": [b]}], lr=lr
    )
    for _ in range(steps):
        w = F.softmax(w_raw, dim=0) if combiner == "softmax" else w_raw
        sim = S @ w + b
        loss = circle_loss_from_sim(sim, pos, neg, margin, gamma)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        w = F.softmax(w_raw, dim=0) if combiner == "softmax" else w_raw
    return w.detach(), b.detach()


def _score_matrix(S: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return S @ w + b


def _cluster_triplet_from_sim(sim: torch.Tensor, cluster: dict) -> tuple[int, int, int, int]:
    c_total = c_sat = c_query = c_sep = 0
    for q in cluster["include"]:
        others = [o for o in cluster["include"] if o != q]
        if not others:
            continue
        c_query += 1
        if cluster["exclude"]:
            if min(sim[q, o].item() for o in others) > max(sim[q, e].item() for e in cluster["exclude"]):
                c_sep += 1
            for o in others:
                for e in cluster["exclude"]:
                    c_total += 1
                    if sim[q, o].item() > sim[q, e].item():
                        c_sat += 1
    return c_total, c_sat, c_query, c_sep


def _report_axes(importance: torch.Tensor, names: list[str], S: torch.Tensor, col: str = "importance") -> None:
    n = S.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=S.device)
    stds = [float(S[:, :, k][off].std()) for k in range(len(names))]
    contrib = [float(importance[k].item()) * stds[k] for k in range(len(names))]
    order = sorted(range(len(names)), key=lambda k: -contrib[k])
    print(f"\n  {'axis':<18} {col:>10} {'std(s)':>8} {'imp*std':>8}")
    for k in order:
        print(f"  {names[k]:<18} {importance[k].item():>10.4f} {stds[k]:>8.4f} {contrib[k]:>8.4f}")


# ── Runner ────────────────────────────────────────────────────────────────────
def _run(
    sup,
    *,
    combiner: str,
    hidden: int,
    dropout: float,
    steps: int,
    lr: float,
    margin: float,
    gamma: float,
    seed: int,
    weight_decay: float,
    feature_source: str,
    dev: torch.device,
) -> dict:
    S, names = build_similarity_tensor(sup, feature_source=feature_source)
    S = S.to(dev)
    n = len(sup.fonts)
    K = len(names)
    is_mlp = combiner in ("mlp", "residual")
    residual = combiner == "residual"

    # ── Global fit (axis weights + pairwise positives/negatives) ─────────────
    pos, neg = _pair_weight_matrices(sup, dev)
    if is_mlp:
        model = _train_mlp_combiner(S, pos, neg, hidden=hidden, dropout=dropout,
                                    residual=residual, steps=steps, lr=lr, margin=margin,
                                    gamma=gamma, seed=seed, weight_decay=weight_decay)
        sim = model(S)
        importance = _mlp_importance(S, model)
    else:
        w, b = _train_combiner(S, pos, neg, combiner=combiner, steps=steps, lr=lr,
                               margin=margin, gamma=gamma, seed=seed, weight_decay=weight_decay)
        sim = _score_matrix(S, w, b)
        importance = w
    pos_scores = torch.stack([sim[a, b] for a, b, _w in sup.pos_pairs])
    neg_scores = torch.stack([sim[a, b] for a, b, _w in sup.neg_pairs])
    pairwise = pairwise_auc(pos_scores, neg_scores)

    # ── Leave-cluster-out triplet / separation ───────────────────────────────
    total = satisfied = n_query = n_sep = 0
    per_cluster: list[tuple] = []
    for ci, cluster in enumerate(sup.clusters):
        held = set(cluster["include"]) | set(cluster["exclude"])
        if not cluster["include"]:
            continue
        pos_c, neg_c = _pair_weight_matrices(sup, dev, exclude=held)
        if is_mlp:
            model_c = _train_mlp_combiner(S, pos_c, neg_c, hidden=hidden, dropout=dropout,
                                          residual=residual, steps=steps, lr=lr, margin=margin,
                                          gamma=gamma, seed=seed, weight_decay=weight_decay)
            sim_c = model_c(S)
        else:
            w_c, b_c = _train_combiner(S, pos_c, neg_c, combiner=combiner, steps=steps, lr=lr,
                                       margin=margin, gamma=gamma, seed=seed, weight_decay=weight_decay)
            sim_c = _score_matrix(S, w_c, b_c)
        c_total, c_sat, c_query, c_sep = _cluster_triplet_from_sim(sim_c, cluster)
        total += c_total
        satisfied += c_sat
        n_query += c_query
        n_sep += c_sep
        per_cluster.append((ci, (c_sat / c_total) if c_total else None, (c_sep / c_query) if c_query else None))

    return {
        "combiner": combiner,
        "n_axes": K,
        "triplet_accuracy": (satisfied / total) if total else None,
        "separation": (n_sep / n_query) if n_query else None,
        "pairwise_auc": pairwise["auc"],
        "pairwise_pos_mean": pairwise["pos_mean"],
        "pairwise_neg_mean": pairwise["neg_mean"],
        "num_triplets": total,
        "num_clusters": len(sup.clusters),
        "per_cluster": per_cluster,
        "importance": importance.cpu(),
        "names": names,
        "S": S.cpu(),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts"))
    p.add_argument("--list", default=SIMILARITIES_PATH)
    p.add_argument("--judgements", default=JUDGEMENTS_PATH)
    p.add_argument("--combiner", choices=("linear", "softmax", "mlp", "residual"), default="linear")
    p.add_argument("--hidden", type=int, default=32, help="MLP hidden width (--combiner mlp)")
    p.add_argument("--dropout", type=float, default=0.1, help="MLP dropout (--combiner mlp)")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--margin", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=128.0)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--text-half",
        action="store_true",
        help="keep only the bottom half of the universe by display score (the text arm)",
    )
    p.add_argument(
        "--feature-source",
        choices=("hand", "recogniser"),
        default="hand",
        help="glyph-feature axes from hand tags (cedars.csv) or the feature recognisers",
    )
    p.add_argument(
        "--universe",
        choices=("supervised", "full"),
        default="supervised",
        help="supervised families only, or every non-skipped family in the library",
    )
    args = p.parse_args()

    dev = device(args.device)
    sup = build_supervision(args.repo, args.list, args.judgements, universe=args.universe)
    if args.text_half:
        scores = display_scores(sup)
        sup = text_half_supervision(sup)
        print(
            f"text half: kept {len(sup.fonts)} fonts "
            f"(display score <= {scores.quantile(0.5):.1f}); "
            f"clusters={len(sup.clusters)} pos_pairs={len(sup.pos_pairs)} neg_pairs={len(sup.neg_pairs)}",
            flush=True,
        )
    print(
        f"universe={len(sup.fonts)} clusters={len(sup.clusters)} "
        f"pos_pairs={len(sup.pos_pairs)} neg_pairs={len(sup.neg_pairs)} "
        f"combiner={args.combiner} feature_source={args.feature_source}",
        flush=True,
    )

    r = _run(
        sup, combiner=args.combiner, hidden=args.hidden, dropout=args.dropout,
        steps=args.steps, lr=args.lr, margin=args.margin, gamma=args.gamma,
        seed=args.seed, weight_decay=args.weight_decay, feature_source=args.feature_source,
        dev=dev,
    )

    ta = f"{r['triplet_accuracy']:.3f}" if r["triplet_accuracy"] is not None else "n/a"
    sp = f"{r['separation']:.3f}" if r["separation"] is not None else "n/a"
    auc = f"{r['pairwise_auc']:.3f}"
    print(f"\n[{r['combiner']} combiner] axes={r['n_axes']} triplets={r['num_triplets']}")
    print(f"  triplet accuracy = {ta}")
    print(f"  separation rate  = {sp}")
    print(f"  pairwise AUC     = {auc}  (pos_mean={r['pairwise_pos_mean']:.3f} neg_mean={r['pairwise_neg_mean']:.3f})")
    for ci, ta_c, sp_c in r["per_cluster"]:
        ta_s = f"{ta_c:.3f}" if ta_c is not None else "  -"
        sp_s = f"{sp_c:.3f}" if sp_c is not None else " -"
        print(f"  cluster {ci}: triplet={ta_s} sep={sp_s}")
    col = "sensitivity" if args.combiner in ("mlp", "residual") else "weight"
    _report_axes(r["importance"], r["names"], r["S"], col=col)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
