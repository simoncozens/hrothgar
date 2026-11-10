"""Dataset assembly for the glyph-feature recognisers.

A *task* is a feature to recognise (``e-crossbar``: flat/angled/rounded, …) and
the glyph prompt that shows it.  This module turns a Google Fonts checkout into
the ``(X, y)`` tensors a recogniser trains on:

* :func:`render_prompts` — a font's prompt glyphs as one ``(G, 1, H, W)`` image
  batch, rendered into the font's own tight frame (and optionally cropped to the
  region the feature lives in);
* :func:`build_task` — labelled examples from the hand labels (``cedars.csv``)
  or from fontquant's measurements, grouped by family so a held-out split can
  keep a family on one side;
* :func:`build_weird` — out-of-domain examples (symbol/barcode/redaction fonts)
  for the recognisers' "unrecognisable" class.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn.functional as F
import uharfbuzz as hb

from hrothgar.fontquant import quantify_row
from hrothgar.googlefonts import GoogleFont, GoogleFonts, load_family_tags
from hrothgar.render import frame_for_codepoints, render

_BLANK = 0.995

# ── Image helpers ─────────────────────────────────────────────────────────────
def crop_bottom(img: torch.Tensor, frac: float) -> torch.Tensor:
    """Crop the bottom ``frac`` of a ``(H, W)`` glyph image, resize back to ``(H, W)``.

    Used to spend the canvas on a local feature that lives at the baseline (e.g.
    stem terminals) rather than the full glyph silhouette.
    """
    if frac <= 0.0 or frac >= 1.0:
        return img
    h = img.shape[-2]
    keep = img[..., int((1.0 - frac) * h):, :]  # (keep_h, W)
    keep = keep.unsqueeze(0).unsqueeze(0)  # (1, 1, keep_h, W)
    return F.interpolate(keep, size=(h, img.shape[-1]), mode="bilinear", align_corners=False)[0, 0]


def crop_middle(img: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Crop the vertical band ``[lo, hi]`` (fractions of image height), resize back.

    Used to spend the canvas on a feature that sits mid-glyph (e.g. the
    bowl/stem junction of a double-storey ``a``) rather than the whole silhouette.
    """
    h = img.shape[-2]
    keep = img[..., int(lo * h):int(hi * h), :]  # (keep_h, W)
    keep = keep.unsqueeze(0).unsqueeze(0)  # (1, 1, keep_h, W)
    return F.interpolate(keep, size=(h, img.shape[-1]), mode="bilinear", align_corners=False)[0, 0]


def crop_right(img: torch.Tensor, frac: float) -> torch.Tensor:
    """Crop the right ``frac`` of a ``(H, W)`` glyph image, resize back to ``(H, W)``.

    Used to spend the canvas on a local feature that lives at the right edge (e.g.
    the c-overhang terminal) rather than the full glyph silhouette.
    """
    if frac <= 0.0 or frac >= 1.0:
        return img
    w = img.shape[-1]
    keep = img[..., :, int((1.0 - frac) * w):]  # (H, keep_w)
    keep = keep.unsqueeze(0).unsqueeze(0)  # (1, 1, H, keep_w)
    return F.interpolate(keep, size=(img.shape[-2], img.shape[-1]), mode="bilinear", align_corners=False)[0, 0]


def is_blank(img: torch.Tensor) -> bool:
    """Whether any prompt glyph rendered as empty paper (ink on white: ~1.0)."""
    return bool((img.amin() > _BLANK).any())


# ── Rendering ────────────────────────────────────────────────────────────────
def render_prompts(
    font,
    glyph_str: str,
    size: int,
    crop_bottom_frac: float = 0.0,
    crop_middle=None,
    crop_right_frac: float = 0.0,
    axis_position=None,
) -> torch.Tensor | None:
    """Render the prompt glyphs into a shared tight frame; ``(G, 1, H, W)``.

    Returns ``None`` if any prompt glyph is missing from the font's cmap.
    """
    codepoints = [ord(c) for c in glyph_str]
    hb_font = hb.Font(font.hb_face)
    if any(hb_font.get_nominal_glyph(cp) is None for cp in codepoints):
        return None
    frame = frame_for_codepoints(font, codepoints)
    imgs = [
        render(font, codepoint=cp, size=size, frame=frame, axis_position=axis_position)[0]
        for cp in codepoints
    ]
    if crop_bottom_frac > 0.0:
        imgs = [crop_bottom(img, crop_bottom_frac) for img in imgs]
    if crop_middle is not None:
        imgs = [crop_middle(img, *crop_middle) for img in imgs]
    if crop_right_frac > 0.0:
        imgs = [crop_right(img, crop_right_frac) for img in imgs]
    return torch.stack(imgs).unsqueeze(1)  # (G, 1, H, W)


def axis_positions(font) -> list:
    """Axis positions to render for a font: default + ``wght``/``wdth`` extremes.

    Static fonts yield ``[None]`` (the default instance).  Variable fonts yield
    the default plus each ``wght``/``wdth`` extreme with the other axes held at
    their default — bolder/narrower/wider views of the *same* design, whose label
    transfers unchanged (the design-space topology is preserved).
    """
    from hrothgar.dataset import fvar_axes

    axes = fvar_axes(font.path)
    if not axes:
        return [None]
    positions: list = [None]  # default instance
    for tag in ("wght", "wdth"):
        for i, a in enumerate(axes):
            if a[0] != tag:
                continue
            lo, _default, hi = a[1], a[2], a[3]
            for val in (lo, hi):
                pos = [ax[2] for ax in axes]  # all axes at default
                pos[i] = val
                positions.append(tuple(pos))
    return positions


# ── Task assembly ────────────────────────────────────────────────────────────
def build_task(
    task: dict,
    family_to_font: dict[str, object],
    cedars: dict[str, dict[str, str]],
    size: int,
    seed: int = 0,
    crop_bottom_frac: float = 0.0,
    crop_middle=None,
    crop_right_frac: float = 0.0,
    expand_axes: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor, list[str]]:
    """Return ``(X, y, class_names, injected, families)`` for one task.

    ``task`` carries ``glyphs`` (the prompt) and either ``label_fn(fq, cedars_row)``
    — a fontquant/cedars-derived label — or ``source``/``col`` naming a column to
    read from the hand labels (``"cedars"``) or from fontquant.  ``injected``
    marks extra glyphs bulk-adding an under-represented class; ``families``
    carries one family name per sample so a held-out split can group by family
    (and never leak another instance of the same family into the negatives).
    With ``expand_axes``, variable fonts contribute their ``wght``/``wdth``
    extremes as extra samples with the same label.
    """
    classes = task["classes"]
    class_idx = {c: i for i, c in enumerate(classes)}
    label_fn = task.get("label_fn")
    X: list[torch.Tensor] = []
    y: list[int] = []
    injected_flags: list[bool] = []
    families: list[str] = []
    skipped_blank = 0

    def add(font, fam: str, label: str) -> None:
        nonlocal skipped_blank
        positions = axis_positions(font) if expand_axes else [None]
        for pos in positions:
            img = render_prompts(
                font, task["glyphs"], size, crop_bottom_frac, crop_middle,
                crop_right_frac, axis_position=pos,
            )
            if img is None or is_blank(img):
                skipped_blank += 1
                continue
            X.append(img)
            y.append(class_idx[label])
            injected_flags.append(False)
            families.append(fam)

    if label_fn is not None:
        for fam, font in family_to_font.items():
            label = label_fn(quantify_row(font.path), cedars.get(fam))
            if label in class_idx:
                add(font, fam, label)
    elif task["source"] == "cedars":
        for fam, row in cedars.items():
            val = row.get(task["col"], "").strip()
            if val not in class_idx:
                continue  # missing / not applicable / not assessed
            font = family_to_font.get(fam)
            if font is None:
                continue
            add(font, fam, val)
    else:  # fontquant, simple column lookup
        for fam, font in family_to_font.items():
            val = (quantify_row(font.path).get(task["col"], "")).strip()
            if val not in class_idx:
                continue
            add(font, fam, val)

    # ── Optional injection: render extra glyphs to bulk up an under-represented
    # class.  For a-storey we inject each font's real ``A`` (U+0041) as an extra
    # ``uppercase`` example, drawn from normal-lowercase fonts so the shape is a
    # genuine uppercase A rather than a re-labelled all-caps ``a`` slot.  Injected
    # examples are always at the default instance and carry their font's family.
    inj = task.get("inject")
    if inj is not None:
        rng = random.Random(seed)
        eligible: list[object] = []
        for font in family_to_font.values():
            fq = quantify_row(font.path)
            if fq.get("casing/lowercase_shapes", "").strip() != "lowercase":
                continue
            eligible.append(font)
        rng.shuffle(eligible)
        injected = 0
        for font in eligible:
            if injected >= inj["n"]:
                break
            img = render_prompts(
                font, inj["glyphs"], size, crop_bottom_frac, crop_middle, crop_right_frac
            )
            if img is None or is_blank(img):
                continue
            X.append(img)
            y.append(class_idx[inj["label"]])
            injected_flags.append(True)
            families.append(font.family.lower())
            injected += 1

    if not X:
        return (
            torch.empty(0),
            torch.empty(0, dtype=torch.long),
            classes,
            torch.zeros(0, dtype=torch.bool),
            [],
        )
    return (
        torch.stack(X),
        torch.tensor(y, dtype=torch.long),
        classes,
        torch.tensor(injected_flags, dtype=torch.bool),
        families,
    )


# ── Out-of-domain examples ───────────────────────────────────────────────────
_WEIRD_TAGS = ("/Special use/Symbols", "/Special use/Barcode", "/Special use/Redaction")

# Every glyph in a symbol/barcode/redaction font is non-``a``-like, so any slot
# is a valid "weird" example — we don't have to use the ``a`` slot.
_PROBE_GLYPHS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def build_weird(
    repo: str,
    size: int,
    glyphs: str = _PROBE_GLYPHS,
    crop_bottom_frac: float = 0.0,
    crop_middle=None,
    crop_right_frac: float = 0.0,
) -> tuple[torch.Tensor, list[str]]:
    """Render *any* glyph slot of the symbol/barcode/redaction fonts.

    Returns ``(X, families)`` where ``X`` is ``(N, 1, 1, H, W)`` and ``families``
    carries one family name per example, so a held-out split can group by font
    (and never leak a font's other glyphs into the negatives).
    """
    gf = GoogleFonts(repo)  # loads tags (its .fonts skips the weird ones — unused)
    family_tags = load_family_tags(repo)
    weird_families = {
        fam.lower()
        for fam, tags in family_tags.items()
        if any(tags.get(t, 0) > 0 for t in _WEIRD_TAGS)
    }

    by_family: dict[str, object] = {}
    for path in sorted(Path(repo).glob("ofl/*/*.ttf")):
        try:
            font = GoogleFont(path, gf)
        except Exception:
            continue
        fam = font.family.lower()
        if fam not in weird_families:
            continue
        if fam not in by_family or "regular" in path.name.lower():
            by_family[fam] = font

    X: list[torch.Tensor] = []
    families: list[str] = []
    for fam in sorted(by_family):
        font = by_family[fam]
        for ch in glyphs:
            img = render_prompts(font, ch, size, crop_bottom_frac, crop_middle, crop_right_frac)
            if img is None or is_blank(img):
                continue
            X.append(img)
            families.append(font.family)
    return torch.stack(X) if X else torch.empty(0), families


__all__ = [
    "axis_positions",
    "build_task",
    "build_weird",
    "crop_bottom",
    "crop_middle",
    "crop_right",
    "is_blank",
    "render_prompts",
]
