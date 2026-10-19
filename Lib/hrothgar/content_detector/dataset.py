"""Dataset and batch curation for the content detector.

The unit of training is a *cell*: one font × one anchor codepoint.  For each cell
the collate renders

* the **positive** — the anchor's glyph in this font; and
* **negatives** from the same font, drawn from a curated per-anchor pool:

  * *hard* negatives: the anchor's confusables (case sibling, Latin
    near-neighbours, digits, Cyrillic/Greek homoglyphs) and its diacritic
    variants (``é``/``è`` for ``e``);
  * *random* negatives: any other glyph the font happens to carry.

The batch sampler deliberately spreads each anchor's positives across *strata*
(sans / serif / display / handwriting), so no anchor can collapse onto a single
style — that cross-style spread is what forces the anchor to capture "e-ness"
rather than "this particular e".

Rendering happens inside ``__getitem__`` rather than in the collate, because
``GoogleFont`` (and therefore ``hb.Face``) is not picklable: cells are plain
``(path, anchor, stratum)`` tuples, so they cross the DataLoader worker boundary
but only tensors cross back.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Sampler

from hrothgar.content_detector.config import ContentDetectorConfig
from hrothgar.content_detector.constants import (
    CONFUSABLES,
    EXCLUDED_FAMILIES,
    nfd_base_codepoint,
)
from hrothgar.dataset import DatasetMaker, fvar_axes, style_bucket
from hrothgar.fontquant import FontQuantTable
from hrothgar.googlefonts import GoogleFonts
from hrothgar.render import Frame, render_glyph_uniform


# --------------------------------------------------------------------------- #
# Framing
# --------------------------------------------------------------------------- #


def content_frame(config: ContentDetectorConfig) -> Frame:
    """The em-space frame the content detector draws every glyph into.

    A dataset-global policy rather than a per-font one (see
    ``ContentDetectorConfig.frame_*``): framing is a property of the checkpoint,
    so training, the validation panel and the library audit all come through
    here, and the values are recorded in the config sidecar.
    """
    return Frame(
        config.frame_left, config.frame_right, config.frame_bottom, config.frame_top
    )


# --------------------------------------------------------------------------- #
# Axis positions
# --------------------------------------------------------------------------- #

# Axes that change a drawing the most, in the order we want to sample them.
# ``opsz`` and ``GRAD`` are effectively extra styles: a serif at optical-size
# maximum is a far more contrasted (Didone-like) drawing than its default.
AXIS_PRIORITY = ("opsz", "GRAD", "wght", "wdth")


def _axis_rank(tag: str) -> int:
    return AXIS_PRIORITY.index(tag) if tag in AXIS_PRIORITY else len(AXIS_PRIORITY)


def axis_positions_for_font(path, count: int, rng: random.Random) -> list[tuple]:
    """Up to ``count`` axis positions for a variable font (the default first).

    The first entries are single-axis extremes on :data:`AXIS_PRIORITY` axes; any
    remaining budget is filled with random per-axis combinations (min / default /
    max), which is how a font such as Roboto Serif can be seen at grade-min *and*
    optical-size-max at once — near enough a different Didone.  Non-variable fonts
    (and ``count <= 1``) yield the empty position only.
    """
    axes = fvar_axes(Path(path))
    if not axes or count <= 1:
        return [()]

    ordered = sorted(axes, key=lambda axis: (_axis_rank(axis[0]), axis[0]))
    positions: list[tuple] = [()]
    for tag, _lo, default, _hi in ordered:
        for value in (_lo, _hi):
            if value == default:
                continue
            # A full-length position, in fvar order, with all other axes at default.
            position = tuple(value if a[0] == tag else a[2] for a in axes)
            if position not in positions:
                positions.append(position)
            if len(positions) >= count:
                return positions

    attempts = 0
    while len(positions) < count and attempts < count * 20:
        attempts += 1
        position = tuple(rng.choice((a[1], a[2], a[3])) for a in axes)
        if position == tuple(a[2] for a in axes):
            position = ()  # an all-default draw is just the default instance
        if position not in positions:
            positions.append(position)
    return positions


# --------------------------------------------------------------------------- #
# Relative height
# --------------------------------------------------------------------------- #

# The glyph ink height / font x-height ratio is clamped here before being halved
# into a [0, 1] regression target for the auxiliary height head.
HEIGHT_RATIO_CLAMP = 2.0


def normalised_height(scale_y: float, x_height_em: float) -> float:
    """Glyph ink height relative to the font's x-height, mapped to ``[0, 1]``.

    ``scale_y / x_height`` is ~1 for an x-height letter and ~1.5 for a capital in
    *every* font, which is the content fact the case pairs (o/O, c/C, s/S) hinge
    on — absolute height is not, because a large-x-height font's ``o`` is as tall
    as a small-cap-height font's ``O``.
    """
    ratio = scale_y / max(x_height_em, 1e-6)
    return min(ratio, HEIGHT_RATIO_CLAMP) / HEIGHT_RATIO_CLAMP


# --------------------------------------------------------------------------- #
# Shape identity
# --------------------------------------------------------------------------- #

# Two renders are treated as the same glyph below this mean absolute pixel
# difference.  Identical outlines render bit-identically (0.0), while the closest
# genuine confusable pair seen in practice (o vs 0) differs by ~0.006, so the gap
# is unambiguous.
RENDER_IDENTITY_EPS = 1e-3

# A rendered glyph is "blank" below this ink fraction (ink is near 0.0 on a white
# 1.0 background, so ``image < 0.5`` marks ink pixels).
BLANK_INK_EPS = 1e-3


def is_blank_render(image: torch.Tensor, eps: float = BLANK_INK_EPS) -> bool:
    """Whether a rendered glyph ``(H, W)`` or ``(1, H, W)`` contains no ink."""
    return float((image < 0.5).float().mean()) < eps


def is_shape_identical(first: torch.Tensor, second: torch.Tensor) -> bool:
    """Whether two rendered glyphs share an outline (mean|diff| below the eps).

    Fonts build cross-script homoglyphs from the Latin outlines, so e.g. Cyrillic
    ``о`` renders bit-identically to Latin ``o`` while being a different codepoint.
    Such a glyph is not a negative (or a positive) of any kind — it is the same
    picture with two labels — so the dataset drops it wherever it would appear.
    """
    return float((first - second).abs().mean()) < RENDER_IDENTITY_EPS


# --------------------------------------------------------------------------- #
# Negative curation
# --------------------------------------------------------------------------- #


def diacritic_variants(codepoint: int, pool: Sequence[int]) -> list[int]:
    """Pool members sharing ``codepoint``'s base letter but not the codepoint.

    For ``e`` this is its precomposed accents (``é è ê ë``); for ``á`` it is the
    differently-accented and unaccented ``a`` forms.  These are the "same letter
    with a changed / missing diacritic" negatives.
    """
    base = nfd_base_codepoint(codepoint)
    return [
        cp for cp in pool if cp != codepoint and nfd_base_codepoint(cp) == base
    ]


def hard_negative_pool(codepoint: int, pool: Sequence[int]) -> list[int]:
    """Curated confusables plus every diacritic variant of ``codepoint``."""
    hard = list(CONFUSABLES.get(codepoint, ()))
    for cp in diacritic_variants(codepoint, pool):
        if cp not in hard:
            hard.append(cp)
    return hard


def sample_negatives(
    font_codepoints: set[int],
    anchor: int,
    pool: Sequence[int],
    *,
    hard_count: int,
    random_count: int,
    rng: random.Random,
) -> list[int]:
    """Pick up to ``hard_count`` curated and ``random_count`` random negatives.

    Only codepoints the font actually carries are eligible, and the anchor's own
    glyph is never returned.
    """
    hard = [
        cp
        for cp in hard_negative_pool(anchor, pool)
        if cp != anchor and cp in font_codepoints
    ]
    other = [cp for cp in pool if cp != anchor and cp in font_codepoints]

    chosen = rng.sample(hard, min(hard_count, len(hard)))
    chosen += rng.sample(other, min(random_count, len(other)))

    deduped: list[int] = []
    seen: set[int] = set()
    for cp in chosen:
        if cp != anchor and cp not in seen:
            seen.add(cp)
            deduped.append(cp)
    return deduped


def targets_for_codepoints(
    codepoints: Sequence[int], anchors: Sequence[int]
) -> torch.Tensor:
    """Build the ``(N, K)`` multi-hot target grid.

    ``targets[n, k]`` is 1 iff glyph ``n`` *is* anchor ``k``.  A codepoint with
    no anchor slot (a confusable, a diacritic variant, ``-1`` for an image loaded
    off disk) yields an all-zero row — a negative for every anchor.
    """
    index = {cp: k for k, cp in enumerate(anchors)}
    targets = torch.zeros((len(codepoints), len(anchors)), dtype=torch.float32)
    for n, cp in enumerate(codepoints):
        k = index.get(cp)
        if k is not None:
            targets[n, k] = 1.0
    return targets


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #


def augment_image(
    image: torch.Tensor, config: ContentDetectorConfig, generator: torch.Generator
) -> torch.Tensor:
    """Apply a small affine jitter to a ``(1, H, W)`` glyph.

    Ink stays 0 on a white 1 background; edge padding replicates the (white)
    margin, so a shift never introduces a hard border.  A no-op when
    ``config.augment`` is off.
    """
    if not config.augment:
        return image

    def rand(scale: float) -> float:
        return (torch.rand(1, generator=generator).item() * 2.0 - 1.0) * scale

    _, h, w = image.shape
    angle = rand(config.augment_max_rotation_deg) * math.pi / 180.0
    scale = 1.0 + rand(config.augment_max_scale)
    tx = rand(config.augment_max_shift) * 2.0
    ty = rand(config.augment_max_shift) * 2.0
    cos, sin = math.cos(angle) * scale, math.sin(angle) * scale
    theta = torch.tensor(
        [[[cos, -sin, tx], [sin, cos, ty]]], dtype=image.dtype
    )
    grid = F.affine_grid(theta, (1, 1, h, w), align_corners=False)
    return F.grid_sample(
        image[None], grid, mode="bilinear", padding_mode="border", align_corners=False
    )[0]


# --------------------------------------------------------------------------- #
# Cells, dataset, sampler
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Cell:
    """One font × one anchor × one axis-position *slot*.  Picklable across workers.

    ``positions`` holds every candidate position for the font and ``slot`` selects
    one, rotated by the epoch: a variable font should show different drawings as
    training proceeds, not the same three forever (which is exactly the kind of
    frozen target a model can memorise).
    """

    path: str
    anchor: int
    stratum: str
    positions: tuple[tuple[float, ...], ...] = ((),)
    slot: int = 0

    def position_for(self, epoch: int) -> tuple[float, ...]:
        return self.positions[(self.slot + epoch) % len(self.positions)]


@dataclass
class _RenderFont:
    """The minimal font interface rendering needs, rebuilt inside a worker."""

    path: Path
    hb_face: object
    codepoints: set[int]
    x_height_em: float = 1.0


_FONT_CACHE: dict[str, _RenderFont] = {}


def _x_height_em(font: _RenderFont) -> float:
    """The font's x-height in em units (``vertical_metrics`` x_height / upem).

    Falls back to the rendered ink height of ``x`` when the font carries no OS/2
    ``sxHeight``, so a missing metric can't blow the ratio up.
    """
    import uharfbuzz as hb

    upem = font.hb_face.upem or 1000
    x_height = hb.Font(font.hb_face).get_metric_position(hb.OTMetricsTag.X_HEIGHT)
    if x_height:
        return x_height / upem
    if ord("x") in font.codepoints:
        _image, geometry = render_glyph_uniform(font, ord("x"), 64)
        return max(geometry["scale_y"], 1e-3)
    return 1.0


def _render_font(path: str) -> _RenderFont:
    """Load (and cache) a render-only font handle for ``path``."""
    cached = _FONT_CACHE.get(path)
    if cached is None:
        import uharfbuzz as hb

        face = hb.Face(hb.Blob.from_file_path(path))
        cached = _RenderFont(Path(path), face, set(face.unicodes))
        cached.x_height_em = _x_height_em(cached)
        _FONT_CACHE[path] = cached
    return cached


class _CellDataset(TorchDataset):
    """Renders one cell's positive and negatives into tensors."""

    def __init__(self, cells: Sequence[Cell], config: ContentDetectorConfig):
        self.cells = list(cells)
        self.config = config
        self._pool = list(config.negatives)
        # Advanced by the training loop each epoch so the sampled negatives, the
        # augmentation jitter and the axis position are all re-drawn — otherwise
        # every epoch shows byte-identical images and the model can memorise them.
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.cells)

    def __getitem__(self, index: int) -> dict:
        cell = self.cells[index]
        font = _render_font(cell.path)
        # Seed per item *and per epoch*, so a cell's negatives and jitter are
        # reproducible but not frozen across epochs.
        rng = random.Random(
            ((index + 1) * 2654435761 + self.epoch * 0x9E3779B1) % (2**32)
        )
        size = self.config.glyph_size
        position = list(cell.position_for(self.epoch))
        frame = content_frame(self.config)

        positive, positive_geometry = render_glyph_uniform(
            font, cell.anchor, size, axis_position=position, frame=frame
        )
        if is_blank_render(positive):
            # An axis extreme (or a broken font) can render empty, and a blank
            # positive would teach the anchor to accept emptiness. Emit it as a
            # negative for every anchor instead.
            return {
                "images": positive.unsqueeze(0).unsqueeze(0),
                "targets": targets_for_codepoints([-1], self.config.anchors),
                "codepoints": torch.tensor([-1], dtype=torch.long),
                "height": torch.zeros(1),
                "height_mask": torch.zeros(1),
            }

        candidates = sample_negatives(
            font.codepoints,
            cell.anchor,
            self._pool,
            hard_count=self.config.hard_negatives_per_cell,
            random_count=self.config.random_negatives_per_cell,
            rng=rng,
        )

        codepoints = [cell.anchor]
        rasters = [positive]
        heights = [normalised_height(positive_geometry["scale_y"], font.x_height_em)]
        for cp in candidates:
            image, geometry = render_glyph_uniform(
                font, cp, size, axis_position=position, frame=frame
            )
            # Drop any codepoint that renders to the anchor's own outline: it is
            # the same shape with a contradictory target, not a negative.
            if is_shape_identical(positive, image):
                continue
            codepoints.append(cp)
            rasters.append(image)
            heights.append(normalised_height(geometry["scale_y"], font.x_height_em))

        generator = torch.Generator().manual_seed(rng.randrange(2**31))
        images = [
            augment_image(raster.unsqueeze(0), self.config, generator)
            for raster in rasters
        ]

        return {
            "images": torch.stack(images),  # (M, 1, H, W)
            "targets": targets_for_codepoints(codepoints, self.config.anchors),
            "codepoints": torch.tensor(codepoints, dtype=torch.long),
            "height": torch.tensor(heights, dtype=torch.float32),
            "height_mask": torch.ones(len(codepoints)),
        }


class ContentBatchSampler(Sampler):
    """Yield cell batches balanced over anchors and spread across strata.

    Each batch draws ``anchors_per_batch`` anchors (round-robin, so every anchor
    gets equal exposure) and, per anchor, ``fonts_per_anchor`` fonts taken from
    as many *distinct* strata as possible.
    """

    def __init__(
        self,
        cells: Sequence[Cell],
        config: ContentDetectorConfig,
        *,
        rng: random.Random | None = None,
    ):
        self.cells = list(cells)
        self.batch_size = config.anchors_per_batch * config.fonts_per_anchor
        self.anchors_per_batch = config.anchors_per_batch
        self.fonts_per_anchor = config.fonts_per_anchor
        self.anchors = list(config.anchors)
        self._rng = rng if rng is not None else random.Random(1234)
        self._cursor = 0

        self._by_anchor: dict[int, list[int]] = defaultdict(list)
        for i, cell in enumerate(self.cells):
            self._by_anchor[cell.anchor].append(i)

        self.batches_per_epoch = max(1, len(self.cells) // max(1, self.batch_size))

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        n_anchors = len(self.anchors)
        for _ in range(self.batches_per_epoch):
            batch: list[int] = []
            for offset in range(self.anchors_per_batch):
                anchor = self.anchors[(self._cursor + offset) % n_anchors]
                batch.extend(self._cells_for_anchor(anchor))
            self._cursor = (self._cursor + self.anchors_per_batch) % n_anchors
            self._rng.shuffle(batch)
            yield batch

    def _cells_for_anchor(self, anchor: int) -> list[int]:
        pool = self._by_anchor.get(anchor, [])
        if not pool:
            return []
        by_stratum: dict[str, list[int]] = defaultdict(list)
        for i in pool:
            by_stratum[self.cells[i].stratum].append(i)

        strata = sorted(by_stratum)
        self._rng.shuffle(strata)
        picked = [
            self._rng.choice(by_stratum[stratum])
            for stratum in strata[: self.fonts_per_anchor]
        ]
        while len(picked) < self.fonts_per_anchor:
            picked.append(self._rng.choice(pool))
        return picked


# --------------------------------------------------------------------------- #
# Dataset maker
# --------------------------------------------------------------------------- #


def exclude_families(
    fonts: Sequence, excluded: frozenset[str] = EXCLUDED_FAMILIES
) -> list:
    """Drop fonts whose family is held out of the dataset.

    Held-out families are mis-encoded in the source (see ``EXCLUDED_FAMILIES``),
    so as training data they are wrong labels; they are excluded from the eval
    panel too, so they can serve as a clean regression test.
    """
    return [font for font in fonts if font.family not in excluded]


def exclude_unusable_fonts(fonts: Sequence, font_metadata: FontQuantTable) -> list:
    """Drop fonts whose lowercase slots do not hold lowercase shapes, and blank fonts.

    An all-caps / small-caps font puts an uppercase (or small-cap) shape in every
    lowercase slot, so training the lowercase anchors on it teaches "``E`` is an
    ``e``" — this is how Alien Block's filled ``s`` ends up widening the ``s``
    anchor.  A blank font (Adobe Blank) puts an empty image in every slot, which is
    worse.  Both signals come from ``fontquant.csv``.
    """
    return [
        font
        for font in fonts
        if font_metadata.has_lowercase(font.path)
        and not font_metadata.is_blank(font.path)
    ]


class ContentDatasetMaker:
    """Build train/test content-detector loaders from a Google Fonts checkout."""

    def __init__(
        self,
        repo_url,
        config: ContentDetectorConfig,
        *,
        split_seed: int = 1234,
        canary_size: int | None = None,
        font_metadata: FontQuantTable | None = None,
    ):
        self.config = config
        self.split_seed = split_seed
        self._generated = self._load_generated(config.generated_manifest)
        self._gen_rng = random.Random(split_seed)

        # Require every anchor: a font missing one can't train that slot.
        self.googlefonts = GoogleFonts(
            str(repo_url), having=set(config.anchors), max_fonts=canary_size
        )
        fonts = exclude_families(self.googlefonts.fonts)
        if font_metadata is not None:
            before = len(fonts)
            fonts = exclude_unusable_fonts(fonts, font_metadata)
            print(
                f"Excluded {before - len(fonts)} fonts with no lowercase shapes / blank"
            )
        train_fonts, test_fonts = DatasetMaker._split_fonts_by_family(
            fonts, split_seed=split_seed
        )
        # Keep the font objects too: the evaluation panel needs to render them.
        self.train_fonts = train_fonts
        self.test_fonts = test_fonts
        self.train_cells = self._build_cells(train_fonts)
        self.test_cells = self._build_cells(test_fonts)
        self.train_dataset = _CellDataset(self.train_cells, config)
        self.test_dataset = _CellDataset(self.test_cells, config)
        print(
            f"Content detector: {len(self.train_cells)} train / "
            f"{len(self.test_cells)} test cells "
            f"({len(train_fonts)} / {len(test_fonts)} fonts)"
        )

    def _build_cells(self, fonts) -> list[Cell]:
        cells: list[Cell] = []
        for index, font in enumerate(fonts):
            # Seed per font so the sampled positions don't depend on the order the
            # library hands us the fonts.
            rng = random.Random(self.split_seed + index)
            # More candidates than we use per epoch, so the slot rotation keeps
            # showing the font at new drawings as training proceeds.
            candidates = tuple(
                axis_positions_for_font(
                    font.path, self.config.axis_positions_per_font * 3, rng
                )
            )
            slots = min(self.config.axis_positions_per_font, len(candidates))
            stratum = style_bucket(font)
            for slot in range(slots):
                for anchor in self.config.anchors:
                    cells.append(
                        Cell(str(font.path), anchor, stratum, candidates, slot)
                    )
        return cells

    def train_loader(self) -> DataLoader:
        sampler = ContentBatchSampler(self.train_cells, self.config)
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def test_loader(self) -> DataLoader:
        sampler = ContentBatchSampler(self.test_cells, self.config)
        return DataLoader(
            self.test_dataset,
            batch_sampler=sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def set_epoch(self, epoch: int) -> None:
        """Re-draw each cell's negatives, jitter and axis position for ``epoch``."""
        self.train_dataset.epoch = epoch
        self.test_dataset.epoch = epoch

    def _load_generated(self, manifest_path: str | None) -> list[dict]:
        """Parse the review manifest into the train subset we actually consume.

        Only ``split == "train"`` glyphs with a label we route are kept:

        * acceptable   -> positive (bridges the GT/generated gap);
        * ambiguous    -> negative for every anchor (open-set reject);
        * bad-skeleton -> "broken" (within-anchor quality margin);

        wrong-content and bad-execution are deliberately skipped — wrong-content
        is held out for evaluation, and bad-execution is a style signal that
        belongs to set_membership, not content.
        """
        if not manifest_path:
            return []
        path = Path(manifest_path)
        if not path.exists():
            print(f"Warning: generated manifest not found: {path}")
            return []
        records: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("split") != "train":
                continue
            if rec.get("label") not in ("acceptable", "ambiguous", "bad-skeleton"):
                continue
            records.append(rec)
        return records

    def _generated_batch(self, count: int):
        """Sample ``count`` generated glyphs and route them into batch tensors.

        Returns ``(images, targets, codepoints, broken)`` or ``None`` when no
        manifest is configured / has no usable records.  ``broken`` marks the
        bad-skeleton glyphs, which claim their anchor (so the ranking/positive
        hinges still apply) but are excluded from the BCE and enter only the
        quality margin.  Generated glyphs carry no height target; the collate
        masks them out of the height loss.
        """
        if not self._generated:
            return None
        from PIL import Image

        anchor_index = self.config.anchor_index()
        records = self._gen_rng.sample(self._generated, min(count, len(self._generated)))
        root = Path(self.config.generated_manifest).parent

        images, target_rows, codepoints, broken = [], [], [], []
        for rec in records:
            img_path = root / rec["file"]
            if not img_path.exists():
                continue
            cp = int(rec["codepoint"])
            k = anchor_index.get(cp)
            label = rec["label"]

            if label == "ambiguous":
                target_rows.append(torch.zeros(len(self.config.anchors)))
                codepoints.append(-1)
                broken.append(False)
            elif k is None:
                # acceptable / bad-skeleton for a codepoint with no anchor slot.
                continue
            else:
                row = torch.zeros(len(self.config.anchors))
                row[k] = 1.0
                target_rows.append(row)
                codepoints.append(cp)
                broken.append(label == "bad-skeleton")

            image = Image.open(img_path).convert("L")
            images.append(
                torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0)
            )

        if not images:
            return None
        return (
            torch.stack(images),  # (M, 1, H, W)
            torch.stack(target_rows),  # (M, K)
            torch.tensor(codepoints, dtype=torch.long),  # (M,)
            torch.tensor(broken, dtype=torch.bool),  # (M,)
        )

    def collate_fn(self, items: list[dict]) -> dict:
        images = torch.cat([item["images"] for item in items], dim=0)
        targets = torch.cat([item["targets"] for item in items], dim=0)
        codepoints = torch.cat([item["codepoints"] for item in items], dim=0)
        heights = torch.cat([item["height"] for item in items], dim=0)
        height_mask = torch.cat([item["height_mask"] for item in items], dim=0)

        # Rendered cells are all "correct" (not broken), so they participate in
        # the BCE and never the quality margin.
        n_rendered = images.shape[0]
        broken = torch.zeros(n_rendered, dtype=torch.bool)
        bce_mask = torch.ones(n_rendered)

        gen = self._generated_batch(self.config.generated_per_batch)
        if gen is not None:
            g_images, g_targets, g_codepoints, g_broken = gen
            images = torch.cat([images, g_images], dim=0)
            targets = torch.cat([targets, g_targets], dim=0)
            codepoints = torch.cat([codepoints, g_codepoints], dim=0)
            broken = torch.cat([broken, g_broken], dim=0)
            # Broken glyphs are excluded from the BCE (margin-only).
            bce_mask = torch.cat([bce_mask, (~g_broken).float()], dim=0)
            # Generated glyphs carry no height target.
            heights = torch.cat([heights, torch.zeros(g_images.shape[0])], dim=0)
            height_mask = torch.cat(
                [height_mask, torch.zeros(g_images.shape[0])], dim=0
            )

        return {
            "images": images,
            "targets": targets,
            "codepoints": codepoints,
            "height": heights,
            "height_mask": height_mask,
            "broken": broken,
            "bce_mask": bce_mask,
        }
