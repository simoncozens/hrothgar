"""Content-detector configuration.

The content detector answers, per codepoint, "is this glyph the codepoint it
claims to be?" as a *calibrated probability*, using one learned anchor vector
per codepoint.  Unlike a softmax classifier it is one-vs-rest, so every anchor's
probability can be low at once — which is what lets it reject a glyph that is
none of the codepoints it knows ("I can't tell if this is an e or not").
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hrothgar.content_detector.constants import PILOT_ANCHORS, PILOT_NEGATIVES


@dataclass
class ContentDetectorConfig:
    """Configuration for the content detector."""

    # The codepoints that get a learned anchor slot, and the wider pool random
    # negatives are drawn from.  Both are codepoints, not characters, so the
    # sidecar JSON round-trips them faithfully.
    anchors: list[int] = field(default_factory=lambda: list(PILOT_ANCHORS))
    negatives: list[int] = field(default_factory=lambda: list(PILOT_NEGATIVES))

    # Input rendering: each glyph is drawn into the em-space ``render.Frame``
    # below (see ``render_glyph_uniform``).  Aspect ratio is *kept* here, because
    # several content distinctions (l/I, o/O) are partly about proportion, unlike
    # the crop-to-ink framing the generator consumes.
    glyph_size: int = 256

    # The frame every glyph is drawn into: the em rectangle
    # ``x in [frame_left, frame_right]``, ``y in [frame_bottom, frame_top]``,
    # relative to the glyph origin and baseline.  This is the *dataset-global*
    # tight window measured by ``scripts/analyze_glyph_extents.py``: the legacy
    # window is 2.0em square, but the library's Latin Core ink occupies ~1.3em at
    # the 80th percentile, so half the canvas was empty.  1.40em buys 1.43x more
    # pixels per em at the same canvas, which is what lifts the capital/lowercase
    # cue in small-x-height designs (Dongle, Jomhuria, Chathura) above the
    # encoder's acuity.
    #
    # Deliberately *global*, not per-font.  A per-font ink envelope also
    # normalises each font's scale and baseline position; the encoder absorbs
    # that nuisance by giving up size sensitivity, and size is precisely the
    # case cue (measured: the uppercase anchors' hard negatives then outscore
    # their own positives).  Recorded in the sidecar so a checkpoint's images are
    # reproducible.
    frame_left: float = -0.20
    frame_right: float = 1.20
    frame_bottom: float = -0.36
    frame_top: float = 1.04

    # Batch curation.  A batch is ``anchors_per_batch * fonts_per_anchor``
    # cells (a cell is one font × one anchor); each cell emits one positive and
    # the hard/random negatives below.
    anchors_per_batch: int = 4
    fonts_per_anchor: int = 4
    hard_negatives_per_cell: int = 3
    random_negatives_per_cell: int = 1

    # Variable fonts contribute several *axis positions* per anchor, so the same
    # letter is seen at very different drawings (a font's optical-size / grade
    # extremes are effectively extra styles for free).  1 = default instance only.
    axis_positions_per_font: int = 3

    # Shared CNN encoder (same shape as the set-membership encoder).
    encoder_base_channels: int = 32
    encoder_feature_dim: int = 256
    encoder_downsample: int = 4
    # L2-normalized embedding space shared by glyphs and anchors.
    embedding_dim: int = 128
    # Cosine logit temperature.  A cosine similarity is scaled by this before
    # the bias, so negatives can be pushed to a low probability.
    initial_logit_scale: float = 10.0

    # Auxiliary height head.  It predicts the glyph's ink height *relative to the
    # font's own x-height* (``dataset.normalised_height``) from the same pooled
    # encoder features the embedding is built from.  It never feeds the content
    # head — that would leak a known-at-training-time target into the score — but
    # training it forces the shared encoder to *represent* height, which is the one
    # cue that separates case pairs whose shapes differ only by scale (o/O, c/C,
    # s/S).  Absolute height is useless here (a large-x-height font's ``o`` is as
    # tall as a small-cap-height font's ``O``), hence the per-font normalisation.
    height_hidden_dim: int = 32
    height_head_weight: float = 0.1

    # Training.
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3

    # Cosine-space margin hinges added to the BCE (see ``losses.margin_hinges``).
    # ``ranking`` forces a glyph's own anchor above every other anchor, which is
    # what keeps near-duplicate anchors (a/à) from both firing and what stops the
    # embedding collapsing onto a shared "it's ink" component.
    #
    # ``ranking`` is deliberately set *above* what the model currently achieves (a
    # hingeless run sits at rank_margin ~0.30), so the hinge actually bites: a
    # margin at or below the current geometry is inactive and buys nothing.  The
    # *absolute* ``positive`` hinge is left at 0.0 — it only asks a glyph to sit on
    # the right side of orthogonal from its own anchor, since a larger value just
    # pushes the embedding to a shared offset the anchor can follow, and the
    # ranking hinge already carries the discrimination.
    ranking_margin: float = 0.50
    positive_margin: float = 0.0
    # Within-anchor quality margin (see ``losses.quality_margin_hinge``).  A
    # *correct* glyph of an anchor must beat a *broken* (bad-skeleton) glyph of
    # the same anchor by this much in cosine space, so a "broken but
    # recognisable" glyph lands between "correct" and "other codepoint" instead
    # of collapsing onto either.  0 disables it.
    quality_margin: float = 0.0

    # Affine jitter applied to *every* rendered glyph (positives and negatives
    # alike — augmenting only the positives would let the model key on the
    # augmentation as a shortcut).  The point is style-invariance, which is the
    # whole difficulty of display/handwriting glyphs.
    augment: bool = True
    augment_max_shift: float = 0.08
    augment_max_rotation_deg: float = 8.0
    augment_max_scale: float = 0.12

    # Evaluation panel: held-out fonts used to report per-anchor AUC.
    panel_fonts: int = 100

    # Optional directory of extra negatives loaded from disk (e.g. harvested
    # generator failures).  Off until we wire it up.
    negative_image_dir: str | None = None

    # Generated-glyph review manifest (from scripts/labeling/build_critic_dataset.py)
    # and how many generated glyphs to sample into each training batch.  Routes by
    # label: acceptable -> positive; ambiguous -> negative for every anchor;
    # bad-skeleton -> within-anchor "broken" (margin-only, excluded from the BCE);
    # wrong-content and bad-execution are skipped (held out / not content).
    generated_manifest: str | None = None
    generated_per_batch: int = 16

    # Data loading.
    num_workers: int = 8

    def anchor_index(self) -> dict[int, int]:
        """Map anchor codepoint -> slot index."""
        return {codepoint: i for i, codepoint in enumerate(self.anchors)}

    def save_sidecar(self, model_path):
        """Save config as a sidecar JSON alongside the model weights."""
        import json
        from dataclasses import asdict
        from pathlib import Path as _Path

        from hrothgar.utils import git_short_sha

        config_path = _Path(str(model_path).replace(".pth", ".conf.json"))
        data = asdict(self)
        data["git_sha"] = git_short_sha()
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")

    @classmethod
    def from_sidecar(cls, model_path):
        """Load config from a sidecar JSON alongside the model weights."""
        import dataclasses
        import json
        from pathlib import Path as _Path

        config_path = _Path(str(model_path).replace(".pth", ".conf.json"))
        if not config_path.exists():
            raise FileNotFoundError(f"Config sidecar not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
