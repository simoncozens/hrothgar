"""Weight-free probes of the shared :class:`GlyphEncoder`'s spatial resolution.

These experiments deliberately do **not** use the trained contrastive model, so
they answer a different question from "does the trained embedding put similar
fonts near each other?".  They answer *"can this architecture even carry the
signal that distinguishes two styles?"*, which is the prerequisite that the
trained model cannot out-learn.

Two things are measured, both independent of any learned weights:

1.  **Architecture resolution** — the encoder's receptive field and effective
    output stride in input pixels.  The encoder performs no pooling but does use
    stride-2 convolutions, so its output feature map sits on a ``1/downsample``
    grid.  A feature smaller than the stride, or an isolated single-pixel change,
    is diluted across an output cell and largely lost.

2.  **Variable-axis sweep** — a parametric font is rendered at the same glyph
    set and in the same per-font frame the similarity dataset uses, while one
    design axis is swept *away from its default instance*.  We measure (a) how
    much the *input* image changes, (b) how much of that change survives a
    ``downsample``-factor box pool (the encoder's output resolution), (c) how
    much the *output* feature map of a freshly-initialised (untrained) encoder
    changes, and (d) the *sensitivity* — the smallest axis nudge that still
    moves at least one output cell, which bounds the smallest typographic
    feature the architecture can resolve.

The untrained encoder cannot say whether a *trained* model would use the signal;
it can only say whether the signal is physically present above the architecture's
resolution floor.  That is exactly the question the "weak signal" finding raises.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from hrothgar.glyph_encoder import GlyphEncoder
from hrothgar.render import frame_for_codepoints, render
from hrothgar.similarity.config import DEFAULT_INPUT_CODEPOINTS

# How many pixels must change before a difference is considered "ink moved".
_CHANGE_THRESHOLD = 0.1
# One 8-bit grayscale level — the smallest change worth calling "detectable".
_GRAYSCALE = 1.0 / 255.0


@dataclass(frozen=True)
class EncoderGeometry:
    """Spatial resolution of a :class:`GlyphEncoder`, in input pixels.

    ``receptive_field`` is the input-pixel span that one output cell reads;
    ``effective_stride`` is the input-pixel distance between neighbouring output
    cells.  Features thinner than ``effective_stride`` are sub-sampled, and an
    isolated change smaller than the stride is averaged away by the downsampling.
    """

    input_size: int
    output_size: int
    receptive_field: int
    effective_stride: int

    @property
    def downsample(self) -> int:
        return self.effective_stride


def encoder_geometry(encoder: GlyphEncoder, input_size: int) -> EncoderGeometry:
    """Derive receptive field and output stride from the encoder's actual convs.

    Reads the kernel sizes and strides off the built network rather than
    re-deriving the layer list, so this stays correct if :class:`GlyphEncoder`
    changes.
    """
    receptive_field = 1
    stride = 1
    for module in encoder.net:
        if isinstance(module, nn.Conv2d):
            k = int(module.kernel_size[0])
            s = int(module.stride[0])
            receptive_field += (k - 1) * stride
            stride *= s
    output_size = math.ceil(input_size / stride)
    return EncoderGeometry(input_size, output_size, receptive_field, stride)


def variable_axes(path: Path) -> list[list]:
    """Return ``[[tag, min, default, max], ...]`` in fvar order (no ``[`` gate).

    Unlike :func:`hrothgar.dataset.fvar_axes`, this works for files whose names
    don't contain ``[...]`` (e.g. ``RobotoDelta-Roman-VF.ttf``).
    """
    from fontTools.ttLib import TTFont

    try:
        fvar = TTFont(path, lazy=True)["fvar"]
    except Exception:
        return []
    return [
        [a.axisTag, a.minValue, a.defaultValue, a.maxValue] for a in fvar.axes
    ]


def axis_position(axes: list[list], tag: str, value: float) -> list[float]:
    """Full design-coordinate list with ``tag`` set to ``value``, others default."""
    return [value if a[0] == tag else a[2] for a in axes]


def render_input_set(
    font, codepoints: list[int], glyph_size: int, axis_position: list[float] | None
) -> torch.Tensor:
    """Render the full input set in the dataset's per-font frame as ``(G, 1, H, W)``."""
    frame = frame_for_codepoints(font, codepoints)
    return torch.stack(
        [
            render(
                font,
                codepoint=cp,
                size=glyph_size,
                axis_position=axis_position,
                frame=frame,
            )[0]
            for cp in codepoints
        ]
    ).unsqueeze(1)


def _input_difference(base: torch.Tensor, other: torch.Tensor) -> dict:
    diff = (other - base).abs()
    return {
        "mean_abs": float(diff.mean()),
        "max_abs": float(diff.max()),
        "fraction_changed": float((diff > _CHANGE_THRESHOLD).float().mean()),
        "changed_pixels": int((diff > _CHANGE_THRESHOLD).sum()),
    }


def _survival_metrics(diff: torch.Tensor, pooled: torch.Tensor) -> dict:
    """Shared metrics for a difference pooled to some output resolution."""
    energy_full = float((diff**2).sum())
    energy_pooled = float((pooled**2).sum())
    return {
        "energy_full": energy_full,
        "energy_pooled": energy_pooled,
        "energy_ratio": energy_pooled / energy_full if energy_full > 0 else 0.0,
        "mean_abs_full": float(diff.abs().mean()),
        "mean_abs_pooled": float(pooled.abs().mean()),
        "changed_output_cells": int((pooled.abs() > _GRAYSCALE).sum()),
        "output_cells_total": int(pooled.numel()),
    }


def downsampling_survival(diff: torch.Tensor, factor: int) -> dict:
    """How much of a difference survives a ``factor``-box pool (energy + mean).

    ``diff`` is a signed ``(G, H, W)`` image difference.  ``mean_abs_full`` is the
    mean |change| at input resolution; ``mean_abs_pooled`` is the mean |change|
    at output resolution (after the signed difference is box-averaged, so
    neighbouring pixels of opposite sign partially cancel).

    ``energy_ratio`` spans ``[1/factor^2, 1/factor]``: a single isolated pixel
    lands at ``1/factor^2`` (its amplitude is divided by ``factor^2``), while a
    change that is smooth at the ``factor``-pixel scale (constant within each
    pooling block) saturates at ``1/factor``.  So the ratio separates "isolated
    sub-pixel change" from "extended feature change".

    ``changed_output_cells`` counts output-grid cells that move by at least one
    grayscale level after pooling — a direct, weight-free measure of how much of
    the encoder's output map the change can reach.
    """
    diff = diff.unsqueeze(1)  # (G, H, W) -> (G, 1, H, W)
    pooled = F.avg_pool2d(diff, factor)
    return {"factor": factor, **_survival_metrics(diff, pooled)}


def coarse_grid_survival(diff: torch.Tensor, grid_size: int) -> dict:
    """Survival through the summary's ``grid_size`` adaptive pool.

    The :class:`FontStyleEmbedder` max-pools the encoder output to
    ``coarse_grid_size`` before the linear head, so this is the resolution the
    *summary embedding* actually sees — independent of the encoder's own
    ``downsample`` (adaptive pooling normalises to the same grid either way).
    """
    diff = diff.unsqueeze(1)  # (G, H, W) -> (G, 1, H, W)
    pooled = F.max_pool2d(diff, (grid_size, grid_size))
    return {"grid_size": grid_size, **_survival_metrics(diff, pooled)}


def _encode_set(encoder: GlyphEncoder, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return encoder(images)  # (G, F, H/ds, W/ds)


def _output_difference(base: torch.Tensor, other: torch.Tensor) -> dict:
    b = base.flatten(1)
    o = other.flatten(1)
    diff = (o - b).norm(dim=1)
    return {
        "frobenius": float(diff.mean()),
        "relative": float((diff / b.norm(dim=1).clamp(min=1e-8)).mean()),
        "cosine": float(F.cosine_similarity(b, o, dim=1).mean()),
    }


def _axis_row(axes: list[list], tag: str) -> tuple[float, float, float]:
    """``(min, default, max)`` for ``tag`` in fvar order."""
    row = next(a for a in axes if a[0] == tag)
    return float(row[1]), float(row[2]), float(row[3])


def _sensitivity(
    font,
    codepoints: list[int],
    axes: list[list],
    tag: str,
    default: float,
    to: float,
    glyph_size: int,
    downsample: int,
    fine: int = 24,
) -> dict:
    """How much of an axis change is visible, and the smallest detectable nudge.

    Renders ``fine`` equally-spaced steps from the default instance toward
    ``to`` and, for each, measures the input difference and how many *output*
    cells (after a ``downsample``-box pool) move by at least one grayscale
    level.  The smallest ``|delta|`` that still moves an output cell is the
    axis-resolution floor — the analogue of the receptive-field bound expressed
    in design-axis units.
    """
    span = to - default
    base = render_input_set(
        font, codepoints, glyph_size, axis_position(axes, tag, default)
    )

    rows = []
    for i in range(1, fine + 1):
        delta = span * i / fine
        value = default + delta
        image = render_input_set(
            font, codepoints, glyph_size, axis_position(axes, tag, value)
        )
        diff = (image - base).squeeze(1)  # (G, H, W)
        pooled = F.avg_pool2d(diff.unsqueeze(1), downsample)  # (G, 1, H/ds, W/ds)
        rows.append(
            {
                "value": value,
                "delta": delta,
                "mean_abs": float(diff.abs().mean()),
                "changed_pixels": int((diff.abs() > _CHANGE_THRESHOLD).sum()),
                "changed_output_cells": int((pooled.abs() > _GRAYSCALE).sum()),
            }
        )

    first = next((r for r in rows if r["changed_output_cells"] > 0), None)
    # Coarse average slope of input change per design-axis unit over the span.
    mean_abs_per_unit = rows[-1]["mean_abs"] / abs(span) if span else 0.0
    return {
        "default": default,
        "to": to,
        "span": span,
        "fine_steps": fine,
        "mean_abs_per_unit": mean_abs_per_unit,
        "min_detectable_delta": first["delta"] if first else None,
        "min_detectable_mean_abs": first["mean_abs"] if first else None,
        "rows": rows,
    }


def sweep_axis(
    font,
    codepoints: list[int],
    *,
    axis_tag: str,
    glyph_size: int,
    steps: int,
    downsample: int,
    encoder_seeds: int,
    coarse_grid_size: int = 8,
    to: float | None = None,
) -> dict:
    """Sweep ``axis_tag`` from its default instance and measure differences.

    The sweep is anchored at the default instance (the base the input/output
    differences are measured against) and moves to ``to``.  When ``to`` is
    omitted it defaults to whichever endpoint is farthest from the default.

    Returns the axis range, per-step input-difference metrics, the downsampling
    survival at the encoder output and at the summary's ``coarse_grid_size``
    grid, the sensitivity analysis, and untrained-encoder output differences
    (averaged over ``encoder_seeds`` random initialisations).
    """
    axes = variable_axes(font.path)
    lo, default, hi = _axis_row(axes, axis_tag)
    if to is None:
        to = hi if abs(hi - default) >= abs(lo - default) else lo

    values = [default + (to - default) * i / (steps - 1) for i in range(steps)]

    images = {
        v: render_input_set(font, codepoints, glyph_size, axis_position(axes, axis_tag, v))
        for v in values
    }
    base = images[values[0]]  # the default instance
    full_swing = (images[to] - base).squeeze(1)  # (G, H, W)

    input_diffs = {v: _input_difference(base, images[v]) for v in values}
    survival = downsampling_survival(full_swing, downsample)
    summary_survival = coarse_grid_survival(full_swing, coarse_grid_size)
    sensitivity = _sensitivity(
        font, codepoints, axes, axis_tag, default, to, glyph_size, downsample
    )

    # Untrained encoder: does the architecture pass the extreme difference at all?
    output_diffs = []
    for seed in range(encoder_seeds):
        torch.manual_seed(seed)
        encoder = GlyphEncoder(downsample=downsample).eval()
        output_diffs.append(
            _output_difference(_encode_set(encoder, base), _encode_set(encoder, images[to]))
        )
    output = {
        key: float(sum(d[key] for d in output_diffs) / len(output_diffs))
        for key in output_diffs[0]
    }

    return {
        "axis": axis_tag,
        "font": str(font.path),
        "default": default,
        "to": to,
        "lo": lo,
        "hi": hi,
        "steps": values,
        "input_diffs": input_diffs,
        "downsampling_survival": survival,
        "summary_survival": summary_survival,
        "sensitivity": sensitivity,
        "output_diff": output,
    }


def run_probe(
    specs: list[dict],
    *,
    codepoints: list[int] | None = None,
    glyph_size: int = 128,
    downsample: int = 4,
    coarse_grid_size: int = 8,
    steps: int = 5,
    encoder_seeds: int = 5,
) -> dict:
    """Run the full probe over ``specs`` (each ``{path, axis}``) and return a dict."""
    codepoints = codepoints if codepoints is not None else list(DEFAULT_INPUT_CODEPOINTS)
    probe_encoder = GlyphEncoder(downsample=downsample)
    geometry = encoder_geometry(probe_encoder, glyph_size)

    results = []
    for spec in specs:
        from hrothgar.googlefonts import StandaloneFont

        font = StandaloneFont(spec["path"])
        results.append(
            sweep_axis(
                font,
                codepoints,
                axis_tag=spec["axis"],
                glyph_size=glyph_size,
                steps=steps,
                downsample=downsample,
                encoder_seeds=encoder_seeds,
                coarse_grid_size=coarse_grid_size,
                to=spec.get("to"),
            )
        )

    return {
        "geometry": {
            "input_size": geometry.input_size,
            "output_size": geometry.output_size,
            "receptive_field": geometry.receptive_field,
            "effective_stride": geometry.effective_stride,
        },
        "glyph_size": glyph_size,
        "downsample": downsample,
        "coarse_grid_size": coarse_grid_size,
        "num_glyphs": len(codepoints),
        "sweeps": results,
    }
