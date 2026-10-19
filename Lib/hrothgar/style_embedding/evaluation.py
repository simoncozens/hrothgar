"""Shared evaluation panel and retrieval suite for the font similarity embedder.

`scripts/similarity_eval.py` and the training validation loop both build an
`EvalPanel` and run `run_suite` over it, so a logged number means the same thing
in both places.  The metric definitions themselves live in ``metrics.py``.

A panel is a *fixed, seeded* set of fonts plus their labels and the variable-font
axis perturbations to sweep.  Renders are deterministic, so the panel can cache
its glyph images once and then validation only costs encoder forward passes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from hrothgar.dataset import fvar_axes
from hrothgar.render import geometry_tensor, render_glyph_with_geometry
from hrothgar.style_embedding.config import FontStyleEmbedderConfig
from hrothgar.style_embedding.metrics import (
    geometry_stats,
    label_agreement_at_k,
    membership_ranking,
    pairing_scores,
    sibling_retrieval,
    signal_attribution,
)
from hrothgar.style_embedding.render_utils import render_input_set

# Variable axes to sweep, and the "these should still look like the same font"
# reading of each.
TARGET_AXES = ("wdth", "wght", "slnt", "ital", "opsz")
DEFAULT_KS = (1, 3, 5)
_BLANK_THRESHOLD = 0.995


@dataclass
class EvalPanel:
    """A fixed evaluation set: fonts, labels, axis perturbations, cached images."""

    config: FontStyleEmbedderConfig
    fonts: list
    families: list[str]
    categories: list[str]
    # tag -> [(index into ``fonts``, axis position to perturb to)]
    axis_specs: dict[str, list[tuple[int, list[float]]]] = field(default_factory=dict)
    stability_fonts: int = 0
    images: torch.Tensor | None = None  # (P, G, 1, H, W) when cached
    # Reference descriptors for signal attribution (see ``_descriptors``).
    proportion: torch.Tensor | None = None  # (P, 10)
    shape: torch.Tensor | None = None  # (P, 64)


def _descriptors(font, config: FontStyleEmbedderConfig):
    """Independent references: macro proportions (geometry) and mean shape.

    ``proportion`` is the mean and standard deviation of the five geometry
    labels across the input glyph set.  ``shape`` is the mean 8x8 crop-to-ink
    render across the set, so it carries shape but not proportions.  Neither is
    derived from the model, so correlating the embedding with them says what
    the embedding is actually reading.
    """
    geoms: list[torch.Tensor] = []
    shapes: list[torch.Tensor] = []
    for codepoint in config.input_codepoints:
        image, geometry = render_glyph_with_geometry(font, codepoint, config.glyph_size)
        geoms.append(geometry_tensor(geometry))
        pooled = F.adaptive_avg_pool2d(image.unsqueeze(0).unsqueeze(0), (8, 8))
        shapes.append(pooled.flatten())
    stacked = torch.stack(geoms)
    return torch.cat([stacked.mean(0), stacked.std(0)]), torch.stack(shapes).mean(0)


def _axis_position(axes: list[list], tag: str, value: float) -> list[float]:
    return [value if a[0] == tag else a[2] for a in axes]


def _extreme(axes: list[list], tag: str) -> float | None:
    row = next((a for a in axes if a[0] == tag), None)
    if row is None:
        return None
    _tag, lo, default, hi = row
    extreme = hi if abs(hi - default) >= abs(lo - default) else lo
    return None if extreme == default else extreme


def build_eval_panel(
    fonts: list,
    config: FontStyleEmbedderConfig,
    *,
    max_fonts: int | None = None,
    seed: int = 0,
    axes: tuple[str, ...] = TARGET_AXES,
    stability_fonts: int = 0,
    cache_images: bool = False,
    with_descriptors: bool = False,
) -> EvalPanel:
    """Select a seeded panel, dropping fonts that render a blank glyph."""
    # Sort before sampling so the panel is reproducible regardless of the order
    # the caller's font list arrives in.
    selected = sorted(fonts, key=lambda f: str(f.path))
    if max_fonts is not None and len(selected) > max_fonts:
        selected = random.Random(seed).sample(selected, max_fonts)
        selected.sort(key=lambda f: str(f.path))

    kept: list = []
    families: list[str] = []
    categories: list[str] = []
    images: list[torch.Tensor] = []
    proportions: list[torch.Tensor] = []
    shapes: list[torch.Tensor] = []
    axis_specs: dict[str, list[tuple[int, list[float]]]] = {t: [] for t in axes}

    for font in selected:
        rendered = render_input_set(font, config.input_codepoints, config.glyph_size)
        if bool((rendered.amin(dim=(-2, -1)) > _BLANK_THRESHOLD).any()):
            continue
        idx = len(kept)
        kept.append(font)
        families.append(font.family)
        categories.append(font.category())
        if cache_images:
            images.append(rendered)
        if with_descriptors:
            proportion, shape = _descriptors(font, config)
            proportions.append(proportion)
            shapes.append(shape)
        fvar = fvar_axes(font.path)
        if fvar:
            for tag in axes:
                value = _extreme(fvar, tag)
                if value is not None:
                    axis_specs[tag].append((idx, _axis_position(fvar, tag, value)))

    panel = EvalPanel(
        config=config,
        fonts=kept,
        families=families,
        categories=categories,
        axis_specs={t: v for t, v in axis_specs.items() if v},
        stability_fonts=stability_fonts,
    )
    if cache_images and images:
        panel.images = torch.stack(images)
    if with_descriptors and proportions:
        panel.proportion = torch.stack(proportions)
        panel.shape = torch.stack(shapes)
    return panel


def _to_space(
    model, summaries: torch.Tensor, device: torch.device, space: str
) -> torch.Tensor:
    if space == "summary":
        return summaries
    with torch.no_grad():
        return F.normalize(model.projection(summaries.to(device)), p=2, dim=-1).cpu()


def _encode_images(
    model, images: torch.Tensor, device: torch.device, chunk: int = 32
) -> torch.Tensor:
    out = []
    for start in range(0, images.shape[0], chunk):
        with torch.no_grad():
            out.append(model.encode(images[start : start + chunk].to(device)).cpu())
    return torch.cat(out)


def embed_panel(
    model, panel: EvalPanel, device: torch.device, *, space: str = "summary"
) -> torch.Tensor:
    """Embed the panel's fonts, using cached images when available."""
    if panel.images is not None:
        summaries = _encode_images(model, panel.images, device)
    else:
        summaries = torch.stack(
            [model.compute_embedding(f, device) for f in panel.fonts]
        )
    return _to_space(model, summaries, device, space)


def _axis_embeddings(
    model, panel: EvalPanel, tag: str, device: torch.device, space: str
):
    """``(anchors, embeddings)`` for the valid axis perturbations of ``tag``."""
    anchors: list[int] = []
    embeddings: list[torch.Tensor] = []
    for idx, position in panel.axis_specs[tag]:
        try:
            emb = model.compute_embedding(
                panel.fonts[idx], device, axis_position=position
            )
        except ValueError:
            continue
        anchors.append(idx)
        embeddings.append(emb)
    if not embeddings:
        return [], torch.empty(0, panel.config.encoder_feature_dim)
    return anchors, _to_space(model, torch.stack(embeddings), device, space)


def sampling_stability(
    model, panel: EvalPanel, device: torch.device, *, seed: int = 0
) -> dict:
    """Cosine between two disjoint random glyph subsets of the same font."""
    rng = random.Random(seed)
    cosines: list[float] = []
    for i in range(min(panel.stability_fonts, len(panel.fonts))):
        if panel.images is not None:
            images = panel.images[i]
        else:
            images = render_input_set(
                panel.fonts[i], panel.config.input_codepoints, panel.config.glyph_size
            )
        images = images[~(images.amin(dim=(-2, -1)) > _BLANK_THRESHOLD).squeeze(1)]
        n = images.shape[0]
        if n < 4:
            continue
        take = min(panel.config.glyph_sample_size, n) // 2 * 2
        perm = rng.sample(range(n), take)
        mask = torch.zeros(2, n)
        mask[0, perm[: take // 2]] = 1
        mask[1, perm[take // 2 :]] = 1
        batch = images.unsqueeze(0).expand(2, -1, -1, -1, -1)
        with torch.no_grad():
            emb = model.encode(batch.to(device), glyph_mask=mask.bool().to(device))
        emb = F.normalize(emb.float(), dim=-1)
        cosines.append(float((emb[0] * emb[1]).sum()))
    if not cosines:
        return {"n": 0, "mean": 0.0, "min": 0.0, "p10": 0.0}
    t = torch.tensor(cosines)
    return {
        "n": len(cosines),
        "mean": float(t.mean()),
        "min": float(t.min()),
        "p10": float(t.quantile(0.1)),
    }


def membership_eval(
    model,
    panel: EvalPanel,
    device: torch.device,
    *,
    codepoints_per_font: int = 4,
    space: str = "summary",
    seed: int = 0,
) -> dict:
    """Does a single glyph rank its own font first?  (Set-membership score.)

    For a shared sample of glyph slots, embeds each glyph on its own and ranks
    it against every panel font's reference vector (built from the full glyph
    set).  This is the generator-evaluation use case: "does this glyph look
    like it belongs to this font?".
    """
    if not panel.fonts:
        return {"recall": {}, "mrr": 0.0, "auc": 0.0, "n": 0}
    if panel.images is not None:
        images = panel.images
    else:
        images = torch.stack(
            [
                render_input_set(
                    f, panel.config.input_codepoints, panel.config.glyph_size
                )
                for f in panel.fonts
            ]
        )
    n_glyphs = images.shape[1]
    slots = random.Random(seed).sample(
        range(n_glyphs), min(codepoints_per_font, n_glyphs)
    )

    glyph_vecs: list[torch.Tensor] = []
    own: list[int] = []
    mask = torch.zeros(len(slots), n_glyphs)
    for j, slot in enumerate(slots):
        mask[j, slot] = 1
    for i in range(len(panel.fonts)):
        batch = images[i].unsqueeze(0).expand(len(slots), -1, -1, -1, -1)
        with torch.no_grad():
            emb = model.encode(batch.to(device), glyph_mask=mask.bool().to(device))
        glyph_vecs.append(emb.cpu())
        own.extend([i] * len(slots))

    if panel.images is not None:
        font_vecs = _encode_images(model, panel.images, device)
    else:
        font_vecs = torch.stack(
            [model.compute_embedding(f, device) for f in panel.fonts]
        )
    return membership_ranking(
        _to_space(model, torch.cat(glyph_vecs), device, space),
        _to_space(model, font_vecs, device, space),
        own,
    )


def run_suite(
    model,
    panel: EvalPanel,
    device: torch.device,
    *,
    space: str = "summary",
    ks: tuple[int, ...] = DEFAULT_KS,
    top_k: int = 5,
    embeddings: torch.Tensor | None = None,
    membership_codepoints: int = 0,
    seed: int = 0,
) -> dict:
    """Compute the full retrieval suite for ``panel``; returns a nested dict."""
    base = (
        embeddings
        if embeddings is not None
        else embed_panel(model, panel, device, space=space)
    )
    out: dict = {
        "family_sibling": sibling_retrieval(base, panel.families, ks=ks).as_dict(),
        "category_agreement": label_agreement_at_k(base, panel.categories, k=top_k),
        "geometry": geometry_stats(base),
    }
    if panel.proportion is not None and panel.shape is not None:
        out["attribution"] = signal_attribution(
            base, {"proportion": panel.proportion, "shape": panel.shape}
        )
    axis: dict[str, dict] = {}
    for tag in panel.axis_specs:
        anchors, embeddings_t = _axis_embeddings(model, panel, tag, device, space)
        if anchors:
            axis[tag] = {
                "family": pairing_scores(
                    base, anchors, embeddings_t, base_groups=panel.families
                ),
                "anchor": pairing_scores(base, anchors, embeddings_t),
            }
    if axis:
        out["axis_invariance"] = axis
    if panel.stability_fonts:
        out["sampling_stability"] = sampling_stability(model, panel, device, seed=seed)
    if membership_codepoints:
        out["membership"] = membership_eval(
            model,
            panel,
            device,
            codepoints_per_font=membership_codepoints,
            space=space,
            seed=seed,
        )
    return out


def flatten_metrics(metrics: dict, prefix: str = ""):
    """Yield ``(scalar_name, float)`` for a nested metrics dict."""
    for key, value in metrics.items():
        name = f"{prefix}{str(key).replace(' ', '_')}"
        if isinstance(value, dict):
            yield from flatten_metrics(value, name + "/")
        else:
            yield name, float(value)
