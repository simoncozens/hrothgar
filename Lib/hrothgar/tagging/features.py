"""Frozen gap features for the tag-prediction head.

A *gap* feature is a per-font vector derived from the frozen encoder's per-glyph
spatial map ``(G, F, h, w)``:

  * global average pool each glyph to ``(G, F)``,
  * pool across glyphs by **mean**, optionally concatenated with the cross-glyph
    standard deviation (``gap_mv``).

The encoder is frozen, so these features are computed once and cached; the head
is trained on them.  Rendering uses the *per-instance* tag cells from
:mod:`hrothgar.tagging.tags` (a cell is rendered at its own axis location, into
the per-font frame measured from the default instance).
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from hrothgar.googlefonts import GoogleFonts
from hrothgar.render import frame_for_codepoints, render
from hrothgar.similarity.config import FontStyleEmbedderConfig
from hrothgar.similarity.model import FontStyleEmbedder
from hrothgar.tagging.config import TaggingConfig
from hrothgar.tagging.tags import (
    TagCell,
    is_italic_instance,
    load_tag_cells,
    resolve_axis_position,
)

MIN_GLYPHS = 8


def pick_font_file(fonts, italic: bool = False):
    """Choose the upright (or italic) regular/default file for a family."""
    def is_italic_file(f):
        return "italic" in f.path.name.lower()

    pool = [f for f in fonts if is_italic_file(f) == italic]
    if not pool:
        pool = fonts  # no matching-style file; fall back to whatever exists
    # Prefer the variable/default file ("[axes]") or an explicit "regular".
    for f in pool:
        name = f.path.name.lower()
        if "[" in name or "regular" in name:
            return f
    return pool[0]


def gap_feature(
    encoder: torch.nn.Module,
    imgs: torch.Tensor,
    device: torch.device,
    use_variance: bool,
) -> np.ndarray:
    """Compute the per-font gap (or gap_mv) feature from rendered glyphs."""
    with torch.no_grad():
        feat = encoder(imgs.to(device))  # (G, F, h, w)
        gap = feat.mean(dim=(2, 3))  # (G, F)
        if use_variance:
            vec = torch.cat([gap.mean(dim=0), gap.std(dim=0)])  # (2F,)
        else:
            vec = gap.mean(dim=0)  # (F,)
    return vec.cpu().numpy().astype(np.float32)


@dataclass
class FeatureTable:
    """Precomputed gap features and per-cell tag labels."""

    features: np.ndarray  # (N, D) float32
    families: list[str]  # (N,)
    instances: list[str]  # (N,)
    tag_names: list[str]  # (T,)
    values: np.ndarray  # (N, T) float32, NaN where unassessed
    config: dict = field(default_factory=dict)

    @property
    def mask(self) -> np.ndarray:
        return ~np.isnan(self.values)

    @property
    def n_cells(self) -> int:
        return int(self.features.shape[0])

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "features": self.features,
                "families": self.families,
                "instances": self.instances,
                "tag_names": self.tag_names,
                "values": self.values,
                "config": self.config,
            },
            str(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeatureTable":
        # The table contains numpy arrays + python lists, so it needs
        # weights_only=False (these are our own locally-generated files).
        data = torch.load(str(path), map_location="cpu", weights_only=False)
        return cls(
            features=data["features"],
            families=data["families"],
            instances=data["instances"],
            tag_names=data["tag_names"],
            values=data["values"],
            config=data.get("config", {}),
        )


def build_feature_table(
    repo: str | Path,
    model: FontStyleEmbedder,
    config: TaggingConfig,
    device: torch.device,
    *,
    limit: int | None = None,
) -> FeatureTable:
    """Render every tag cell that carries a target tag and compute its gap feature."""
    gf = GoogleFonts(repo)
    input_cps = set(config.input_codepoints)
    by_family: dict[str, list] = defaultdict(list)
    for font in gf.fonts:
        if input_cps <= set(font.codepoints):
            by_family[font.family].append(font)

    cells: list[TagCell] = load_tag_cells(repo)
    target = set(config.tag_names)

    features: list[np.ndarray] = []
    families: list[str] = []
    instances: list[str] = []
    values: list[list[float]] = []

    for cell in tqdm(cells, desc="computing gap features"):
        if not (set(cell.values) & target):
            continue
        fonts = by_family.get(cell.family)
        if not fonts:
            continue
        if limit is not None and len(families) >= limit:
            break

        font = pick_font_file(fonts, italic=is_italic_instance(cell.instance))
        axis_position = resolve_axis_position(cell.instance, font.path)
        frame = frame_for_codepoints(font, config.input_codepoints)
        imgs = torch.stack(
            [
                render(
                    font,
                    codepoint=cp,
                    size=config.glyph_size,
                    axis_position=axis_position,
                    frame=frame,
                )[0]
                for cp in config.input_codepoints
            ]
        ).unsqueeze(1)  # (G, 1, H, W)
        imgs = imgs[~(imgs.amin(dim=(-2, -1)) > 0.995).squeeze(1)]
        if imgs.shape[0] < MIN_GLYPHS:
            continue

        features.append(gap_feature(model.encoder, imgs, device, config.use_variance))
        families.append(cell.family)
        instances.append(cell.instance)
        values.append(
            [cell.values.get(t, float("nan")) for t in config.tag_names]
        )

    feats = np.stack(features) if features else np.zeros((0, 0), dtype=np.float32)
    vals = np.asarray(values, dtype=np.float32) if values else np.zeros((0, 0))
    print(
        f"built feature table: {feats.shape[0]} cells x {feats.shape[1]} features, "
        f"{vals.shape[1]} tags"
    )
    return FeatureTable(
        features=feats,
        families=families,
        instances=instances,
        tag_names=list(config.tag_names),
        values=vals,
        config={"use_variance": config.use_variance},
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts"),
    )
    parser.add_argument("--model", default="models/similarity.pth")
    parser.add_argument("--out", default="output/tagging_features.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use-variance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    device = torch.device(args.device)

    cfg = FontStyleEmbedderConfig.from_sidecar(args.model)
    model = FontStyleEmbedder(cfg)
    model.load(args.model, device)
    model.to(device)
    model.eval()

    tag_cfg = TaggingConfig(use_variance=args.use_variance)
    table = build_feature_table(args.repo, model, tag_cfg, device, limit=args.limit)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.save(args.out)
    print(f"saved features to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
