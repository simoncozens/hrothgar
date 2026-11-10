"""Frozen text-arm similarity index: library positions + the learned combiner.

The text-arm similarity check (:mod:`hrothgar.similarity.text`) scores a font by
comparing *auto-extractable* features — fontquant geometry quantifiers plus the
small glyph-feature recognisers — against every font in the library, combining
the per-axis similarities with a learned linear/softmax combiner.

Before this module existed, every invocation re-scanned the Google Fonts
checkout, re-measured every library font with ``fontquant``, re-ran the
recognisers over all of them and re-trained the combiner.  That is minutes of
work per query and needs a full checkout — which makes it impossible to ship to
users as a Python package.

So: build the index once, freeze it, ship it.

The index contains

* **positions** — the raw per-axis feature data for every family in the library
  (fontquant quantities + the standardization statistics a query font must use,
  recogniser verdicts, and tag vectors), in a self-describing form,
* **the combiner** — the learned per-axis weights ``w`` and bias ``b``, trained
  on the text-arm half of the library, and
* **the text-arm row set** — which families a query is ranked against (positions
  for the rest are kept so *any* family can be queried offline).

and it can be *rebuilt* from supervision data.  Rebuilding is the expensive
path; querying is O(N × d) and needs only the fontquant measurements (looked up
in the measured library, or taken live for fonts it has not seen) plus the
recogniser checkpoints.

Usage::

    # build/refresh the index (needs a Google Fonts checkout + supervised pairs)
    python -m hrothgar.similarity.text_index --repo ~/others-repos/fonts

    # query it (from hrothgar.similarity.text)
    python -m hrothgar.similarity.text --ttf MyFont-Regular.ttf
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from hrothgar.fontquant import default_fontquant_csv
from hrothgar.similarity.axis_combiner import (
    _pair_weight_matrices,
    _recogniser_models_dir,
    _standardize_tensor,
    _train_combiner,
    bank_similarity_tensor,
    build_feature_bank,
    categorical_code_rows,
    categorical_sim_row,
    numeric_rows,
    numeric_sim_row,
    recogniser_code_rows,
    tag_cosine_row,
    tag_rows,
)
from hrothgar.similarity.metrics import pairwise_auc
from hrothgar.similarity.supervision import (
    JUDGEMENTS_PATH,
    SIMILARITIES_PATH,
    build_supervision,
    text_half_supervision,
)
from hrothgar.utils import device, git_short_sha

INDEX_KIND = "hrothgar.similarity.text_index"
INDEX_VERSION = 1

#: The index that ships inside the package.
DEFAULT_INDEX_PATH = Path(__file__).resolve().parent / "text_index.pth"

#: Feature source used by the text arm (see ``axis_combiner.build_feature_bank``).
FEATURE_SOURCE = "recogniser"


# ── Anchor features ───────────────────────────────────────────────────────────
@dataclass
class AnchorFeatures:
    """One font's features, laid out to match a :class:`TextIndex` axis order.

    ``numeric`` holds the *standardized* ``(value, presence)`` pair per axis (in
    the same space as the library, using the index's frozen statistics);
    ``raw_numeric`` keeps the pre-standardization values for ``--explain``.
    """

    label: str
    numeric: dict[str, tuple[torch.Tensor, torch.Tensor]]
    raw_numeric: dict[str, torch.Tensor]
    categorical: dict[str, int]
    tags: dict[str, tuple[torch.Tensor, torch.Tensor]]
    family: str | None = None


# ── The index ─────────────────────────────────────────────────────────────────
@dataclass
class TextIndex:
    """Frozen text-arm library positions + the learned per-axis combiner.

    ``families``/``numeric``/``categorical``/``tags`` hold a row for every
    family in the library (so any family can be *queried* offline), while
    ``rankable`` lists the rows that belong to the text arm (the families a
    query is actually ranked against).
    """

    families: list[str]
    axis_names: list[str]
    numeric: dict[str, dict]
    categorical: dict[str, dict]
    tags: dict[str, dict]
    w: torch.Tensor
    b: torch.Tensor
    rankable: list[int] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._lib_numeric = {
            name: _standardize_tensor(ax["values"], ax["means"], ax["stds"])
            for name, ax in self.numeric.items()
        }
        self._family_index = {f.lower(): i for i, f in enumerate(self.families)}
        mask = torch.zeros(len(self.families), dtype=torch.bool)
        if self.rankable:
            mask[torch.tensor(self.rankable, dtype=torch.long)] = True
        else:
            mask[:] = True
        self._rank_mask = mask

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kind": INDEX_KIND,
                "version": INDEX_VERSION,
                "families": self.families,
                "axis_names": self.axis_names,
                "numeric": self.numeric,
                "categorical": self.categorical,
                "tags": self.tags,
                "w": self.w,
                "b": self.b,
                "rankable": self.rankable,
                "meta": self.meta,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_INDEX_PATH) -> TextIndex:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"text index {path} not found; build one with "
                "`python -m hrothgar.similarity.text_index --repo <fonts checkout>`"
            )
        payload = torch.load(path, map_location="cpu", weights_only=True)
        kind = payload.get("kind")
        if kind != INDEX_KIND:
            raise ValueError(f"{path} is not a hrothgar text index (kind={kind!r})")
        version = payload.get("version")
        if version != INDEX_VERSION:
            raise ValueError(
                f"{path} has index version {version!r}, this code expects {INDEX_VERSION}; rebuild it"
            )
        return cls(
            families=list(payload["families"]),
            axis_names=list(payload["axis_names"]),
            numeric=payload["numeric"],
            categorical=payload["categorical"],
            tags=payload["tags"],
            w=payload["w"],
            b=payload["b"],
            rankable=list(payload.get("rankable") or []),
            meta=payload.get("meta", {}),
        )

    # ── lookup ───────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.families)

    def family_index(self, name: str) -> int | None:
        return self._family_index.get(name.strip().lower())

    @property
    def rankable_mask(self) -> torch.Tensor:
        """``(N,)`` bool mask of the rows a query is ranked against."""
        return self._rank_mask

    def features_for_family(self, name: str) -> AnchorFeatures | None:
        """The frozen row for a library family (no fontquant call, no checkout)."""
        i = self.family_index(name)
        if i is None:
            return None
        numeric = {
            axis: (V[i], P[i]) for axis, (V, P) in self._lib_numeric.items()
        }
        raw_numeric = {axis: ax["values"][i] for axis, ax in self.numeric.items()}
        categorical = {
            axis: int(ax["codes"][i]) for axis, ax in self.categorical.items()
        }
        tags = {
            axis: (ax["values"][i], ax["mask"][i]) for axis, ax in self.tags.items()
        }
        return AnchorFeatures(
            label=self.families[i],
            numeric=numeric,
            raw_numeric=raw_numeric,
            categorical=categorical,
            tags=tags,
            family=self.families[i],
        )

    # ── scoring ──────────────────────────────────────────────────────────────
    def similarity_row(self, af: AnchorFeatures) -> torch.Tensor:
        """Per-axis similarity of one anchor against the library: ``(N, K)``.

        When the anchor *is* a library family, its own row has self-similarity 1
        on the tag axes rather than the pairwise tensor's 0-diagonal convention;
        callers should exclude that row, as :meth:`rank` does.
        """
        mats: list[torch.Tensor] = []
        for name in self.axis_names:
            if name in self.numeric:
                V, P = self._lib_numeric[name]
                value, presence = af.numeric[name]
                mats.append(numeric_sim_row(value, presence, V, P))
            elif name in self.categorical:
                ax = self.categorical[name]
                mats.append(
                    categorical_sim_row(af.categorical[name], ax["codes"], ax["boolean"])
                )
            else:
                ax = self.tags[name]
                value, mask = af.tags[name]
                mats.append(tag_cosine_row(value, mask, ax["values"], ax["mask"]))
        return torch.stack(mats, dim=-1)

    def score(self, af: AnchorFeatures) -> torch.Tensor:
        """The combiner's score for every library family: ``(N,)``."""
        return self.similarity_row(af) @ self.w + self.b

    def rank_scores(self, af: AnchorFeatures) -> torch.Tensor:
        """Scores with the anchor's own row and non-text-arm rows set to ``-inf``."""
        scores = self.score(af)
        keep = self._rank_mask
        if af.family is not None:
            i = self.family_index(af.family)
            if i is not None:
                keep = keep.clone()
                keep[i] = False
        return torch.where(keep, scores, torch.full_like(scores, float("-inf")))

    def rank(self, af: AnchorFeatures, k: int = 20) -> list[tuple[str, float]]:
        """Top-``k`` ``(family, score)`` pairs within the text arm.

        The anchor's own row (when it is a library family) and every non-text-arm
        family are excluded.
        """
        scores = self.rank_scores(af)
        k = min(k, int(torch.isfinite(scores).sum()))
        if k <= 0:
            return []
        order = torch.argsort(scores, descending=True)[:k]
        return [(self.families[int(i)], float(scores[int(i)])) for i in order]

    # ── explanation ──────────────────────────────────────────────────────────
    def summary(self, af: AnchorFeatures) -> dict[str, str]:
        """Human-readable raw-feature strings for one anchor, keyed by axis."""
        out: dict[str, str] = {}
        for axis, ax in self.numeric.items():
            cols = ax["columns"]
            values = af.raw_numeric[axis]
            if _is_vector_axis(cols):
                vals = ", ".join(
                    f"{float(v):.2f}" for v in values if not torch.isnan(v)
                )
                prefix = cols[0].rsplit("[", 1)[0]
                out[axis] = f"{prefix}=[{vals}]"
            else:
                parts = [
                    f"{_short_column(col)}={_fmt(v)}" for col, v in zip(cols, values)
                ]
                out[axis] = " ".join(parts)
        for axis, ax in self.categorical.items():
            code = af.categorical[axis]
            out[axis] = ax["vocab"][code] if 0 <= code < len(ax["vocab"]) else "—"
        for axis, ax in self.tags.items():
            value, mask = af.tags[axis]
            present = [
                f"{tag}={float(v) * 100:.0f}"
                for tag, v, m in zip(ax["tags"], value, mask)
                if float(m) > 0 and float(v) > 0
            ]
            out[axis] = ", ".join(present) if present else "none"
        return out


def _is_vector_axis(columns: list[str]) -> bool:
    """True when every column name is ``name[i]`` (an expanded 26-vector)."""
    if not columns:
        return False
    prefix = columns[0].rsplit("[", 1)[0]
    return all(c.startswith(prefix + "[") and c.endswith("]") for c in columns)


# fontquant column categories, stripped for the ``--explain`` dump so a column
# reads as ``weight=…`` rather than ``appearance/weight=…``.
_FONTQUANT_PREFIXES = (
    "appearance/",
    "parametric/",
    "opentype/",
    "stroke_contrast/",
    "structure/",
    "casing/",
    "proportion/",
)


def _short_column(column: str) -> str:
    for prefix in _FONTQUANT_PREFIXES:
        if column.startswith(prefix):
            return column[len(prefix):]
    return column


def _fmt(value) -> str:
    v = float(value)
    return "—" if v != v else f"{v:.2f}"


# ── Building the index ────────────────────────────────────────────────────────
def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _input_stamp(paths) -> dict[str, float]:
    """``{absolute path: mtime}`` for the inputs that exist."""
    stamp: dict[str, float] = {}
    for p in paths:
        if p is None:
            continue
        path = Path(p)
        if path.exists():
            stamp[str(path.resolve())] = path.stat().st_mtime
    return stamp


def _checkpoint_stamp() -> dict[str, str]:
    """sha256 of each recogniser checkpoint currently in use."""
    models_dir = _recogniser_models_dir()
    stamp: dict[str, str] = {}
    for stem in ("a_storey", "e_crossbar", "finials", "c_overhang"):
        path = models_dir / f"{stem}.pth"
        if path.exists():
            stamp[stem] = _sha256(path)
    return stamp


def default_repo() -> str | None:
    """``$GOOGLE_FONTS_REPO``, else this project's historical checkout if present."""
    env = os.environ.get("GOOGLE_FONTS_REPO")
    if env:
        return env
    legacy = Path("/home/simon/others-repos/fonts")
    return str(legacy) if legacy.exists() else None


def build_index(
    repo: str,
    sim_path: str = SIMILARITIES_PATH,
    judge_path: str = JUDGEMENTS_PATH,
    *,
    combiner: str = "linear",
    steps: int = 2000,
    lr: float = 1e-2,
    margin: float = 0.25,
    gamma: float = 128.0,
    seed: int = 0,
    weight_decay: float = 1e-3,
    device_name: str | None = None,
    verbose: bool = True,
) -> TextIndex:
    """Recompute the text-arm index from a checkout + supervised pairs.

    This is the expensive path (scan the checkout, quantify every library font,
    run the recognisers, train the combiner).  It reproduces exactly what
    ``hrothgar.similarity.text`` used to do per query: the combiner is trained on
    the text-arm half of the library, but the frozen positions cover the whole
    library (standardized in the text arm's space) so any family can be queried
    offline.
    """
    dev = device(device_name)
    sup_full = build_supervision(repo, sim_path, judge_path, universe="full")
    sup = text_half_supervision(sup_full)
    if verbose:
        print(
            f"text-arm universe: {len(sup.fonts)} families "
            f"(of {len(sup_full.fonts)} in the library), "
            f"clusters={len(sup.clusters)} pos_pairs={len(sup.pos_pairs)} "
            f"neg_pairs={len(sup.neg_pairs)}",
            flush=True,
        )

    # Which full-universe rows form the text arm (the arm's training/ranking set).
    text_keys = {family.lower() for family in sup.families}
    text_rows = [
        i for i, family in enumerate(sup_full.families) if family.lower() in text_keys
    ]
    if [sup_full.families[i] for i in text_rows] != list(sup.families):
        raise RuntimeError(
            "text half is not an order-preserving subset of the full universe; "
            "cannot align combiner supervision with the feature bank"
        )

    # Positions for every family; standardization fitted on the text arm only.
    bank = build_feature_bank(
        sup_full, feature_source=FEATURE_SOURCE, stats_rows=text_rows
    )
    S, names = bank_similarity_tensor(_subset_bank(bank, text_rows))
    S = S.to(dev)

    pos, neg = _pair_weight_matrices(sup, dev)
    w, b = _train_combiner(
        S,
        pos,
        neg,
        combiner=combiner,
        steps=steps,
        lr=lr,
        margin=margin,
        gamma=gamma,
        seed=seed,
        weight_decay=weight_decay,
    )

    sim = S @ w + b
    pairwise = pairwise_auc(
        torch.stack([sim[a, c] for a, c, _ in sup.pos_pairs]),
        torch.stack([sim[a, c] for a, c, _ in sup.neg_pairs]),
    )
    if verbose:
        print(
            f"combiner trained ({combiner}, {steps} steps): "
            f"pairwise AUC = {pairwise['auc']:.3f}",
            flush=True,
        )

    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "created_mtime": time.time(),
        "git_sha": git_short_sha(),
        "feature_source": FEATURE_SOURCE,
        "repo": str(Path(repo).resolve()),
        "combiner": combiner,
        "steps": steps,
        "lr": lr,
        "margin": margin,
        "gamma": gamma,
        "seed": seed,
        "weight_decay": weight_decay,
        "num_families": len(bank["families"]),
        "pairwise_auc": pairwise["auc"],
        "inputs": _input_stamp(
            [sim_path, judge_path, default_fontquant_csv()]
        ),
        "checkpoints": _checkpoint_stamp(),
    }

    return TextIndex(
        families=list(bank["families"]),
        axis_names=list(names),
        numeric=bank["numeric"],
        categorical=bank["categorical"],
        tags=bank["tags"],
        w=w.cpu(),
        b=b.cpu(),
        rankable=text_rows,
        meta=meta,
    )


def _subset_bank(bank: dict, rows: list[int]) -> dict:
    """Restrict a feature bank to ``rows`` (keeping its frozen statistics)."""
    idx = torch.tensor(rows, dtype=torch.long)
    return {
        "families": [bank["families"][i] for i in rows],
        "axis_names": bank["axis_names"],
        "numeric": {
            name: {**ax, "values": ax["values"][idx]}
            for name, ax in bank["numeric"].items()
        },
        "categorical": {
            name: {**ax, "codes": ax["codes"][idx]}
            for name, ax in bank["categorical"].items()
        },
        "tags": {
            name: {
                **ax,
                "values": ax["values"][idx],
                "mask": ax["mask"][idx],
            }
            for name, ax in bank["tags"].items()
        },
    }


def index_is_stale(index_path: str | Path, index: TextIndex | None = None) -> tuple[bool, list[str]]:
    """Whether the supervision/feature inputs are newer than the frozen index.

    Returns ``(stale, reasons)``.  A missing input is not stale (we cannot
    rebuild from data we do not have), and a missing checkpoint is not stale
    either; a *changed* checkpoint is.
    """
    index_path = Path(index_path)
    if index is None:
        try:
            index = TextIndex.load(index_path)
        except (FileNotFoundError, ValueError):
            return True, [f"{index_path} missing or unreadable"]
    reasons: list[str] = []
    built = float(index.meta.get("created_mtime", 0.0))
    if not built:
        built = index_path.stat().st_mtime if index_path.exists() else 0.0
    for path_str in (index.meta.get("inputs") or {}):
        path = Path(path_str)
        if not path.exists():
            # The index records absolute paths from the machine that built it;
            # fall back to the bare filename (the files conventionally sit in
            # the working directory) so another checkout can still detect staleness.
            path = Path(path.name)
            if not path.exists():
                continue
        if path.stat().st_mtime > built + 1e-6:
            reasons.append(f"{path} is newer than the index")
    recorded = index.meta.get("checkpoints") or {}
    if recorded:
        current = _checkpoint_stamp()
        for stem, digest in current.items():
            if recorded.get(stem) not in (None, digest):
                reasons.append(f"recogniser checkpoint {stem}.pth changed")
    return bool(reasons), reasons


# ── Anchor features for a query font ──────────────────────────────────────────
def anchor_features(
    font,
    index: TextIndex,
    recognisers: dict | None = None,
) -> AnchorFeatures:
    """Extract a query font's features into the index's axis space.

    The font is measured with the ``fontquant`` package and run through the
    feature recognisers, then standardized with the index's frozen column
    statistics so it lands in the same space as the library.
    """
    raw = numeric_rows([font])
    codes = categorical_code_rows([font], {})
    codes.update(recogniser_code_rows([font], recognisers))
    tag_mats = tag_rows([font])

    numeric: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    raw_numeric: dict[str, torch.Tensor] = {}
    for axis, ax in index.numeric.items():
        values = torch.tensor([raw[axis][0]], dtype=torch.float32)
        # The anchor is standardized with the *library's* statistics, not its own.
        value, presence = _standardize_tensor(values, ax["means"], ax["stds"])
        numeric[axis] = (value[0], presence[0])
        raw_numeric[axis] = values[0]

    return AnchorFeatures(
        label=getattr(font, "family", Path(font.path).stem),
        numeric=numeric,
        raw_numeric=raw_numeric,
        categorical={axis: codes[axis][0] for axis in index.categorical},
        tags={axis: (tag_mats[axis][0][0], tag_mats[axis][1][0]) for axis in index.tags},
        family=None,
    )


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Build or refresh the frozen text-arm similarity index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--repo", default=None, help="Google Fonts checkout (default: $GOOGLE_FONTS_REPO)")
    p.add_argument("--list", default=SIMILARITIES_PATH)
    p.add_argument("--judgements", default=JUDGEMENTS_PATH)
    p.add_argument("--out", default=str(DEFAULT_INDEX_PATH))
    p.add_argument("--combiner", choices=("linear", "softmax"), default="linear")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--margin", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=128.0)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--check", action="store_true", help="only report whether the index is stale")
    args = p.parse_args()

    if args.check:
        stale, reasons = index_is_stale(args.out)
        print("stale" if stale else "up to date")
        for reason in reasons:
            print(f"  - {reason}")
        return 0 if not stale else 1

    repo = args.repo or default_repo()
    if not repo:
        p.error("--repo (or $GOOGLE_FONTS_REPO) is required to build the index")

    index = build_index(
        repo,
        args.list,
        args.judgements,
        combiner=args.combiner,
        steps=args.steps,
        lr=args.lr,
        margin=args.margin,
        gamma=args.gamma,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device_name=args.device,
    )
    out = index.save(args.out)
    print(
        f"wrote {out} ({len(index)} families, "
        f"{int(index.rankable_mask.sum())} in the text arm, "
        f"{len(index.axis_names)} axes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
