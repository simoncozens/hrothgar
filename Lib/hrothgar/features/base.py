"""Shared machinery for the glyph-image feature recogniser family.

A *feature recogniser* is a one-vs-rest classifier over a small set of closed
classes (e.g. ``a``-storey: single/double/uppercase) plus an "unrecognisable"
verdict read from a calibrated threshold on the max sigmoid.  Concrete
recognisers subclass :class:`FeatureRecognizer`, set their ``classes`` and
``glyphs``, and implement ``label_fn`` / ``build_real``.

The class provides four entry points the whole family shares:

* ``train``   — build the dataset, fit, calibrate the threshold, save weights + a
  ``.conf.json`` sidecar;
* ``evaluate``— load and report held-out accuracy and fontquant disagreement;
* ``from_checkpoint`` / ``load`` — reconstruct from disk (``SaveLoadModel``);
* ``recognise`` — take a TTF path (or a font object) and return a judgement.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
import uharfbuzz as hb
from torch import nn

from hrothgar.glyph_encoder import GlyphEncoder
from hrothgar.outline_tokens.a_storey_ood import _build_weird
from hrothgar.outline_tokens.oracle_baseline import _load_fontquant_raw
from hrothgar.outline_tokens.recognizer_audit import _crop_bottom, _crop_middle, _family_to_font, _load_cedars
from hrothgar.render import frame_for_codepoints, render
from hrothgar.utils import SaveLoadModel, git_short_sha

_BLANK = 0.995


@dataclass
class FeatureRecognizerConfig:
    """Common configuration for a glyph-image feature recogniser."""

    glyphs: str = "a"
    classes: list[str] = field(default_factory=list)
    glyph_size: int = 96
    base_channels: int = 16
    out_dim: int = 64
    downsample: int = 4
    # Spatial pooling grid: 1 = global max-pool (coarse tasks), >1 = coarse grid
    # preserving where a small feature sits (e.g. the e-crossbar).
    spatial_grid: int = 1
    # Regression mode: predict a continuous score in [0, score_max] instead of
    # classifying into ``classes``.  ``classes`` is then unused.
    regression: bool = False
    score_max: float = 100.0
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 64
    test_frac: float = 0.2
    reject_frac: float = 0.05
    seed: int = 0
    class_balance: bool = True
    # Crop the bottom ``crop_bottom_frac`` of each glyph image (and resize back
    # to ``glyph_size``) so a local feature that sits at the baseline — e.g. the
    # stem terminals — gets the full canvas rather than the whole glyph.
    crop_bottom_frac: float = 0.0
    # Crop the vertical band ``[lo, hi]`` (fractions of image height) of each glyph
    # and resize back, to focus a local feature that sits mid-glyph.
    crop_middle: tuple | None = None
    # Render variable-font ``wght``/``wdth`` extremes as extra training samples
    # (with the family's label).  Requires a family-level held-out split.
    expand_axes: bool = False

    def save_sidecar(self, model_path) -> None:
        data = asdict(self)
        data["git_sha"] = git_short_sha()
        sidecar = Path(str(model_path)).with_suffix(".conf.json")
        with sidecar.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")

    @classmethod
    def from_sidecar(cls, model_path):
        import dataclasses

        sidecar = Path(str(model_path)).with_suffix(".conf.json")
        with sidecar.open() as f:
            data = json.load(f)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# ── Rendering / utilities ─────────────────────────────────────────────────────
def _is_blank(img: torch.Tensor) -> bool:
    return bool((img.amin() > _BLANK).any())


def _all_caps_families(family_to_font: dict, fontquant: dict) -> set[str]:
    """All-caps/small-caps families (fontquant ``casing/lowercase_shapes``).

    Their lowercase glyph slots hold uppercase shapes, so a recogniser over a
    *lowercase* prompt (``lmn`` terminals/serifs) is out of domain.  We use
    fontquant's casing column rather than the ``a_storey`` recogniser: fontquant
    has one false negative (Eater) but ``a_storey`` has five false positives
    (Akronim, Jolly Lodger, Explora, Edu AU VIC WA NT Arrows, Shizuru — bi-case
    fonts it misreads as uppercase).
    """
    skip: set[str] = set()
    for fam, font in family_to_font.items():
        row = fontquant.get(Path(font.path).name.lower())
        if row is None:
            continue
        casing = (row.get("casing/lowercase_shapes", "") or "").strip()
        if casing in ("uppercase", "smallcaps"):
            skip.add(fam)
    return skip


def render_prompt(font, glyphs: str, size: int, crop_bottom_frac: float = 0.0, crop_middle=None) -> torch.Tensor | None:
    """Render a prompt glyph string into a shared tight frame; ``(G, 1, H, W)``.

    Returns ``None`` if any prompt glyph is missing from the font's cmap.
    """
    cps = [ord(c) for c in glyphs]
    hbfont = hb.Font(font.hb_face)
    if any(hbfont.get_nominal_glyph(cp) is None for cp in cps):
        return None
    frame = frame_for_codepoints(font, cps)
    imgs = [render(font, codepoint=cp, size=size, frame=frame)[0] for cp in cps]
    if crop_bottom_frac > 0.0:
        imgs = [_crop_bottom(img, crop_bottom_frac) for img in imgs]
    if crop_middle is not None:
        imgs = [_crop_middle(img, *crop_middle) for img in imgs]
    return torch.stack(imgs).unsqueeze(1)


def _pos_weights(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Per-head BCE ``pos_weight = neg/pos`` (balances class imbalance)."""
    pw = []
    for c in range(num_classes):
        pos = int((y == c).sum())
        neg = int((y != c).sum())
        pw.append(neg / pos if pos else 1.0)
    return torch.tensor(pw, dtype=torch.float32)


def _stratified_split(y: torch.Tensor, test_frac: float, seed: int) -> tuple[list[int], list[int]]:
    idx_by_class: dict[int, list[int]] = {}
    for i, c in enumerate(y.tolist()):
        idx_by_class.setdefault(c, []).append(i)
    rng = random.Random(seed)
    train, test = [], []
    for c, idxs in idx_by_class.items():
        rng.shuffle(idxs)
        n_test = max(1, int(round(test_frac * len(idxs)))) if len(idxs) > 1 else 0
        test.extend(idxs[:n_test])
        train.extend(idxs[n_test:])
    return train, test


def _family_stratified_split(
    y: torch.Tensor, families: list[str], test_frac: float, seed: int
) -> tuple[list[int], list[int]]:
    """Like :func:`_stratified_split`, but holds out whole families.

    Stratifies by class at the *family* level (every instance of a family shares
    its label), so no family's instances leak between train and test.
    """
    fam_idx: dict[str, list[int]] = {}
    fam_class: dict[str, int] = {}
    for i, fam in enumerate(families):
        fam_idx.setdefault(fam, []).append(i)
        fam_class[fam] = int(y[i])
    by_class: dict[int, list[str]] = {}
    for fam, c in fam_class.items():
        by_class.setdefault(c, []).append(fam)
    rng = random.Random(seed)
    train, test = [], []
    for _c, fams in by_class.items():
        rng.shuffle(fams)
        n_test = max(1, int(round(test_frac * len(fams)))) if len(fams) > 1 else 0
        for fam in fams[:n_test]:
            test.extend(fam_idx[fam])
        for fam in fams[n_test:]:
            train.extend(fam_idx[fam])
    return train, test


def _auc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    gt = (a[:, None] > b[None, :]).float().mean().item()
    eq = (a[:, None] == b[None, :]).float().mean().item()
    return gt + 0.5 * eq


def _rank_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation (Pearson on ranks)."""
    a = a.float()
    b = b.float()
    if a.numel() < 2 or b.numel() < 2:
        return float("nan")
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (ra.norm() * rb.norm()).clamp(min=1e-8)
    return float((ra * rb).sum() / denom)


# ── Base recogniser ───────────────────────────────────────────────────────────
class FeatureRecognizer(SaveLoadModel):
    """Glyph-image one-vs-rest feature recogniser (shared base)."""

    config_cls = FeatureRecognizerConfig

    def __init__(self, config: FeatureRecognizerConfig):
        super().__init__()
        self.config = config
        self.class_names = list(config.classes)
        C = len(config.classes)
        self.encoder = GlyphEncoder(config.base_channels, config.out_dim, config.downsample)
        feat_dim = config.out_dim * config.spatial_grid * config.spatial_grid
        if config.regression:
            self.head = nn.Linear(feat_dim, 1)
        else:
            self.head = nn.Linear(feat_dim, C)
            self.register_buffer("tau", torch.tensor(float("nan")))

    # ── model ────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is ``(B, G, 1, H, W)``; returns ``(B, C)`` logits."""
        b, g, _c, _h, _w = x.shape
        feats = self.encoder(x.flatten(0, 1))  # (B*G, out_dim, h, w)
        grid = self.config.spatial_grid
        pooled = F.adaptive_max_pool2d(feats, (grid, grid))  # (B*G, out_dim, grid, grid)
        pooled = pooled.reshape(b, g, self.config.out_dim, grid, grid).max(dim=1).values  # (B, out_dim, grid, grid)
        return self.head(pooled.reshape(b, -1))

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def verdict(self, scores: torch.Tensor) -> tuple[list[str], torch.Tensor]:
        """Map ``(N, C)`` sigmoid scores to verdicts + confidence (max sigmoid)."""
        mx, cls = scores.max(dim=1)
        out = []
        for i in range(scores.shape[0]):
            out.append("unrecognisable" if mx[i].item() < self.tau.item() else self.class_names[int(cls[i].item())])
        return out, mx

    # ── the recognising use case ─────────────────────────────────────────────
    def recognise_font(self, font, device=None) -> dict:
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        img = render_prompt(font, self.config.glyphs, self.config.glyph_size, self.config.crop_bottom_frac, self.config.crop_middle)
        if img is None or _is_blank(img):
            if self.config.regression:
                return {"display_score": 0.0}
            return {"verdict": "blank", "scores": {c: 0.0 for c in self.class_names}, "confidence": 0.0}
        self.eval()
        with torch.no_grad():
            out = self.forward(img.unsqueeze(0).to(device))[0].cpu()  # (C,) or (1,)
        if self.config.regression:
            score = float(torch.sigmoid(out[0]).item() * self.config.score_max)
            return {"display_score": score}
        s = torch.sigmoid(out)
        verdicts, conf = self.verdict(s.unsqueeze(0))
        return {
            "verdict": verdicts[0],
            "scores": {c: float(s[i]) for i, c in enumerate(self.class_names)},
            "confidence": float(conf[0]),
        }

    def recognise(self, ttf_path: str | Path, device=None) -> dict:
        """Take a TTF file and return a judgement."""
        from hrothgar.googlefonts import StandaloneFont

        return self.recognise_font(StandaloneFont(ttf_path), device=device)

    # ── training / calibration ───────────────────────────────────────────────
    def fit(self, X: torch.Tensor, y: torch.Tensor, Xw: torch.Tensor, *, device=None, families=None) -> dict:
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.config.regression:
            return self._fit_regression(X, y, device)
        return self._fit_classification(X, y, Xw, device, families)

    def _fit_classification(self, X, y, Xw, device, families=None) -> dict:
        """Train the OVR heads on real + weird negatives and calibrate ``tau``."""
        cfg = self.config
        C = len(self.class_names)
        if families is not None and len(families) == len(y):
            train_idx, calib_idx = _family_stratified_split(y, families, cfg.test_frac, cfg.seed)
        else:
            train_idx, calib_idx = _stratified_split(y, cfg.test_frac, cfg.seed)

        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        self.to(device)
        opt = torch.optim.Adam(self.parameters(), lr=cfg.lr)
        pw = _pos_weights(y[train_idx], C).to(device)

        Xd, yd, Xwd = X.to(device), y.to(device), Xw.to(device)
        train_t = torch.tensor(train_idx, device=device)

        class_lists = [train_t[yd[train_t] == c] for c in range(C)]
        n_target = max(len(l) for l in class_lists if len(l) > 0)

        def make_balanced() -> torch.Tensor:
            parts = [l[torch.randint(0, len(l), (n_target,), device=device)] for l in class_lists if len(l) > 0]
            return torch.cat(parts)

        for _ in range(cfg.epochs):
            self.train()
            idx = make_balanced() if cfg.class_balance else train_t
            perm = torch.randperm(len(idx), device=device)
            for start in range(0, len(perm), cfg.batch_size):
                batch = idx[perm[start : start + cfg.batch_size]]
                logits = self(Xd[batch])
                t = torch.zeros(len(batch), C, device=device)
                t[torch.arange(len(batch), device=device), yd[batch]] = 1.0
                loss = F.binary_cross_entropy_with_logits(logits, t, pos_weight=pw)
                if Xwd.numel():
                    loss = loss + F.binary_cross_entropy_with_logits(self(Xwd), torch.zeros(Xwd.shape[0], C, device=device))
                opt.zero_grad()
                loss.backward()
                opt.step()

        self.eval()
        with torch.no_grad():
            calib_scores = self.scores(Xd[torch.tensor(calib_idx)])
            calib_max = calib_scores.max(dim=1).values
            calib_pred = calib_scores.argmax(dim=1).cpu()
            calib_y = yd[torch.tensor(calib_idx)].cpu()
        self.tau.copy_(torch.quantile(calib_max, cfg.reject_frac))

        per_class = {}
        for c, name in enumerate(self.class_names):
            mask = calib_y == c
            n_c = int(mask.sum())
            if n_c:
                per_class[name] = int(((calib_pred == c) & mask).sum()) / n_c
        return {
            "tau": float(self.tau),
            "accuracy": float((calib_pred == calib_y).float().mean()),
            "per_class": per_class,
        }

    def _fit_regression(self, X, y, device) -> dict:
        """Train a scalar score regressor (MSE on a sigmoid-scaled [0, score_max] target)."""
        cfg = self.config
        n = X.shape[0]
        rng = random.Random(cfg.seed)
        idx = list(range(n))
        rng.shuffle(idx)
        n_calib = max(1, int(round(cfg.test_frac * n)))
        calib_idx = idx[:n_calib]
        train_idx = idx[n_calib:]

        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        self.to(device)
        opt = torch.optim.Adam(self.parameters(), lr=cfg.lr)
        Xd = X.to(device)
        yd = (y.to(device) / cfg.score_max)  # [0, 1]
        train_t = torch.tensor(train_idx, device=device)

        for _ in range(cfg.epochs):
            self.train()
            perm = torch.randperm(len(train_t), device=device)
            for start in range(0, len(perm), cfg.batch_size):
                batch = train_t[perm[start : start + cfg.batch_size]]
                pred = torch.sigmoid(self(Xd[batch])).squeeze(-1)  # (B,)
                loss = F.mse_loss(pred, yd[batch])
                opt.zero_grad()
                loss.backward()
                opt.step()

        self.eval()
        with torch.no_grad():
            calib_t = torch.tensor(calib_idx, device=device)
            pred = (torch.sigmoid(self(Xd[calib_t])).squeeze(-1) * cfg.score_max).cpu()
            y_calib = (yd[calib_t] * cfg.score_max).cpu()
        return {
            "mae": float((pred - y_calib).abs().mean()),
            "mse": float(((pred - y_calib) ** 2).mean()),
            "rank_corr": _rank_corr(pred, y_calib),
        }

    # ── per-model hooks (subclasses implement these) ──────────────────────────
    @staticmethod
    def label_fn(fq: dict | None, cedars_row: dict | None) -> str | None:
        """Return the ground-truth label for a font, or ``None`` if unassessed."""
        raise NotImplementedError

    @classmethod
    def excluded_families(cls, family_to_font: dict, cedars: dict, fontquant: dict) -> set:
        """Families outside this recogniser's domain (skipped in eval).

        Default: no exclusions.  Subclasses override this to drop fonts whose
        prompt glyph is not actually the feature being recognised (e.g. the
        ``stem-terminals`` recogniser excludes all-caps/small-caps fonts, whose
        ``lmn`` slots hold uppercase shapes).
        """
        return set()

    def build_real(self, repo: str):
        """Build ``(X, y)`` for the real (closed-set) glyph data."""
        raise NotImplementedError

    # ── entry points ─────────────────────────────────────────────────────────
    @classmethod
    def train_from_repo(cls, repo: str, model_path: str, config=None, device=None):
        config = config or cls.config_cls()
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        model = cls(config)
        data = model.build_real(repo)
        X, y = data[0], data[1]
        families = data[3] if len(data) >= 4 and isinstance(data[3], list) else None
        if config.regression:
            Xw = torch.empty(0)
        else:
            Xw, _ = _build_weird(repo, config.glyph_size, crop_bottom_frac=config.crop_bottom_frac, crop_middle=config.crop_middle)
        stats = model.fit(X, y, Xw, device=device, families=families)
        model.save(model_path)
        print(f"trained {cls.__name__} -> {model_path}")
        if config.regression:
            print(f"  mae={stats['mae']:.2f}  mse={stats['mse']:.2f}  rank_corr={stats['rank_corr']:.3f}")
        else:
            print(f"  tau={stats['tau']:.3f}  accuracy={stats['accuracy']:.3f}  per-class={stats['per_class']}")
        return model

    @classmethod
    def evaluate(cls, repo: str, model_path: str, device=None) -> dict:
        model = cls.from_checkpoint(model_path, device=device)
        if model.config.regression:
            return model._evaluate_regression(repo, device=device)
        fontquant = _load_fontquant_raw()
        cedars = _load_cedars()
        family_to_font = _family_to_font(repo)
        excluded = cls.excluded_families(family_to_font, cedars, fontquant)

        agree = mismatch = unrecognised = missing = blank = excluded_n = 0
        mismatches: list[tuple[str, str, str, float]] = []
        for fam in sorted(family_to_font):
            if fam in excluded:
                excluded_n += 1
                continue
            font = family_to_font[fam]
            fq = fontquant.get(Path(font.path).name.lower())
            cq = cedars.get(fam)
            label = cls.label_fn(fq, cq)
            res = model.recognise_font(font, device=device)
            verdict = res["verdict"]
            if verdict == "blank":
                blank += 1
            elif label is None:
                missing += 1
            elif verdict == "unrecognisable":
                unrecognised += 1
            elif verdict != label:
                mismatch += 1
                mismatches.append((font.family, label, verdict, res["confidence"]))
            else:
                agree += 1

        report = {"agree": agree, "mismatch": mismatch, "unrecognised": unrecognised, "missing": missing, "blank": blank, "excluded": excluded_n}
        print(f"fonts={len(family_to_font)}  {report}")
        print(f"\n── {len(mismatches)} mismatches (label vs recogniser) ──")
        for fam, label, verdict, conf in sorted(mismatches, key=lambda r: -r[3]):
            print(f"  {fam:<28s} label={label:<10s} recogniser={verdict:<14s} conf={conf:.3f}")
        return report

    def _evaluate_regression(self, repo: str, device=None) -> dict:
        from hrothgar.googlefonts import compute_display_score

        family_to_font = _family_to_font(repo)
        names, preds, targets = [], [], []
        for fam in sorted(family_to_font):
            font = family_to_font[fam]
            res = self.recognise_font(font, device=device)
            names.append(font.family)
            preds.append(res["display_score"])
            targets.append(compute_display_score(font.tags()))
        preds_t = torch.tensor(preds)
        targets_t = torch.tensor(targets)
        rho = _rank_corr(preds_t, targets_t)
        mae = float((preds_t - targets_t).abs().mean())
        print(f"display regression: n={len(names)}  rank_corr={rho:.3f}  mae={mae:.2f}")
        # Worst under/over-predictions for eyeballing.
        err = preds_t - targets_t
        order = err.argsort(descending=True)
        print("  most over-predicted (model says display, teacher says text):")
        for i in order[:10].tolist():
            print(f"    {names[i]:<28s} pred={preds_t[i]:5.1f} teacher={targets_t[i]:5.1f}")
        print("  most under-predicted (model says text, teacher says display):")
        for i in order[-10:].tolist():
            print(f"    {names[i]:<28s} pred={preds_t[i]:5.1f} teacher={targets_t[i]:5.1f}")
        return {"rank_corr": rho, "mae": mae}

    # ── load/save ────────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        super().save(path)
        self.config.save_sidecar(path)

    @classmethod
    def from_checkpoint(cls, path: str, device=None):
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        config = cls.config_cls.from_sidecar(path)
        model = cls(config).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model
