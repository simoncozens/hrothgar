"""Visual encoder: a rendered specimen phrase -> 30-dim style coordinates.

This is the "student" that distils the frozen axis-weighted teacher (the 30-dim
PCA coordinates saved by ``similarity.embedding --mode teacher``) from pixels.

Architecture: a shared ``GlyphEncoder`` produces a spatial feature map (the
"bigger latent"), from which two heads branch — a regression head that pools to
the 30 style coordinates, and a decoder that reconstructs the phrase image (a
regulariser that forces the encoder to actually read the glyphs).  Interpolating
the *appearance* latent is the decoder's domain; the 30-dim style coords remain a
separate, fixed read.

The specimen is ``PHRASE`` (``hamburgefontsivclHAS``): the classic lowercase
``hamburgefontsiv`` — covering the glyphs the teacher axes depend on (a c e g l m
n o s t) — plus ``cl`` and a short uppercase tail.  Uppercase was trimmed because
it is less style-determinative per pixel and a shorter phrase lets every
lowercase glyph render larger.  Fonts are rendered at a *fixed* size (swept to
fit ~95% of fonts) so relative width survives; the ~5% outliers are clipped by
Skia.

Two measurement notes that matter more than they look:

* The teacher's coordinates are PCA scores, so their variance falls steeply with
  index (PC1 holds ~23% of the variance, PC30 ~1%).  Judge the coordinate head by
  ``R^2`` (variance explained), not raw MSE — an MSE that looks small in absolute
  terms can still be most of the variance.  (The encoder's reconstruction is
  ``excellent`` even when the 30-dim read is near the mean, because the decoder
  sees the *spatial* map while the coord head only sees the globally-pooled
  vector.)
* ``--coord-weighting`` optionally up-weights the low-variance PCs
  (``standardize`` = 1/std, ``inverse-variance`` = 1/var, both measured on the
  training split only).  This changes the geometry the model optimises to a
  whitened one.  ``coords = (X-mean) @ Vt.T`` with orthonormal ``Vt``, so raw
  Euclidean distance in coord space *is* Euclidean distance in the teacher's
  weighted feature space; whitening is a different metric.  Retrieval is
  therefore reported in both spaces (``_nn_report``).

Usage::

    python -m hrothgar.similarity.visual --mode train --teacher models/text_teacher.npz
    python -m hrothgar.similarity.visual --mode evaluate --model-path models/visual.pth
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path

from glyphloss import glyph_reconstruction_loss
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from tqdm import tqdm

from hrothgar.glyph_encoder import GlyphEncoder
from hrothgar.outline_tokens.harness import _device
from hrothgar.outline_tokens.recognizer_audit import _family_to_font
from hrothgar.render import render_phrase

PHRASE = "hamburgefontsivclHAS"
COORD_DIM = 30


def _parse_grid(spec: str) -> tuple[int, int]:
    """Parse an ``"HxW"`` grid specification."""
    try:
        h, w = spec.lower().split("x")
        return int(h), int(w)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected 'HxW', got {spec!r}") from exc


class VisualEncoder(nn.Module):
    """Shared GlyphEncoder -> spatial latent -> (coord head, decoder).

    ``coord_head`` selects how the spatial feature map is read into coordinates:

    * ``gap_linear`` — the original: global average pool then one ``nn.Linear``.
      This is algebraically ``GAP(Conv1x1(F))``, i.e. each coordinate is the
      phrase-wide *mean* of a single feature detector.  It can answer "is this
      feature present?" but not "how much, where, or for which glyph", and a
      single linear layer cannot form ratios (x-height / ascender, contrast).
    * ``gap_mlp`` — global average pool then a two-layer MLP.  Adds the
      nonlinearity needed for ratio features while keeping the global read.
    * ``grid_mlp`` — 1x1 projection then adaptive pooling to a coarse ``(H, W)``
      grid, flattened into a two-layer MLP.  Retains coarse *position*, so the
      teacher's per-glyph features (its 26-vector proportions) are no longer
      averaged together across the phrase.

    ``pooled_recon_grid`` optionally adds a decoder from the *same pooled vector
    the coordinate head reads* back to a coarse image.  The main ``decoder`` sees
    the full spatial map and can therefore reconstruct perfectly without the
    pooled vector carrying any style at all; this auxiliary is what actually
    forces shared shape information into the coordinate read-out.
    """

    def __init__(
        self,
        out_dim: int = 128,
        downsample: int = 4,
        coord_dim: int = COORD_DIM,
        coord_head: str = "gap_linear",
        head_hidden: int = 256,
        head_channels: int = 16,
        head_grid: tuple[int, int] = (4, 32),
        pooled_recon_grid: tuple[int, int] | None = None,
    ):
        super().__init__()
        self.encoder = GlyphEncoder(base_channels=32, out_dim=out_dim, downsample=downsample)
        self.coord_head_kind = coord_head
        self.head_grid = tuple(head_grid)
        self.pooled_recon_grid = None if pooled_recon_grid is None else tuple(pooled_recon_grid)

        self.spatial_proj: nn.Conv2d | None = None
        if coord_head == "grid_mlp":
            self.spatial_proj = nn.Conv2d(out_dim, head_channels, 1, bias=False)
            pooled_dim = head_channels * head_grid[0] * head_grid[1]
            self.coord_head = nn.Sequential(
                nn.Linear(pooled_dim, head_hidden), nn.SiLU(), nn.Linear(head_hidden, coord_dim)
            )
        elif coord_head == "gap_mlp":
            pooled_dim = out_dim
            self.coord_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                nn.Linear(out_dim, head_hidden), nn.SiLU(), nn.Linear(head_hidden, coord_dim),
            )
        elif coord_head == "gap_linear":
            pooled_dim = out_dim
            # Kept identical to the original head so older checkpoints still load.
            self.coord_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(out_dim, coord_dim)
            )
        else:
            raise ValueError(f"unknown coord_head {coord_head!r}")

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.pooled_recon: nn.Sequential | None = None
        if self.pooled_recon_grid is not None:
            gh, gw = self.pooled_recon_grid
            self.pooled_recon = nn.Sequential(
                nn.Linear(pooled_dim, head_hidden), nn.SiLU(), nn.Linear(head_hidden, gh * gw)
            )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(out_dim, 128, 3, padding=1),
            nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 1, 3, padding=1),
        )

    def pooled_features(self, feats: torch.Tensor) -> torch.Tensor:
        """The vector the coordinate head (and pooled-recon head) consumes."""
        if self.spatial_proj is not None:
            return F.adaptive_avg_pool2d(self.spatial_proj(feats), self.head_grid).flatten(1)
        return self.gap(feats).flatten(1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        feats = self.encoder(x)  # (B, out_dim, H/ds, W/ds)
        if self.spatial_proj is not None:
            pooled = self.pooled_features(feats)
            coords = self.coord_head(pooled)  # (B, 30)
        else:
            coords = self.coord_head(feats)  # (B, 30)
            pooled = self.pooled_features(feats) if self.pooled_recon is not None else None
        recon = torch.sigmoid(self.decoder(feats))  # (B, 1, H, W)
        recon_coarse = None
        if self.pooled_recon is not None and pooled is not None:
            gh, gw = self.pooled_recon_grid
            recon_coarse = torch.sigmoid(self.pooled_recon(pooled)).view(-1, 1, gh, gw)
        return coords, recon, recon_coarse


def render_phrase_img(font, size: int, width: int, height: int) -> np.ndarray:
    """Render the specimen phrase to a ``(H, W)`` float image (ink~0, white~1)."""
    arr = render_phrase(str(font.path), PHRASE, size=size, width=width, height=height)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32) / 255.0


def _fits(img: np.ndarray) -> bool:
    """True if the phrase does not touch the right edge (not clipped)."""
    return not bool((img[:, -2:] < 0.5).any())


def sweep_size(fonts, width: int, height: int, target_frac: float = 0.95) -> int:
    """Largest font size for which ~``target_frac`` of fonts fit in the canvas."""
    sample = fonts[:: max(1, len(fonts) // 200)]
    sizes = list(range(16, 128, 4))
    best = sizes[0]
    print(f"sweeping size over {len(sample)} fonts (width={width} height={height})")
    for size in sizes:
        n_fit = sum(1 for f in sample if _fits(render_phrase_img(f, size, width, height)))
        frac = n_fit / len(sample)
        print(f"  size={size:>2}  fit={frac:.2f}")
        if frac >= target_frac:
            best = size
        else:
            break
    print(f"  -> chosen size {best}")
    return best


def _render_dataset(fonts, size: int, width: int, height: int, input_h: int, input_w: int):
    """Render every font, resize to the model input; return ``(imgs, families)``."""
    imgs = []
    for f in tqdm(fonts, desc="rendering phrases"):
        img = render_phrase_img(f, size, width, height)
        t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        t = F.interpolate(t, size=(input_h, input_w), mode="bilinear", align_corners=False)
        imgs.append(t.squeeze(0))
    return torch.stack(imgs), [f.family for f in fonts]


def _family_split(families: list[str], test_frac: float, seed: int):
    import random
    uniq = sorted(set(families))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n_test = max(1, int(round(test_frac * len(uniq))))
    test_fams = set(uniq[:n_test])
    train, test = [], []
    for i, fam in enumerate(families):
        (test if fam in test_fams else train).append(i)
    return train, test


def _nn_overlap(coords_a: torch.Tensor, coords_b: torch.Tensor, k: int = 10) -> float:
    da = torch.cdist(coords_a, coords_a)
    db = torch.cdist(coords_b, coords_b)
    n = coords_a.shape[0]
    oa = da.argsort(dim=1)
    ob = db.argsort(dim=1)
    fracs = []
    for i in range(n):
        s = set(oa[i, 1 : k + 1].tolist())
        e = set(ob[i, 1 : k + 1].tolist())
        fracs.append(len(s & e) / k)
    return float(torch.tensor(fracs).mean())


def _coord_weights(train_coords: torch.Tensor, mode: str) -> torch.Tensor | None:
    """Per-dimension loss weights, derived from the *training* split only.

    The teacher's 30 coordinates are PCA scores, so their variance falls steeply
    with index.  Unweighted MSE spends almost all of its gradient on the first
    few PCs and leaves the tail dims at ``mse/var ~= 1`` (the model just predicts
    the mean).  ``standardize`` uses ``1/std`` (z-scoring the targets);
    ``inverse-variance`` uses ``1/var``, the same idea more aggressively.

    Weights are rescaled to mean 1 so the loss keeps the same scale as plain MSE.
    """
    if mode == "none":
        return None
    var = train_coords.var(dim=0, unbiased=False).clamp_min(1e-8)
    w = 1.0 / (var if mode == "inverse-variance" else var.sqrt())
    return w * (w.numel() / w.sum())


def _coord_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    err = (pred - target) ** 2
    return err.mean() if weights is None else (err * weights).mean()


def _variance_explained(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    """``(raw MSE, R^2 vs a mean predictor)`` over unweighted coordinates."""
    mse = F.mse_loss(pred, target).item()
    var = target.var(dim=0, unbiased=False).mean().item()
    return mse, 1.0 - mse / max(var, 1e-12)


def _nn_report(pred: torch.Tensor, target: torch.Tensor, inv_std: torch.Tensor | None = None) -> None:
    """Print nearest-neighbour overlap in raw (teacher) and whitened space."""
    for k in (1, 5, 10):
        line = f"  NN overlap@{k:<2} raw: {_nn_overlap(pred, target, k=k):.3f}"
        if inv_std is not None:
            line += f"   whitened: {_nn_overlap(pred * inv_std, target * inv_std, k=k):.3f}"
        print(line)


def _sidecar_path(model_path: str) -> Path:
    return Path(str(Path(model_path).with_suffix("")) + ".conf.json")


def _save_sidecar(model_path: str, cfg: dict) -> None:
    _sidecar_path(model_path).write_text(json.dumps(cfg, indent=2) + "\n")


def _load_sidecar(model_path: str) -> dict:
    p = _sidecar_path(model_path)
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("train", "evaluate"), required=True)
    p.add_argument("--repo", default=os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts"))
    p.add_argument("--teacher", default="models/text_teacher.npz")
    p.add_argument("--model-path", default="models/visual.pth")
    p.add_argument("--size", type=int, default=0, help="font size (0 = sweep)")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--input-h", type=int, default=64)
    p.add_argument("--input-w", type=int, default=512)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--recon-weight", type=float, default=0)
    p.add_argument(
        "--coord-weighting",
        choices=("none", "standardize", "inverse-variance"),
        default="none",
        help="weight the coord loss by 1/std or 1/var of the training coords",
    )
    p.add_argument("--out-dim", type=int, default=128)
    p.add_argument("--downsample", type=int, default=4, choices=(2, 4, 8))
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    dev = _device(args.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Teacher + fonts ────────────────────────────────────────────────────────
    t = np.load(args.teacher, allow_pickle=True)
    coords = torch.from_numpy(t["coords"].astype(np.float32))  # (N, 30)
    fams = [str(x) for x in t["families"]]
    fam2font = _family_to_font(args.repo)
    fonts = [fam2font[fam.lower()] for fam in fams]
    print(f"teacher: {coords.shape[0]} fonts, {coords.shape[1]} coords")

    if args.mode == "train":
        size = args.size if args.size > 0 else sweep_size(fonts, args.width, args.height)
        X, families = _render_dataset(fonts, size, args.width, args.height, args.input_h, args.input_w)
        print(f"dataset: {X.shape} (rendered at size={size})")
        train_idx, test_idx = _family_split(families, args.test_frac, args.seed)
        print(f"split: train={len(train_idx)} test={len(test_idx)}")

        # Coord statistics come from the training split only, never the held-out one.
        weights = _coord_weights(coords[train_idx], args.coord_weighting)
        if weights is not None:
            weights = weights.to(dev)
            print(f"coord weighting: {args.coord_weighting}")
        train_std = coords[train_idx].std(dim=0, unbiased=False).clamp_min(1e-8)
        inv_std = (1.0 / train_std).to(dev)

        from torch.utils.tensorboard import SummaryWriter

        run_id = f"logs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        writer = SummaryWriter(log_dir=run_id)
        model = VisualEncoder(out_dim=args.out_dim, downsample=args.downsample).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        Xd, cd = X.to(dev), coords.to(dev)
        train_t = torch.tensor(train_idx, device=dev)
        global_step = 0
        best_test = 1e6

        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(len(train_t), device=dev)
            tot_coord = tot_raw = tot_recon = 0.0
            nb = 0
            for start in range(0, len(perm), args.batch_size):
                batch = train_t[perm[start : start + args.batch_size]]
                x = Xd[batch]
                coord_pred, recon, recon_course = model(x)
                loss_coord = _coord_loss(coord_pred, cd[batch], weights)
                loss_coord_raw = F.mse_loss(coord_pred, cd[batch])
                # loss_recon = glyph_reconstruction_loss(recon, x)
                writer.add_scalar("Losses/coord", loss_coord.detach().cpu(), global_step)
                writer.add_scalar("Losses/coord_raw", loss_coord_raw.detach().cpu(), global_step)
                # writer.add_scalar("Losses/recon", loss_recon.detach().cpu(), global_step)
                loss = loss_coord #  + args.recon_weight * loss_recon
                writer.add_scalar("Losses/total", loss.detach().cpu(), global_step)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot_coord += loss_coord.item()
                tot_raw += loss_coord_raw.item()
                # tot_recon += loss_recon.item()
                nb += 1
                global_step += 1
            if epoch % 5 == 0 or epoch == args.epochs - 1:
                print(
                    f"  epoch {epoch:>3}  coord_loss={tot_coord/nb:.4f}"
                    f"  coord_mse={tot_raw/nb:.4f}  " # recon_glyphloss={tot_recon/nb:.4f}"
                )
                model.eval()
                with torch.no_grad():
                    coord_pred, recon_v, _ = model(Xd[torch.tensor(test_idx, device=dev)])
                # grid = torch.cat([Xd[test_idx[0]].unsqueeze(0), recon_v[0:1]], dim=0)
                # writer.add_image(
                #             "Validation/gt_ref_recon",
                #             torchvision.utils.make_grid(grid, nrow=2),
                #             global_step,
                #         )

                pred_cpu, target_cpu = coord_pred.cpu(), coords[test_idx]
                test_mse, test_r2 = _variance_explained(pred_cpu, target_cpu)
                print(f"held-out coord MSE: {test_mse:.4f}   R^2: {test_r2:.3f}")
                model.train()
                writer.add_scalar("Validation/coord", test_mse, global_step)
                writer.add_scalar("Validation/r2", test_r2, global_step)
                if weights is not None:
                    test_wmse = _coord_loss(pred_cpu, target_cpu, weights.cpu()).item()
                    writer.add_scalar("Validation/coord_weighted", test_wmse, global_step)
                    print(f"held-out weighted coord loss: {test_wmse:.4f}")
                _nn_report(pred_cpu, target_cpu, inv_std.cpu())
                if test_mse < best_test:
                    torch.save(model.state_dict(), args.model_path)
                    _save_sidecar(args.model_path, {
                        "phrase": PHRASE,
                        "size": size,
                        "width": args.width,
                        "height": args.height,
                        "input_h": args.input_h,
                        "input_w": args.input_w,
                        "out_dim": args.out_dim,
                        "downsample": args.downsample,
                        "coord_dim": COORD_DIM,
                        "coord_weighting": args.coord_weighting,
                        "test_frac": args.test_frac,
                        "seed": args.seed,
                        "epoch": epoch,
                        "held_out_mse": test_mse,
                        "held_out_r2": test_r2,
                        "train_coord_var": coords[train_idx].var(dim=0, unbiased=False).tolist(),
                    })
                    best_test = test_mse
                    print(f"saved -> {args.model_path}")

    else:  # evaluate
        if not Path(args.model_path).exists():
            print(f"model {args.model_path} not found")
            return 1
        # Reproduce the training render/split/architecture from the sidecar; a
        # mismatch here silently evaluates the model out of distribution.
        cfg = _load_sidecar(args.model_path)
        size = cfg.get("size") or (args.size if args.size > 0 else sweep_size(fonts, args.width, args.height))
        width = cfg.get("width", args.width)
        height = cfg.get("height", args.height)
        input_h = cfg.get("input_h", args.input_h)
        input_w = cfg.get("input_w", args.input_w)
        out_dim = cfg.get("out_dim", args.out_dim)
        downsample = cfg.get("downsample", args.downsample)
        test_frac = cfg.get("test_frac", args.test_frac)
        seed = cfg.get("seed", args.seed)
        if not cfg:
            print("warning: no .conf.json sidecar; using CLI/defaults (may not match training)")
        if args.size and cfg.get("size") and args.size != cfg["size"]:
            print(f"warning: --size {args.size} ignored; sidecar says size {cfg['size']}")
        print(
            f"eval config: phrase={PHRASE!r} size={size} canvas={width}x{height} "
            f"input={input_h}x{input_w} out_dim={out_dim} ds={downsample} "
            f"test_frac={test_frac} seed={seed} weighting={cfg.get('coord_weighting', 'none')}"
        )
        model = VisualEncoder(out_dim=out_dim, downsample=downsample).to(dev)
        model.load_state_dict(torch.load(args.model_path, map_location=dev))
        model.eval()
        X, families = _render_dataset(fonts, size, width, height, input_h, input_w)
        _, test_idx = _family_split(families, test_frac, seed)
        X = X.to(dev)
        with torch.no_grad():
            coord_pred, _ = model(X[test_idx])
        pred = coord_pred.cpu()
        target = coords[torch.tensor(test_idx)]
        print(f"held-out: {len(test_idx)} fonts")
        mse, r2 = _variance_explained(pred, target)
        print(f"  coord MSE: {mse:.4f}   R^2: {r2:.3f}")
        # Per-dimension variance explained: ~0 means the model predicts the mean
        # for that PC, ~1 means it reproduces it.  This is the honest readout.
        var = target.var(dim=0, unbiased=False)
        err = (pred - target) ** 2
        print(
            "  per-dim variance explained: "
            + " ".join(f"{1.0 - err[:, i].mean().item() / max(var[i].item(), 1e-9):+.2f}" for i in range(COORD_DIM))
        )
        train_std = (
            torch.tensor(cfg["train_coord_var"]).sqrt().clamp_min(1e-8)
            if "train_coord_var" in cfg
            else target.std(dim=0, unbiased=False).clamp_min(1e-8)
        )
        _nn_report(pred, target, inv_std=1.0 / train_std)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
