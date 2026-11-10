"""Train the tag-prediction head on frozen gap features.

Two stages:

  1. Build (or load) the frozen gap-feature table from a trained similarity
     checkpoint — see :mod:`hrothgar.tagging.features`.  The table is cached so
     the (comparatively slow) rendering runs once.
  2. Train a small MLP head that regresses each tag's centile from the gap
     vector, masking unassessed cells, and report per-tag R² on a held-out
     *family* split.

Usage:
    python -m hrothgar.tagging.train \
        --dataset-path <google-fonts-checkout> \
        --encoder-model models/similarity.pth \
        --model-path models/tagging.pth \
        --features-cache output/tagging_features.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from hrothgar.similarity.config import FontStyleEmbedderConfig
from hrothgar.similarity.model import FontStyleEmbedder
from hrothgar.tagging.config import DEFAULT_TAG_NAMES, TaggingConfig
from hrothgar.tagging.dataset import TagDataset, family_train_val_split
from hrothgar.tagging.features import FeatureTable, build_feature_table
from hrothgar.tagging.losses import masked_loss
from hrothgar.tagging.model import TagPredictor


def _load_or_build(args, config: TaggingConfig, device: torch.device) -> FeatureTable:
    cache = Path(args.features_cache)
    if cache.exists() and not args.rebuild_features:
        print(f"loading cached features from {cache}")
        table = FeatureTable.load(cache)
        if table.tag_names != config.tag_names:
            raise ValueError(
                f"cached features were built for {table.tag_names}, "
                f"but config wants {config.tag_names}"
            )
        return table

    enc_cfg = FontStyleEmbedderConfig.from_sidecar(args.encoder_model)
    encoder_model = FontStyleEmbedder(enc_cfg)
    encoder_model.load(args.encoder_model, device)
    encoder_model.to(device)
    encoder_model.eval()

    table = build_feature_table(
        args.dataset_path, encoder_model, config, device, limit=args.limit
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    table.save(cache)
    print(f"cached features to {cache}")
    return table


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


@torch.no_grad()
def evaluate(
    model: TagPredictor, dataset: TagDataset, device: torch.device
) -> dict[str, float]:
    model.eval()
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    preds, targets, masks = [], [], []
    for feat, target, mask in loader:
        preds.append(model(feat.to(device)).cpu())
        targets.append(target)
        masks.append(mask)
    pred = torch.cat(preds).numpy() * 100.0
    tgt = torch.cat(targets).numpy() * 100.0
    msk = torch.cat(masks).numpy().astype(bool)

    r2s: dict[str, float] = {}
    for t, name in enumerate(dataset.table.tag_names):
        if msk[:, t].sum() < 5:
            r2s[name] = float("nan")
            continue
        r2s[name] = _r2(tgt[msk[:, t], t], pred[msk[:, t], t])
    return r2s


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        default=os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts"),
    )
    parser.add_argument("--encoder-model", default="models/similarity.pth")
    parser.add_argument("--features-cache", default="output/tagging_features.pt")
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--model-path", default="models/tagging.pth")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    # Head.
    parser.add_argument("--use-variance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--loss", choices=["mse", "huber"], default="huber")
    # Training.
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = TaggingConfig(
        tag_names=list(DEFAULT_TAG_NAMES),
        use_variance=args.use_variance,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        dropout=args.dropout,
        loss=args.loss,
    )

    table = _load_or_build(args, config, device)
    train_idx, val_idx = family_train_val_split(table, args.val_fraction, seed=args.seed)
    train_ds = TagDataset(table, train_idx)
    val_ds = TagDataset(table, val_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    print(
        f"cells: {table.n_cells}  train={len(train_ds)}  val={len(val_ds)}  "
        f"features={table.features.shape[1]}  tags={table.tag_names}"
    )

    model = TagPredictor(config, input_dim=table.features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    short = [t.split("/")[-1] for t in config.tag_names]
    best_mean = float("-inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for feat, target, mask in train_loader:
            feat, target, mask = feat.to(device), target.to(device), mask.to(device)
            pred = model(feat)
            loss = masked_loss(pred, target, mask, kind=config.loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        r2 = evaluate(model, val_ds, device)
        mean = float(np.nanmean(list(r2.values())))
        if mean > best_mean:
            best_mean = mean
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            cells = "  ".join(f"{n}={r2[t]:.3f}" for n, t in zip(short, config.tag_names))
            print(f"epoch {epoch:3d}  loss={loss.item():.4f}  mean R²={mean:+.3f}  {cells}")

    if best_state is not None:
        model.load_state_dict(best_state)
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_path)
    config.save_sidecar(args.model_path)
    print(f"saved best model (mean R²={best_mean:+.3f}) to {args.model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
