"""Probe whether the style tokens carry font/style identity.

Loads a trained ``StyleExtractionModelV2``, encodes a deterministic evidence set
for each held-out (test) font into style tokens, mean-pools them, and trains a
linear classifier to predict font family.

Interpretation
--------------
- ``train_acc`` near chance → the style tokens have collapsed to a generic
  average (representation bottleneck; raise ``num_style_tokens`` /
  ``glyph_encoder_feature_dim`` or fix the Perceiver).
- ``train_acc`` high but ``test_acc`` near chance → the representation is
  discriminative on seen fonts but does not generalise to held-out families
  (a generalisation problem, not a representation collapse).
- both high → the representation is fine; the bottleneck is downstream usage.

Methodological note
-------------------
A linear probe is only meaningful when every test class also appears in the
training split.  The original version sampled ~200 fonts across ~150 families
and then did a *random* 80/20 split, which put almost every test family outside
the training set — so the classifier was being asked to predict classes it had
never seen (accuracy ≈ chance with huge run-to-run variance, including a
misleadingly exact 0.0).  This version uses all held-out fonts, groups them by
family, keeps families with enough fonts, and does a *stratified* split so each
family contributes fonts to both train and test.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict

import torch
import torch.nn as nn

from hrothgar.style_extraction import load_model
from hrothgar.style_extraction.dataset import StyleExtractionDatasetMaker
from hrothgar.style_extraction.render_utils import render_glyph
from hrothgar.utils import torch_setup


def _deterministic_evidence(
    font,
    character_set_sorted: list[int],
    cp_to_idx: dict[int, int],
    num_glyphs: int,
    size: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Render a fixed, deterministic evidence set for one font.

    Returns ``(images (G, 1, size, size), codepoint_idx (G,))``, or ``None`` if
    the font has fewer than ``num_glyphs`` available codepoints.
    """
    avail = [cp for cp in character_set_sorted if cp in font.codepoints]
    if len(avail) < num_glyphs:
        return None

    glyphs = []
    idxs = []
    for cp in avail[:num_glyphs]:
        glyphs.append(render_glyph(font, cp, size))
        idxs.append(cp_to_idx[cp])

    images = torch.stack(glyphs).unsqueeze(1)  # (G, 1, size, size)
    codepoint_idx = torch.tensor(idxs, dtype=torch.long)
    return images, codepoint_idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument(
        "--min-fonts-per-family",
        type=int,
        default=2,
        help="Only probe families with at least this many fonts "
        "(so a stratified train/test split is possible).",
    )
    parser.add_argument(
        "--max-families",
        type=int,
        default=None,
        help="Optional cap on the number of families probed.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of each family's fonts held out for test.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Minibatch size for the linear probe.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    device = torch_setup()
    model, config = load_model(args.model_path, device)

    maker = StyleExtractionDatasetMaker(
        repo_url=args.dataset_path,
        batch_size=args.batch_size,
        image_size=config.image_size,
        character_set=config.character_set,
        num_evidence_glyphs=config.num_evidence_glyphs,
    )

    character_set_sorted = sorted(set(config.character_set))
    cp_to_idx = {cp: i for i, cp in enumerate(character_set_sorted)}
    size = config.image_size
    g = config.num_evidence_glyphs

    # One mean-pooled style vector per font, grouped by family.
    fam_to_feats: dict[str, list[torch.Tensor]] = defaultdict(list)
    for font in maker.test_fonts:
        ev = _deterministic_evidence(font, character_set_sorted, cp_to_idx, g, size)
        if ev is None:
            continue
        images, cp_idx = ev
        with torch.no_grad():
            tokens = model.encode_style(
                images.unsqueeze(0).to(device),
                style_codepoint_idx=cp_idx.unsqueeze(0).to(device),
            )
            feat = tokens.mean(dim=1).squeeze(0).cpu()  # (D,)
        fam_to_feats[font.family].append(feat)

    fams = sorted(
        f
        for f, feats in fam_to_feats.items()
        if len(feats) >= args.min_fonts_per_family
    )
    if args.max_families is not None:
        fams = fams[: args.max_families]

    if len(fams) < 2:
        raise SystemExit(
            "Too few families with enough fonts to probe. "
            f"Found {len(fams)} families with >= {args.min_fonts_per_family} fonts."
        )

    # Stratified split: each family contributes fonts to both train and test.
    rng = random.Random(args.seed)
    X_train: list[torch.Tensor] = []
    y_train: list[int] = []
    X_test: list[torch.Tensor] = []
    y_test: list[int] = []
    fam_idx: dict[str, int] = {}
    for fam in fams:
        feats = fam_to_feats[fam]
        rng.shuffle(feats)
        n_test = max(1, int(round(args.test_fraction * len(feats))))
        n_test = min(n_test, len(feats) - 1)  # keep >= 1 train font per family
        idx = len(fam_idx)
        fam_idx[fam] = idx
        for f in feats[:n_test]:
            X_test.append(f)
            y_test.append(idx)
        for f in feats[n_test:]:
            X_train.append(f)
            y_train.append(idx)

    X_train_t = torch.stack(X_train)
    X_test_t = torch.stack(X_test)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # Standardize using train statistics (features are mean-pooled token dims).
    mu = X_train_t.mean(dim=0)
    sd = X_train_t.std(dim=0).clamp_min(1e-5)
    X_train_t = (X_train_t - mu) / sd
    X_test_t = (X_test_t - mu) / sd

    num_classes = len(fam_idx)
    probe = nn.Linear(X_train_t.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    Xt = X_train_t.to(device)
    yt = y_train_t.to(device)
    n = len(Xt)
    gen = torch.Generator().manual_seed(args.seed)

    for _ in range(args.epochs):
        probe.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, args.batch_size):
            idx = perm[i : i + args.batch_size]
            opt.zero_grad()
            loss = loss_fn(probe(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        train_acc = (probe(Xt).argmax(dim=1) == yt).float().mean().item()
        test_acc = (
            (probe(X_test_t.to(device)).argmax(dim=1) == y_test_t.to(device))
            .float()
            .mean()
            .item()
        )

    per_family_sizes = [len(fam_to_feats[f]) for f in fams]
    chance = 1.0 / num_classes

    print(f"Font families probed: {num_classes}")
    print(f"Fonts:               {len(X_train_t)} train / {len(X_test_t)} test")
    print(
        f"Fonts per family:    min={min(per_family_sizes)}, "
        f"mean={sum(per_family_sizes) / len(per_family_sizes):.1f}, "
        f"max={max(per_family_sizes)}"
    )
    print(f"Linear probe train accuracy: {train_acc:.4f}")
    print(f"Linear probe test accuracy:  {test_acc:.4f}")
    print(f"Chance:                      {chance:.4f}")
    print(f"test x-chance:               {test_acc / chance:.1f}")

    if train_acc <= 5 * chance:
        print(
            "VERDICT: style tokens are near-generic — the representation is the bottleneck."
        )
    elif test_acc <= 5 * chance:
        print(
            "VERDICT: representation is discriminative on seen fonts but does not "
            "generalise to held-out families."
        )
    else:
        print(
            "VERDICT: style tokens are discriminative — downstream usage is the problem."
        )


if __name__ == "__main__":
    main()
