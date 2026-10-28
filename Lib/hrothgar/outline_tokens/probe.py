"""Leave-one-design-out probe for the contour-token style classifier.

Trains a fresh :class:`SetEncoder` per held-out design and reports design-level
generalisation: the model must classify a *design it has never seen* by its fine
style, from the ``e`` contour tokens alone.  Chance is the majority-class
proportion; ``--permute`` re-runs with shuffled labels as a memorisation control
(the real accuracy should sit clearly above the permuted accuracy).

``--folds K`` switches to stratified K-fold design cross-validation (much faster,
trains K models instead of one per design); omit it for the full gold-standard
leave-one-design-out loop.

Usage::

    python -m hrothgar.outline_tokens.probe --repo ~/others-repos/fonts
    python -m hrothgar.outline_tokens.probe --repo ~/others-repos/fonts --folds 5
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

import torch
import uharfbuzz as hb
from torch import nn

from hrothgar.googlefonts import GoogleFonts
from hrothgar.outline_tokens import glyph_tokens
from hrothgar.outline_tokens.model import SetEncoder
from hrothgar.outline_tokens.selection import CLASSES, select_designs

# Cap per-glyph token count.  The median "e" has ~14 on-curve points; only
# three display-novelty designs (Frijole 228, Raleway Dots 187, Sigmar One 70)
# exceed 64, and their extra points are decorative (dots/ornament), not the
# fine-style signal we care about.  Truncating also keeps the padded batch
# small instead of padding every glyph to the single 228-point outlier.
MAX_POINTS = 64


def _tokens_for(font, glyph_char: str) -> list[list[float]]:
    gid = hb.Font(font.hb_face).get_nominal_glyph(ord(glyph_char))
    return glyph_tokens(font.hb_face, gid)


def _pad(tokens_list: list[list[list[float]]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of glyph token-sets to ``(B, L, D)`` + a bool mask ``(B, L)``."""
    d = len(tokens_list[0][0])
    l = max(len(t) for t in tokens_list)
    b = len(tokens_list)
    x = torch.zeros(b, l, d, dtype=torch.float32)
    mask = torch.zeros(b, l, dtype=torch.bool)
    for i, tokens in enumerate(tokens_list):
        x[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.float32)
        mask[i, : len(tokens)] = True
    return x, mask


def _train(
    model: nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    y: torch.Tensor,
    steps: int,
    lr: float,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(model(x, mask), y)
        loss.backward()
        opt.step()


def _predict(model: nn.Module, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(x, mask).argmax(dim=1)


def _lofo_preds(x, mask, y, hidden, steps, lr, seed) -> torch.Tensor:
    n = x.shape[0]
    d = x.shape[2]
    c = len(CLASSES)
    preds = torch.empty(n, dtype=torch.long)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        model = SetEncoder(in_dim=d, hidden=hidden, num_classes=c)
        _train(model, x[tr], mask[tr], y[tr], steps, lr, seed)
        preds[i] = _predict(model, x[i : i + 1], mask[i : i + 1])
        if (i + 1) % 50 == 0:
            print(f"  fold {i + 1}/{n}", flush=True)
    return preds


def _fold_preds(x, mask, y, k, hidden, steps, lr, seed) -> torch.Tensor:
    from sklearn.model_selection import StratifiedKFold

    n = x.shape[0]
    d = x.shape[2]
    c = len(CLASSES)
    preds = torch.empty(n, dtype=torch.long)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, va in skf.split(torch.zeros(n).numpy(), y.numpy()):
        model = SetEncoder(in_dim=d, hidden=hidden, num_classes=c)
        _train(model, x[tr], mask[tr], y[tr], steps, lr, seed)
        preds[va] = _predict(model, x[va], mask[va])
    return preds


def run_probe(
    repo: str | os.PathLike,
    glyph: str = "e",
    threshold: float = 50.0,
    precision: int = 4,
    hidden: int = 128,
    steps: int = 200,
    lr: float = 1e-3,
    seed: int = 0,
    limit: int | None = None,
    folds: int | None = None,
    permute: bool = False,
) -> dict:
    gf = GoogleFonts(repo, having={ord(glyph)})
    designs = select_designs(gf, glyph, threshold, precision)
    print(f"selected {len(designs)} unambiguous designs", flush=True)
    if limit is not None:
        designs = designs[:limit]

    raw: list[list[list[float]]] = []
    ys_list: list[int] = []
    names: list[str] = []
    for d in designs:
        tokens = _tokens_for(d.font, glyph)
        if not tokens:
            continue
        raw.append(tokens[:MAX_POINTS])
        ys_list.append(d.label)
        names.append(d.name)
    x, mask = _pad(raw)
    y = torch.tensor(ys_list, dtype=torch.long)

    n = len(raw)
    if n == 0:
        return {"num_designs": 0}

    eval_y = y
    if permute:
        g = torch.Generator().manual_seed(seed)
        eval_y = y[torch.randperm(n, generator=g)]

    if folds is not None:
        preds = _fold_preds(x, mask, eval_y, folds, hidden, steps, lr, seed)
    else:
        preds = _lofo_preds(x, mask, eval_y, hidden, steps, lr, seed)

    acc = (preds == eval_y).sum().item() / n
    majority = Counter(eval_y.tolist()).most_common(1)[0][1] / n

    per_class = {}
    for c in range(len(CLASSES)):
        idx = eval_y == c
        if idx.sum().item() > 0:
            per_class[CLASSES[c]] = (preds[idx] == c).sum().item() / idx.sum().item()

    confusion = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.long)
    for t, p in zip(eval_y.tolist(), preds.tolist()):
        confusion[t, p] += 1

    return {
        "num_designs": n,
        "accuracy": acc,
        "chance": majority,
        "permuted": permute,
        "folds": folds,
        "per_class": per_class,
        "classes": CLASSES,
        "confusion": confusion.tolist(),
        "names": names,
        "preds": preds.tolist(),
        "labels": eval_y.tolist(),
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--repo",
        default=os.environ.get("GOOGLE_FONTS_REPO", "/home/simon/others-repos/fonts"),
    )
    p.add_argument("--glyph", default="e")
    p.add_argument("--threshold", type=float, default=50.0)
    p.add_argument("--precision", type=int, default=4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="cap designs for a quick run")
    p.add_argument("--folds", type=int, default=None, help="use stratified K-fold instead of leave-one-out")
    p.add_argument("--permute", action="store_true", help="memorisation control")
    args = p.parse_args()

    r = run_probe(
        args.repo,
        glyph=args.glyph,
        threshold=args.threshold,
        precision=args.precision,
        hidden=args.hidden,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
        limit=args.limit,
        folds=args.folds,
        permute=args.permute,
    )

    if r.get("num_designs", 0) == 0:
        print("no designs selected")
        return 1

    mode = "PERMUTED-LABEL" if r["permuted"] else "REAL"
    split = f"{r['folds']}-fold" if r["folds"] else "leave-one-design-out"
    print(f"[{mode} | {split}] designs={r['num_designs']}  accuracy={r['accuracy']:.3f}  "
          f"chance={r['chance']:.3f}")

    print("\nper-class recall:")
    for cls, rec in r["per_class"].items():
        print(f"  {cls:24} {rec:.3f}")

    print("\nconfusion (rows=true, cols=predicted):")
    short = [c.split("/")[-1] for c in r["classes"]]
    print(f"  {'':14}" + "".join(f"{s:>14}" for s in short))
    for i, row in enumerate(r["confusion"]):
        print(f"  {short[i]:14}" + "".join(f"{v:>14}" for v in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
