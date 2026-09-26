#!/usr/bin/env python
"""Decoder cross-attention diagnostic (v2 and v3 style-extraction models).

The decoder injects style by cross-attending its content queries (one per
output-grid position) to the K style tokens.  This script captures those
attention weights and reports, per decoder layer, how the attention is shaped:

* ``entropy``   — normalised attention entropy, mean over queries/heads.
                  ``≈ 1`` = uniform (pooling to the mean); ``≈ 0`` = one-hot.
* ``eff_tokens`` — participation ratio ``1/Σp²`` per query (effective number of
                  tokens each query attends to).  ``≈ K`` = uniform; ``≈ 1`` = peaked.
* ``usage_eff``  — effective number of *distinct* tokens receiving attention
                  overall (from the marginal over queries/heads).  A small value
                  means the decoder ignores most of the (diverse) tokens.
* ``q_var``      — variance of attention across query positions.  ``≈ 0`` means
                  attention is position-independent (global style); ``> 0`` means
                  different positions attend to different tokens (localised style).

Healthy signature (decoder is really using per-token, localised style):
low ``entropy``, low ``eff_tokens``, moderate ``usage_eff`` (several tokens used,
but each query reads only a few), and ``q_var > 0``.

Pooling signature (decoder still effectively reads the mean):
``entropy ≈ 1``, ``eff_tokens ≈ K``, ``q_var ≈ 0``.

Example::

    PYTHONPATH=Lib python scripts/decoder_attention_probe.py \\
        --model-path models/style_extraction.pth --dataset-path ../fonts
"""

from __future__ import annotations

import argparse
import math

import torch

from hrothgar.style_extraction import load_model
from hrothgar.style_extraction.dataset import StyleExtractionDatasetMaker
from hrothgar.utils import torch_setup


def _entropy(attn: torch.Tensor) -> torch.Tensor:
    """Normalised entropy over the last dim.  ``attn``: ``(..., K)`` sums to 1."""
    k = attn.shape[-1]
    h = -(attn * (attn + 1e-12).log()).sum(dim=-1) / math.log(k)
    return h  # (...,)


def _participation_ratio(attn: torch.Tensor) -> torch.Tensor:
    return 1.0 / (attn**2).sum(dim=-1)


def _make_hook(sink: dict[int, torch.Tensor], layer_idx: int):
    def hook(module, inp, out):
        query, key, _value = inp  # query=(B,nq,D), key=(B,K,D)
        b, nq, _ = query.shape
        heads = module.heads
        hd = module.head_dim
        q = module.q_proj(query).view(b, nq, heads, hd).transpose(1, 2)  # (B,h,nq,hd)
        k = (
            module.k_proj(key).view(b, key.shape[1], heads, hd).transpose(1, 2)
        )  # (B,h,K,hd)
        a = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd)  # (B,h,nq,K)
        sink[layer_idx] = a.softmax(dim=-1).detach()

    return hook


def _analyze(attn: torch.Tensor) -> dict[str, float]:
    # attn: (B, heads, nq, K)
    b, h, nq, k = attn.shape
    ent = _entropy(attn)  # (B, h, nq)
    eff = _participation_ratio(attn)  # (B, h, nq)
    maxw = attn.max(dim=-1).values  # (B, h, nq)
    marginal = attn.mean(dim=(0, 1, 2))  # (K,)
    usage_eff = _participation_ratio(marginal)  # scalar
    q_var = attn.var(dim=2).mean()  # variance across queries, mean over (B,h,K)
    return {
        "entropy": float(ent.mean()),
        "eff_tokens": float(eff.mean()),
        "maxw": float(maxw.mean()),
        "usage_eff": float(usage_eff),
        "q_var": float(q_var),
    }


def _save_heatmaps(
    captured: dict[int, torch.Tensor],
    grid_size: int,
    out_dir: str,
    topk: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from pathlib import Path

    import matplotlib.pyplot as plt

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for layer, attn in captured.items():
        a = attn[0]  # (heads, nq, K)
        heads, nq, k = a.shape
        # Pick the most-peaked head (lowest mean entropy) as the representative.
        head_ent = _entropy(a).mean(dim=1)  # (heads,)
        head = int(head_ent.argmin())
        marginal = a[head].mean(dim=0)  # (K,)
        top_tokens = marginal.argsort(descending=True)[:topk].tolist()

        fig, axes = plt.subplots(1, topk, figsize=(topk * 1.4, 1.6))
        if topk == 1:
            axes = [axes]
        for ax, tok in zip(axes, top_tokens):
            spatial = a[head, :, tok].reshape(grid_size, grid_size).cpu().numpy()
            im = ax.imshow(spatial, cmap="viridis")
            ax.set_title(f"tok {tok}\n{marginal[tok]:.3f}", fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"decoder layer {layer} — cross-attn head {head}", fontsize=8)
        fig.tight_layout()
        out = str(Path(out_dir) / f"decoder_attn_layer{layer}.png")
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"  saved {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--topk", type=int, default=6, help="Tokens to show per layer in the heatmap."
    )
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--no-heatmaps", action="store_true")
    args = p.parse_args()

    device = torch_setup()
    model, config = load_model(args.model_path, device)

    maker = StyleExtractionDatasetMaker(
        repo_url=args.dataset_path,
        batch_size=args.batch_size,
        image_size=config.image_size,
        character_set=config.character_set,
        num_evidence_glyphs=config.num_evidence_glyphs,
    )
    batch = next(iter(maker.test_loader()))
    style_images = batch["style_images"].to(device)
    style_cp = batch["style_codepoint_idx"].to(device)
    target_idx = batch["target_codepoint_idx"].to(device)

    captured: dict[int, torch.Tensor] = {}
    if hasattr(model, "decoder_blocks"):
        # v2: hook each decoder block's cross-attention.
        handles = []
        for i, blk in enumerate(model.decoder_blocks):
            handles.append(
                blk.cross_attn.register_forward_hook(_make_hook(captured, i))
            )
        with torch.no_grad():
            style_tokens = model.encode_style(
                style_images, style_codepoint_idx=style_cp
            )
            model.decode(target_idx, style_tokens)
        for h in handles:
            h.remove()
    else:
        # v3: a single content-conditioned cross-attention, exposed directly.
        with torch.no_grad():
            style_tokens = model.encode_style(
                style_images, style_codepoint_idx=style_cp
            )
            _img, attn = model.decode(target_idx, style_tokens, return_attention=True)
        captured[0] = attn

    k = style_tokens.shape[1]
    print(f"Style tokens: {k}  |  decoder layers: {len(captured)}")
    print(
        f"  {'layer':<6} {'entropy':>9} {'eff_tokens':>11} {'maxw':>7} {'usage_eff':>10} {'q_var':>9}"
    )
    for layer in sorted(captured):
        m = _analyze(captured[layer])
        print(
            f"  {layer:<6} {m['entropy']:>9.3f} {m['eff_tokens']:>11.1f} "
            f"{m['maxw']:>7.3f} {m['usage_eff']:>10.1f} {m['q_var']:>9.4f}"
        )

    # A one-line verdict from the last (deepest) layer.
    last = _analyze(captured[max(captured)])
    if last["entropy"] > 0.7 and last["eff_tokens"] > 0.5 * k:
        print(
            "\nVERDICT: cross-attention is near-uniform — the decoder is pooling to the mean."
        )
    elif last["entropy"] < 0.4 and last["q_var"] > 1e-3:
        print(
            "\nVERDICT: cross-attention is peaked and position-dependent — localised style use."
        )
    else:
        print(
            "\nVERDICT: intermediate — attention is partially peaked; inspect per-layer rows."
        )

    if not args.no_heatmaps:
        _save_heatmaps(captured, model.grid_size, args.out_dir, args.topk)


if __name__ == "__main__":
    main()
