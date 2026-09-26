#!/usr/bin/env python
"""Hard-vs-soft decode + latent interpolation probe for the AR generator.

Two complementary diagnostics:

1. **Hard vs soft decode** — for held-out glyphs, compare the model's
   *soft* decode (``softmax(logits) @ codebook`` — what the AR model currently
   produces) against a *hard* decode (``argmax(logits)`` → codebook lookup →
   ``gtok.decode``).  If soft decode is dirty (blotches / missing strokes) while
   hard decode is clean, the corruption is a decode artifact, not a token
   prediction or conditioning failure.

2. **Latent interpolation grid** — interpolate between two glyphs' quantized
   GTok latents (raw blend and L2-normalized blend) and decode each step, to
   visualise whether the tokenizer's latent space is interpolatable off the
   codebook.

Example::

    PYTHONPATH=Lib python scripts/ar_decode_probe.py \
        --gtok-model-path models/gtok.pth \
        --ar-model-path models/ar.pth \
        --dataset-path ~/google/fonts_checkout \
        --font "Roboto" \
        --chars aegs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import uharfbuzz as hb

from hrothgar.ar.model import ARModel, ARModelConfig
from hrothgar.glyph_rendering import render_normalized
from hrothgar.googlefonts import GoogleFonts, find_google_font_by_basename
from hrothgar.gtok.model import load_model as load_gtok_model
from hrothgar.utils import pick_device

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_stack(args, device):
    gtok, gtok_config = load_gtok_model(Path(args.gtok_model_path), device)
    image_size = gtok_config.image_size

    ar_config = ARModelConfig.from_sidecar(args.ar_model_path)
    if ar_config.image_size != image_size:
        raise ValueError(
            f"AR model image_size {ar_config.image_size} != GTok image_size "
            f"{image_size}"
        )
    model = ARModel(ar_config, gtok_model=gtok).to(device)
    model.load(args.ar_model_path, device)
    model.eval()

    return gtok, model, image_size


def resolve_font(args, gf: GoogleFonts):
    font = gf.families_by_name.get(args.font)
    if font is not None:
        return font
    try:
        return find_google_font_by_basename(args.dataset_path, args.font)
    except Exception as exc:
        raise ValueError(f"Font not found: {args.font} ({exc})") from exc


# ---------------------------------------------------------------------------
# Hard / soft decode
# ---------------------------------------------------------------------------


def glyph_metrics(font, char: int) -> torch.Tensor:
    """Return the 6-dim metric vector used by the AR generator."""
    upem = float(font.hb_face.upem)
    vm = font.vertical_metrics()
    gid = hb.Font(font.hb_face).get_nominal_glyph(char)
    aw = font.advance_width(gid) / upem if upem > 0 else 0.0
    return torch.tensor(
        [
            float(vm["ascender"]) / upem if upem > 0 else 0.0,
            float(vm["descender"]) / upem if upem > 0 else 0.0,
            float(vm["x_height"]) / upem if upem > 0 else 0.0,
            float(vm["cap_height"]) / upem if upem > 0 else 0.0,
            float(vm["baseline"]) / upem if upem > 0 else 0.0,
            aw,
        ],
        dtype=torch.float32,
    )


def decode_soft(model: ARModel, logits: torch.Tensor) -> torch.Tensor:
    """Soft-decode logits → image (the AR model's current path)."""
    _soft_emb, images = model.soft_decode(logits, temperature=1.0)
    return images


def decode_hard(model: ARModel, logits: torch.Tensor) -> torch.Tensor:
    """Hard-decode argmax tokens → image."""
    indices = logits.argmax(dim=-1)  # (B, seq)
    codebook = model.codebook_embeddings()  # (V, D)
    hard_emb = codebook[indices]  # (B, seq, D)
    return model.gtok.decode(hard_emb)


def run_hard_soft(
    model: ARModel,
    font,
    reference_font,
    chars: str,
    image_size: int,
    device: torch.device,
) -> list[dict]:
    """Return per-char soft/hard reconstruction diagnostics."""
    rows = []
    for ch in chars:
        cp = ord(ch)
        try:
            target_norm, _ = render_normalized(font, cp, image_size)
            content_norm, _ = render_normalized(reference_font, cp, image_size)
        except Exception as exc:
            print(f"  [skip] U+{cp:04X}: {exc}")
            continue

        target = target_norm.unsqueeze(0).to(device)
        content = content_norm.unsqueeze(0).to(device)
        cp_t = torch.tensor([cp], device=device)
        metrics = glyph_metrics(font, cp).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(
                content,
                target_images=target,
                target_codepoints=cp_t,
                metrics=metrics,
            )

        soft = out.reconstructed_images
        hard = decode_hard(model, out.logits)
        gt_tokens = out.target_token_indices
        pred = out.logits.argmax(dim=-1)

        token_acc = (pred == gt_tokens).float().mean().item()
        soft_l1 = F.l1_loss(soft, target).item()
        hard_l1 = F.l1_loss(hard, target).item()

        rows.append(
            {
                "char": ch,
                "target": target[0].cpu(),
                "soft": soft[0].cpu(),
                "hard": hard[0].cpu(),
                "token_acc": token_acc,
                "soft_l1": soft_l1,
                "hard_l1": hard_l1,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Latent interpolation grid
# ---------------------------------------------------------------------------


def interpolation_grid(
    gtok,
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    device: torch.device,
    alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Interpolate two glyph latents (raw + normalized) and decode each step."""
    with torch.no_grad():
        qa, _ = gtok.encode(img_a.unsqueeze(0).to(device))  # (1, seq, D)
        qb, _ = gtok.encode(img_b.unsqueeze(0).to(device))
    qa, qb = qa[0], qb[0]  # (seq, D)

    raw, norm = [], []
    for a in alphas:
        z = (1.0 - a) * qa + a * qb
        raw.append(gtok.decode(z.unsqueeze(0))[0].cpu())
        z_norm = F.normalize(z, p=2, dim=-1)
        norm.append(gtok.decode(z_norm.unsqueeze(0))[0].cpu())
    return raw, norm


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _to_hwc(t: torch.Tensor) -> torch.Tensor:
    """(3, H, W) → (H, W, 3) clipped to [0, 1]."""
    return t.clamp(0.0, 1.0).permute(1, 2, 0)


def save_hard_soft(row: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images = [row["target"], row["soft"], row["hard"]]
    labels = ["GT", "soft", "hard"]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, im, lbl in zip(axes, images, labels):
        ax.imshow(_to_hwc(im), cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(lbl, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"U+{ord(row['char']):04X}  soft_l1={row['soft_l1']:.4f} "
        f"hard_l1={row['hard_l1']:.4f}  tok_acc={row['token_acc']:.3f}",
        fontsize=8,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  saved {out_path}")


def save_interp_grid(raw, norm, alphas, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(alphas)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.6, 3.4))
    for j, a in enumerate(alphas):
        axes[0, j].imshow(_to_hwc(raw[j]), cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, j].imshow(_to_hwc(norm[j]), cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, j].set_title(f"α={a:g}", fontsize=8)
    axes[0, 0].set_ylabel("raw blend", fontsize=8)
    axes[1, 0].set_ylabel("normalized", fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("GTok latent interpolation (raw vs L2-normalized)", fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gtok-model-path", required=True)
    p.add_argument("--ar-model-path", required=True)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--font", required=True, help="Font family name or file basename.")
    p.add_argument("--chars", default="aegs", help="Codepoints for hard/soft decode.")
    p.add_argument(
        "--interp-chars",
        default=None,
        help="Two chars (e.g. 'ae') to interpolate between; defaults to "
        "first two chars of --chars.",
    )
    p.add_argument("--out-dir", default="outputs/ar_decode")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    print(f"Using device: {device}")

    gtok, model, image_size = load_stack(args, device)
    print(f"Loaded GTok + AR model (image_size={image_size})")

    gf = GoogleFonts(args.dataset_path)
    font = resolve_font(args, gf)
    reference_font = font.reference_font() or font
    print(f"Target font: {font.family} ({font.path.name})")
    print(f"Reference (content) font: {reference_font.family}")

    # ── Part 1: hard vs soft decode ────────────────────────────────────────
    print("\nHard vs soft decode:")
    print(f"  {'char':<6} {'soft_l1':>9} {'hard_l1':>9} {'tok_acc':>8}")
    rows = run_hard_soft(model, font, reference_font, args.chars, image_size, device)
    for row in rows:
        print(
            f"  {row['char']:<6} {row['soft_l1']:>9.4f} {row['hard_l1']:>9.4f} "
            f"{row['token_acc']:>8.3f}"
        )
        save_hard_soft(
            row, Path(args.out_dir) / f"hard_soft_U{ord(row['char']):04X}.png"
        )

    # ── Part 2: latent interpolation grid ──────────────────────────────────
    interp_chars = args.interp_chars or args.chars[:2]
    if len(interp_chars) < 2:
        print("\nSkipping interpolation (need at least two chars).")
        return

    ca, cb = interp_chars[0], interp_chars[1]
    try:
        img_a, _ = render_normalized(font, ord(ca), image_size)
        img_b, _ = render_normalized(font, ord(cb), image_size)
    except Exception as exc:
        print(f"\nSkipping interpolation: {exc}")
        return

    alphas = (0.0, 0.25, 0.5, 0.75, 1.0)
    raw, norm = interpolation_grid(gtok, img_a, img_b, device, alphas=alphas)
    save_interp_grid(
        raw,
        norm,
        alphas,
        Path(args.out_dir) / f"interp_{ca}_{cb}.png",
    )
    print(f"\nInterpolated latents for '{ca}' → '{cb}' (raw + normalized).")


if __name__ == "__main__":
    main()
