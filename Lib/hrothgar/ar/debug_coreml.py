#!/usr/bin/env python3
"""Compare Core ML and PyTorch generation outputs step by step.

Usage::

    python -m hrothgar.ar.debug_coreml \\
        MyFont.ttf --char A \\
        --model-dir models/coreml_gen \\
        --ar-model-path models/ar_model.pth \\
        --gtok-model-path models/gtok_model.pth

Reports max absolute difference at each pipeline stage.
Saves ``_pt.png``, ``_cm.png``, ``_inference.png`` for visual comparison.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from hrothgar.ar.export_wrappers import (
    _EncoderExport,
    _MaskGITTransformerExport,
    _SoftDecoderExport,
)
from hrothgar.ar.dataset import _sample_style_codepoints
from hrothgar.ar.model import ARModelConfig, ARModel
from hrothgar.googlefonts import StandaloneFont
from hrothgar.gtok.model import load_model as load_gtok


def _render_refs(
    font: StandaloneFont, count: int, size: int, style_chars: str
) -> np.ndarray:
    """Render *count* style reference glyphs using ``font.render``."""
    refs = []
    for c in style_chars:
        cp = ord(c)
        if font.has_codepoint(cp):
            img = font.render(cp, size=size)
            if not np.allclose(img, 1.0, atol=1e-2):
                refs.append(img)
                if len(refs) >= count:
                    break
    if not refs:
        blank = np.ones((3, size, size), dtype=np.float32)
        refs = [blank] * count
    while len(refs) < count:
        refs.append(refs[-1])
    return np.stack(refs[:count])


def _cosine_unmask_schedule(step: int, total_steps: int, num_tokens: int) -> int:
    if step >= total_steps - 1:
        return num_tokens
    frac_masked = math.cos(math.pi / 2 * (step + 1) / total_steps)
    n_keep = round(num_tokens * (1.0 - frac_masked))
    return max(1, min(n_keep, num_tokens))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Debug Core ML vs PyTorch generation.")
    p.add_argument("font", type=Path)
    p.add_argument("--char", type=str, required=True)
    p.add_argument("--model-dir", type=Path, default=Path("models/coreml_gen"))
    p.add_argument("--ar-model-path", type=Path, required=True)
    p.add_argument("--gtok-model-path", type=Path, required=True)
    p.add_argument("--style-ref-count", type=int, default=8)
    p.add_argument("--reference-font", type=Path, default=None,
                   help="Optional reference font for content image.")
    p.add_argument("--style-chars", type=str, default="ABEGNRSTabdeghknpqy023456789",
                   help="Characters to use as style references.")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/gen_debug"))
    return p


def main() -> None:
    args = _build_parser().parse_args()
    device = torch.device("cpu")

    char = args.char
    if len(char) != 1:
        raise ValueError("--char must be a single character")

    # ---- Load config ----
    config = ARModelConfig.from_sidecar(args.ar_model_path)
    H = config.image_size
    K = args.style_ref_count
    print(f"Config: image_size={H}  K={K}  steps={config.maskgit_num_inference_steps}")

    # ---- Render inputs using StandaloneFont (same as generate.py) ----
    target_font = StandaloneFont(args.font)
    ref_font: Optional[StandaloneFont] = None
    if args.reference_font:
        ref_font = StandaloneFont(args.reference_font, reference=target_font)
        target_font = StandaloneFont(args.font, reference=ref_font)

    # Content: from reference font if available, else from target.
    content_font = ref_font if ref_font else target_font
    if not content_font.has_codepoint(ord(char)):
        content_font = target_font
    content_np = content_font.render(ord(char), size=H)

    # Style: use exact same sampling as generate.py.
    style_codepoints_str = args.style_chars
    common_style_cps = [ord(c) for c in style_codepoints_str] if style_codepoints_str else None
    style_chars = _sample_style_codepoints(
        font=target_font,
        target_char=ord(char),
        style_glyph_count=K,
        common_style_codepoints=common_style_cps,
    )
    style_rasters = [target_font.render(cp, size=H) for cp in style_chars]
    style_np = np.stack(style_rasters)
    print(f"Inputs: content={content_np.shape}  style={style_np.shape}")
    if ref_font:
        print(f"  content from: {args.reference_font}")

    # ---- Load PyTorch model ----
    gtok, _gtok_cfg = load_gtok(args.gtok_model_path, device)
    pt_model = ARModel(config, gtok_model=gtok).to(device)
    pt_model.load(str(args.ar_model_path), device=device)
    pt_model.eval()
    pt_model.freeze_gtok()

    # ---- Load Core ML models ----
    import coremltools as ct

    def _find(base):
        for ext in (".mlmodelc", ".mlpackage"):
            c = base.with_suffix(ext)
            if c.exists():
                return ct.models.MLModel(str(c))
        raise FileNotFoundError(str(base))

    cm_encoder = _find(args.model_dir / "gen_encoder")
    cm_transformer = _find(args.model_dir / "gen_transformer")
    cm_softdecoder = _find(args.model_dir / "gen_softdecoder")

    spec = cm_transformer.get_spec()
    N = spec.description.input[0].type.multiArrayType.shape[1]
    vocab_size = spec.description.output[0].type.multiArrayType.shape[2]
    mask_id = vocab_size
    print(f"Transformer: seq_len={N}  vocab={vocab_size}")

    # ---- Prepare tensors ----
    content_pt = torch.tensor(content_np).unsqueeze(0).to(device)
    style_pt = torch.tensor(style_np).unsqueeze(0).to(device)
    latincore_idx = torch.tensor([_codepoint_to_latincore(ord(char))]).to(device)

    # ---- Stage 1: Encoder ----
    print("\n=== Stage 1: Encoder ===")
    with torch.no_grad():
        pt_cm = pt_model.build_conditioning_map(content_pt, style_pt, latincore_idx)
    pt_cm_np = pt_cm.cpu().numpy()

    cm_cm = cm_encoder.predict({
        "content_image": content_np[np.newaxis, ...].astype(np.float32),
        "style_refs": style_np[np.newaxis, ...].astype(np.float32),
        "latincore_idx": np.array([latincore_idx.item()], dtype=np.int32),
    })["conditioning_map"]

    enc_diff = np.abs(pt_cm_np - cm_cm).max()
    print(f"  cond_map max diff: {enc_diff:.2e}")
    if enc_diff > 0.1:
        print("  ⚠ Large encoder mismatch — stopping")
        return

    # ---- Stage 2: Transformer ----
    print("\n=== Stage 2: Transformer ===")
    temperature = config.maskgit_temperature
    num_steps = config.maskgit_num_inference_steps

    pt_predicted = torch.full((1, N), mask_id, dtype=torch.long, device=device)
    pt_unmasked = torch.zeros(1, N, dtype=torch.bool, device=device)
    cm_predicted = np.full((1, N), mask_id, dtype=np.int32)
    cm_unmasked = np.zeros((1, N), dtype=bool)

    for step in range(num_steps):
        with torch.no_grad():
            pt_logits = pt_model.maskgit_decoder.transformer(
                idx=pt_predicted, imgs_feature_map=pt_cm
            )
        cm_logits = cm_transformer.predict({
            "token_indices": cm_predicted.astype(np.int32),
            "conditioning_map": cm_cm,
        })["logits"]

        logit_diff = np.abs(pt_logits.cpu().numpy() - cm_logits).max()
        print(f"  step {step}: logits max diff = {logit_diff:.2e}")

        # MaskGIT step (same logic for both).
        pt_conf = torch.softmax(pt_logits[0] / temperature, dim=-1).max(dim=-1)
        cm_probs = _softmax(cm_logits[0] / temperature, axis=-1)
        cm_conf = cm_probs.max(axis=-1)
        cm_tokens = cm_probs.argmax(axis=-1)

        def _step(conf, tokens, unmasked, predicted_np):
            conf = conf.copy() if isinstance(conf, np.ndarray) else conf.numpy().copy()
            tokens = tokens.copy() if isinstance(tokens, np.ndarray) else tokens.numpy().copy()
            unmasked = unmasked.copy()
            predicted_np = predicted_np.copy()
            conf[unmasked[0]] = 0.0
            target_keep = _cosine_unmask_schedule(step + 1, num_steps, N)
            n_already = int(unmasked[0].sum())
            n_to_unmask = max(1, target_keep - n_already)
            top = np.argsort(-conf)[:n_to_unmask]
            unmasked[0, top] = True
            predicted_np[0, top] = tokens[top]
            return predicted_np, unmasked

        pt_predicted_np, pt_unmasked_np = _step(
            pt_conf.values.cpu(), pt_conf.indices.cpu(),
            pt_unmasked.cpu().numpy(), pt_predicted.cpu().numpy(),
        )
        pt_predicted = torch.tensor(pt_predicted_np, dtype=torch.long, device=device)
        pt_unmasked = torch.tensor(pt_unmasked_np, dtype=torch.bool, device=device)

        cm_predicted, cm_unmasked = _step(
            cm_conf, cm_tokens, cm_unmasked, cm_predicted,
        )

    # Final logits.
    with torch.no_grad():
        pt_final = pt_model.maskgit_decoder.transformer(
            idx=pt_predicted, imgs_feature_map=pt_cm
        )
    cm_final = cm_transformer.predict({
        "token_indices": cm_predicted.astype(np.int32),
        "conditioning_map": cm_cm,
    })["logits"]

    # ---- Stage 3: Soft decoder ----
    print("\n=== Stage 3: Soft Decoder ===")
    with torch.no_grad():
        _, pt_img = pt_model.soft_decode(pt_final, temperature=1.0)
    pt_img_np = pt_img.cpu().numpy()

    cm_img = cm_softdecoder.predict({
        "logits": cm_final.astype(np.float32),
    })["images"]

    img_diff = np.abs(pt_img_np - cm_img).max()
    print(f"  image max diff: {img_diff:.2e}")

    # ---- Save input images for comparison ----
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.font.stem}_{char}"

    def _save(path, arr):
        if arr.ndim == 4:
            arr = arr[0]
        plt.imsave(path, arr.transpose(1, 2, 0).clip(0, 1), vmin=0, vmax=1)

    _save(args.output_dir / f"{stem}_content.png", content_np)
    for i in range(min(K, style_np.shape[0])):
        _save(args.output_dir / f"{stem}_style_{i}.png", style_np[i])
    print(f"Saved input images to {args.output_dir}/")

    _save(args.output_dir / f"{stem}_pt.png", pt_img_np)
    _save(args.output_dir / f"{stem}_cm.png", cm_img)
    print(f"Saved: {args.output_dir / f'{stem}_pt.png'}")
    print(f"Saved: {args.output_dir / f'{stem}_cm.png'}")

    # Also test the actual inference module.
    from hrothgar.ar.inference_coreml import GeneratorInference
    gen = GeneratorInference(args.model_dir)
    inf_img = gen.generate(
        content_image=content_np,
        style_refs=style_np,
        target_codepoint=ord(char),
    )
    inf_diff = np.abs(pt_img_np.squeeze(0) - inf_img).max()
    print(f"\nInference module vs PyTorch: {inf_diff:.2e}")
    _save(args.output_dir / f"{stem}_inference.png", inf_img[np.newaxis, ...])
    print(f"Saved: {args.output_dir / f'{stem}_inference.png'}")


def _codepoint_to_latincore(cp: int) -> int:
    from hrothgar.dataset import LATIN_CORE
    return LATIN_CORE.index(cp) if cp in LATIN_CORE else 0


if __name__ == "__main__":
    main()
