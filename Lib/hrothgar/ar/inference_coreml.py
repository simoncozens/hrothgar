"""Core ML inference for the MaskGIT generation model.

Runs the exported Core ML models via ``coremltools``.  Implements the MaskGIT
iterative decoding loop in Python.

Usage::

    from hrothgar.ar.inference_coreml import GeneratorInference

    gen = GeneratorInference("models/coreml_gen")
    images = gen.generate(
        content_image=content_numpy,
        style_refs=style_refs_numpy,
        target_codepoint=ord("A"),
    )
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Union

import numpy as np
from hrothgar.dataset_constants import LATIN_CORE
LATIN_CORE.append(0x20B9)


try:
    import coremltools as ct  # type: ignore[import-untyped]
except ImportError:
    raise ImportError("coremltools is required.  pip install coremltools")


def _load_model(model_path: Path) -> ct.models.MLModel:
    return ct.models.MLModel(str(model_path))


# ---------------------------------------------------------------------------
# MaskGIT decode schedule
# ---------------------------------------------------------------------------


def _cosine_unmask_schedule(step: int, total_steps: int, num_tokens: int) -> int:
    if step >= total_steps - 1:
        return num_tokens
    frac_masked = math.cos(math.pi / 2 * (step + 1) / total_steps)
    n_keep = round(num_tokens * (1.0 - frac_masked))
    return max(1, min(n_keep, num_tokens))


# ---------------------------------------------------------------------------
# Generator Inference
# ---------------------------------------------------------------------------


class GeneratorInference:
    """Run the MaskGIT generation pipeline using exported Core ML models."""

    def __init__(self, model_dir: Union[str, Path]) -> None:
        model_dir = Path(model_dir)

        def _find(base: Path) -> Path:
            for ext in (".mlmodelc", ".mlpackage"):
                c = base.with_suffix(ext)
                if c.exists():
                    return c
            raise FileNotFoundError(f"Model not found: {base}.mlmodelc or .mlpackage")

        self._encoder = _load_model(_find(model_dir / "gen_encoder"))
        self._transformer = _load_model(_find(model_dir / "gen_transformer"))
        self._softdecoder = _load_model(_find(model_dir / "gen_softdecoder"))

        # Read shapes from the transformer model spec.
        spec = self._transformer.get_spec()
        token_input = spec.description.input[0]
        self._seq_len = token_input.type.multiArrayType.shape[1]
        logits_output = spec.description.output[0]
        self._vocab_size = logits_output.type.multiArrayType.shape[2]
        self._mask_token_id = self._vocab_size  # [MASK] token convention

        # Load config from sidecar.
        from hrothgar.ar.config import ARModelConfig
        sidecar = model_dir / "gen_config.pth.conf.json"
        if not sidecar.exists():
            for f in model_dir.glob("*.conf.json"):
                sidecar = f
                break
        self._config = ARModelConfig.from_sidecar(sidecar)
        self._image_size = self._config.image_size
        self._num_steps = self._config.maskgit_num_inference_steps
        self._temperature = self._config.maskgit_temperature
        self._target_codepoints = self._config.target_codepoints
        self._target_only = self._config.target_only
        self._style_codepoints = self._config.style_codepoints

    @property
    def image_size(self) -> int:
        return self._image_size

    @property
    def num_inference_steps(self) -> int:
        return self._num_steps

    @property
    def target_glyphset(self) -> Optional[list[int]]:
        """Codepoints the model was trained to generate (None = full Latin Core)."""
        return self._target_codepoints

    @property
    def style_glyphset(self) -> Optional[list[int]]:
        """Codepoints used for style references during training."""
        return self._style_codepoints

    def generate(
        self,
        content_image: np.ndarray,
        style_refs: np.ndarray,
        target_codepoint: int,
    ) -> np.ndarray:
        """Generate a glyph for *target_codepoint*.

        Args:
            content_image: ``(3, H, H)`` float32, values in [0, 1].
            style_refs: ``(K, 3, H, H)`` float32 array of reference glyphs.
            target_codepoint: Unicode codepoint.

        Returns:
            ``(3, H, H)`` float32 generated glyph.
        """
        # ---- Codepoint → LATIN_CORE index ----
        cp = target_codepoint
        latincore_idx = LATIN_CORE.index(cp) if cp in LATIN_CORE else 0

        # ---- Step 1: Build conditioning map ----
        cond_map = self._encoder.predict({
            "content_image": content_image[np.newaxis, ...].astype(np.float32),
            "style_refs": style_refs[np.newaxis, ...].astype(np.float32),
            "latincore_idx": np.array([latincore_idx], dtype=np.int32),
        })["conditioning_map"]

        # ---- Step 2: MaskGIT iterative decoding ----
        N = self._seq_len
        B = 1
        predicted = np.full((B, N), self._mask_token_id, dtype=np.int32)
        unmasked = np.zeros((B, N), dtype=bool)

        for step in range(self._num_steps):
            logits = self._transformer.predict({
                "token_indices": predicted.astype(np.int32),
                "conditioning_map": cond_map,
            })["logits"]  # (1, N, K)

            probs = _softmax(logits[0] / self._temperature, axis=-1)
            max_probs = probs.max(axis=-1)
            pred_tokens = probs.argmax(axis=-1)

            max_probs = np.where(unmasked[0], 0.0, max_probs)
            target_keep = _cosine_unmask_schedule(step + 1, self._num_steps, N)
            n_already = int(unmasked[0].sum())
            n_to_unmask = max(1, target_keep - n_already)

            top_indices = np.argsort(-max_probs)[:n_to_unmask]
            unmasked[0, top_indices] = True
            predicted[0, top_indices] = pred_tokens[top_indices]

        # Safety net: fill remaining masked positions, then get final logits.
        remaining = ~unmasked[0]
        final_logits = None
        if remaining.any():
            final_logits = self._transformer.predict({
                "token_indices": predicted.astype(np.int32),
                "conditioning_map": cond_map,
            })["logits"]
            predicted[0, remaining] = final_logits[0, remaining].argmax(axis=-1)

        # Get logits for the completed token sequence.
        if final_logits is None:
            final_logits = self._transformer.predict({
                "token_indices": predicted.astype(np.int32),
                "conditioning_map": cond_map,
            })["logits"]

        # ---- Step 3: Soft decode ----
        images = self._softdecoder.predict({
            "logits": final_logits.astype(np.float32),
        })["images"]

        return images.squeeze(0).astype(np.float32)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


__all__ = ["GeneratorInference"]
