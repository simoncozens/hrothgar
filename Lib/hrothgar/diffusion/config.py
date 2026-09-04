"""Configuration for the diffusion glyph generator.

Phase 1 is a **class-conditional** diffusion model: each training sample is a
``(glyph image, class id)`` pair, where the class id folds codepoint and style
axis position into a single label.  This is the fastest path to answer "is
diffusion even capable of rendering fine style detail", before we invest in the
exemplar-conditional UNet + cross-attention of Phase 2 (the many-shot mode that
maps onto our actual pain points).

The heavy lifting is delegated to ``denoising_diffusion_pytorch``'s
classifier-free-guidance module, which already provides a class-conditional
``Unet`` and a ``GaussianDiffusion`` that threads ``classes`` through training
and sampling.  We only configure it; we do not reimplement the diffusion
process.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DiffusionConfig:
    """Configuration for the class-conditional diffusion glyph generator."""

    # Rendering / I/O.
    image_size: int = 128
    channels: int = 1  # grayscale

    # Class vocabulary size.  This is data-derived (codepoint set x axis
    # positions), not a hyperparameter; set it from the dataset before building
    # the model.
    num_classes: int = 0

    # UNet.
    dim: int = 64
    dim_mults: tuple[int, ...] = (1, 2, 4, 8)
    attn_dim_head: int = 32
    attn_heads: int = 4

    # Diffusion process.
    timesteps: int = 250
    # Number of reverse steps used at sampling time.  ``None`` means "full
    # reverse" (``timesteps`` steps).  Smaller values switch the library into
    # DDIM sampling and are much faster for the canary's frequent eval.
    sampling_timesteps: int | None = 50
    objective: str = "pred_noise"
    beta_schedule: str = "cosine"
    ddim_sampling_eta: float = 0.0  # 0 = deterministic DDIM

    # Classifier-free guidance.  ``cond_drop_prob`` is the fraction of training
    # samples whose class embedding is replaced with the learned null embedding;
    # ``cond_scale`` is the guidance scale used at sampling time.  The library
    # samples via ``forward_with_cond_scale``, which unpacks a two-tuple only
    # when ``cond_scale != 1``, so keep ``cond_scale != 1.0``.
    cond_drop_prob: float = 0.1
    cond_scale: float = 3.0

    # Training.
    learning_rate: float = 1e-4
    weight_decay: float = 0.0

    # Auxiliary reconstruction objective.  ``glyphloss_weight`` adds a
    # curvature-weighted glyph reconstruction loss on a *sampled* glyph
    # (differentiable DDIM sampling from pure noise), so the gradient reaches
    # the class conditioning — the thing a single-timestep ``x_0`` recovery
    # cannot do.  ``0`` disables it (pure diffusion).
    glyphloss_weight: float = 0.0
    # Number of DDIM steps in the differentiable sample (smaller = cheaper but
    # coarser).  Sampling from pure noise is what forces the conditioning to
    # carry fine detail.
    glyphloss_sample_steps: int = 10
    # Compute the auxiliary loss every N training steps.  Backprop through the
    # sample chain is expensive, so it is amortized across steps.
    glyphloss_sample_every: int = 20

    def save_sidecar(self, model_path) -> None:
        """Save config as a JSON sidecar alongside the model weights."""
        import hrothgar.utils as u

        config_path = Path(str(model_path)).with_suffix(".conf.json")
        data = dataclasses.asdict(self)
        data["git_sha"] = u.git_short_sha()
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")

    @classmethod
    def from_sidecar(cls, model_path):
        """Load config from a JSON sidecar alongside the model weights."""
        config_path = Path(str(model_path)).with_suffix(".conf.json")
        if not config_path.exists():
            raise FileNotFoundError(f"Config sidecar not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(frozen=True)
class DiffusionLossWeights:
    """Weights for the auxiliary losses reported on sampled glyphs.

    The diffusion *training* objective is the noise-prediction MSE produced by
    ``GaussianDiffusion.forward``; these weights belong to the canary's
    optional supervision signals (L1 / LPIPS / glyphloss / axis regression) on
    the sampled output.  They default to 0 so Phase 1 starts as pure diffusion.
    """

    l1: float = 0.0
    lpips: float = 0.0
    glyphloss: float = 0.0
    axis: float = 0.0


@dataclass
class ExemplarDiffusionConfig:
    """Configuration for the Phase 2 exemplar-conditional diffusion model.

    Instead of a class id, the denoiser is conditioned on (a) a set of *evidence
    glyph images* (the font's style references) encoded into a coarse spatial
    feature map, and (b) the *target codepoint* as a learned embedding added to
    the time embedding.  Cross-attention in the UNet lets each spatial location
    query the style feature map — the mechanism Phase 1 lacked.
    """

    image_size: int = 128
    channels: int = 1

    # Codepoint vocabulary size (data-derived; set before building the model).
    num_codepoints: int = 0

    # Style encoder: evidence glyphs -> (context_dim, style_out_res, style_out_res).
    context_dim: int = 256
    style_out_res: int = 8
    style_encoder_base_channels: int = 64
    num_evidence_glyphs: int = 8

    # Denoiser UNet.
    dim: int = 64
    dim_mults: tuple[int, ...] = (1, 2, 4, 8)
    attn_dim_head: int = 32
    attn_heads: int = 4
    # Feature-map resolutions (side length) where self- + cross-attention run.
    attn_resolutions: tuple[int, ...] = (8, 16)

    # Diffusion process.
    timesteps: int = 250
    sampling_timesteps: int | None = 50
    beta_schedule: str = "cosine"
    ddim_sampling_eta: float = 0.0

    # Training.
    learning_rate: float = 1e-4
    weight_decay: float = 0.0

    def save_sidecar(self, model_path) -> None:
        """Save config as a JSON sidecar alongside the model weights."""
        import hrothgar.utils as u

        config_path = Path(str(model_path)).with_suffix(".conf.json")
        data = dataclasses.asdict(self)
        data["git_sha"] = u.git_short_sha()
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")

    @classmethod
    def from_sidecar(cls, model_path):
        """Load config from a JSON sidecar alongside the model weights."""
        config_path = Path(str(model_path)).with_suffix(".conf.json")
        if not config_path.exists():
            raise FileNotFoundError(f"Config sidecar not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class FontIdDiffusionConfig:
    """Configuration for the factorized (codepoint, font-ID) diffusion model.

    VecFusion's "incomplete fonts" recipe: one-hot codepoint -> embedding ``g``
    and one-hot font style -> embedding ``f``, both injected into the denoiser's
    time embedding.  No exemplar images, no cross-attention — the font's style
    is learned during training from its other glyphs and compressed into the
    font embedding.  Suitable only for *in-corpus* fonts (fill-in-the-library),
    not novel-font generalization.
    """

    image_size: int = 128
    channels: int = 1

    # Vocabulary sizes (data-derived; set before building the model).
    num_codepoints: int = 0
    num_fonts: int = 0

    # Denoiser UNet.
    dim: int = 64
    dim_mults: tuple[int, ...] = (1, 2, 4, 8)
    attn_dim_head: int = 32
    attn_heads: int = 4

    # Diffusion process.
    timesteps: int = 250
    sampling_timesteps: int | None = 50
    beta_schedule: str = "cosine"
    ddim_sampling_eta: float = 0.0

    # Training.
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    # Weight of the geometry regression objective (em-unit labels).  The head
    # predicts scale_x/scale_y/left_sidebearing/baseline_offset/advance so the
    # generated glyph can be placed back on the baseline.
    geometry_loss_weight: float = 1.0

    def save_sidecar(self, model_path) -> None:
        """Save config as a JSON sidecar alongside the model weights."""
        import hrothgar.utils as u

        config_path = Path(str(model_path)).with_suffix(".conf.json")
        data = dataclasses.asdict(self)
        data["git_sha"] = u.git_short_sha()
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")

    @classmethod
    def from_sidecar(cls, model_path):
        """Load config from a JSON sidecar alongside the model weights."""
        config_path = Path(str(model_path)).with_suffix(".conf.json")
        if not config_path.exists():
            raise FileNotFoundError(f"Config sidecar not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
