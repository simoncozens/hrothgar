"""Phase 2: exemplar-conditional diffusion glyph generator.

The style conditioning is a **flat per-font style vector** produced by a shared
*style encoder* over the font's evidence glyphs.  Rather than re-reading the
ever-changing evidence set afresh every shot (the original exemplar design) or
training a raw per-font free parameter from scratch (the fontid design), we
combine both:

* **Style encoder** — a set of *evidence glyphs* (the font's style references)
  is encoded by a shared CNN and **globally pooled** (over both the evidence set
  *and* space) into a single ``(context_dim,)`` style vector.  Global pooling is
  deliberate: style properties (serif shape, terminal angle, stroke modulation,
  flicks) are placement-invariant — *where* a flick lands is the codepoint's
  job, not the style's — so the style representation must not try to carry
  spatial coordinates (the v1-v5 collapse history).

* **Per-font memory bank** — a learned ``nn.Embedding(num_instances,
  context_dim)``, one flat vector per font *instance*.  It is **warm-started**
  from the style encoder over a broad, fixed probe glyph set, then trained by
  the diffusion gradient (so each font accumulates a rich, well-determined
  vector over training, like fontid's ``family_emb``) while a **coupling loss**
  ``‖memory[f] − encoder(evidence_f)‖²`` keeps it tied to what the shared
  encoder actually reads from glyphs.  The coupling is what stops the memory
  drifting into pure memoization (the fontid disease) and lets rare fonts
  borrow the shared encoder's strength.

* **Codepoint conditioning** — the target codepoint is a learned embedding
  concatenated with the style vector (the "class" embedding injected into the
  denoiser's ResNet blocks), mirroring the fontid model.  There is no
  cross-attention in this step; a bag-of-per-glyph vectors (the "step two"
  cross-attention variant) can be layered on later if a single flat vector
  proves insufficient for fine-detail placement.

The diffusion process is reimplemented here (training + DDIM sampling) rather
than reusing ``GaussianDiffusion``, because that class threads a single
``classes`` tensor and we now thread ``(style, codepoint)``.  The math is the
standard cosine-schedule / v- or noise-prediction DDPM; only ``pred_noise`` is
implemented for the scaffold.
"""

from __future__ import annotations

import math
from random import random

import torch
import torch.nn.functional as F
from denoising_diffusion_pytorch.classifier_free_guidance import (
    Attention,
    Downsample,
    PreNorm,
    Residual,
    ResnetBlock,
    SinusoidalPosEmb,
    Upsample,
    cosine_beta_schedule,
    default,
    extract,
    linear_beta_schedule,
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
)
from torch import nn

from hrothgar.diffusion.config import ExemplarDiffusionConfig
from hrothgar.render import (
    DESCENDER_SNAP_EPSILON,
    GEOMETRY_NAMES,
    GEOMETRY_SPEC,
    render,
)
from hrothgar.utils import SaveLoadModel


def _apply_activation(raw: torch.Tensor, activation: str) -> torch.Tensor:
    """Apply the geometry head's per-output activation (mirrors ``fontid``)."""
    return torch.sigmoid(raw) if activation == "sigmoid" else torch.tanh(raw)


def _decode_geometry(raw: torch.Tensor) -> torch.Tensor:
    """Decode ``(B, 5)`` raw head logits to em-unit geometry values."""
    cols = []
    for i, (_, activation, scale) in enumerate(GEOMETRY_SPEC):
        cols.append(_apply_activation(raw[:, i], activation) * scale)
    return torch.stack(cols, dim=1)


# ---------------------------------------------------------------------------
# Style encoder
# ---------------------------------------------------------------------------

# A broad, category-spanning probe glyph set used to warm-start the per-font
# memory bank.  It deliberately covers upper/lower/digit so the warm-start
# summarises a wide variety of glyphs (not just one case), which is what lets a
# font with an idiosyncratic flick (e.g. Autour One's rightward l/m/n/h/f/d
# flicks) be captured even if the evidence for a single target codepoint misses
# the glyphs that carry it.
WARM_START_PROBE = "Hamburgefonstiv0123"


class StyleEncoder(nn.Module):
    """Encode a set of evidence glyphs into a single flat style vector.

    Each evidence glyph passes through the same CNN; the resulting per-glyph
    feature maps are mean-pooled across the evidence set (so the order of
    evidence glyphs is irrelevant) and then globally pooled over space, giving
    a placement-invariant ``(B, context_dim)`` summary.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        context_dim: int,
        image_size: int,
        out_res: int,
    ) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.out_res = out_res
        num_downs = int(round(math.log2(image_size / out_res)))
        if image_size / out_res != 2**num_downs:
            raise ValueError("image_size must be a power-of-2 multiple of out_res")

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(),
        ]
        ch = base_channels
        for _ in range(num_downs):
            out_ch = min(ch * 2, context_dim)
            layers += [
                nn.Conv2d(ch, out_ch, 4, stride=2, padding=1),
                nn.SiLU(),
            ]
            ch = out_ch
        layers.append(nn.Conv2d(ch, context_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        """``evidence`` is ``(B, N, C, H, W)``; returns ``(B, context_dim)``."""
        b, n, c, h, w = evidence.shape
        x = evidence.reshape(b * n, c, h, w)
        x = self.net(x)  # (B*N, context_dim, out_res, out_res)
        x = x.reshape(b, n, self.context_dim, self.out_res, self.out_res)
        x = x.mean(dim=1)  # (B, context_dim, out_res, out_res) — pool over evidence
        return x.mean(dim=(2, 3))  # (B, context_dim) — global pool over space


# ---------------------------------------------------------------------------
# Denoiser UNet
# ---------------------------------------------------------------------------


class ExemplarConditionalUnet(nn.Module):
    """Denoiser UNet conditioned on a flat style vector and a codepoint.

    Mirrors the factorized (fontid) UNet: the codepoint embedding and the style
    vector are concatenated into a single "class" embedding injected into every
    ResNet block, alongside the time embedding.  Self-attention runs at the
    configured resolutions; there is no cross-attention (a flat style vector has
    nothing to cross-attend to — a future bag-of-vectors variant would add it).
    """

    def __init__(
        self,
        dim: int,
        num_codepoints: int,
        context_dim: int,
        image_size: int = 128,
        dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        channels: int = 1,
        attn_dim_head: int = 32,
        attn_heads: int = 4,
        attn_resolutions: tuple[int, ...] = (8, 16),
        self_condition: bool = False,
        geometry_std: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        if geometry_std is not None:
            self.register_buffer(
                "geometry_std", torch.tensor(geometry_std, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "geometry_std", torch.ones(len(GEOMETRY_SPEC), dtype=torch.float32)
            )
        self.channels = channels
        self.out_dim = channels
        self.self_condition = self_condition

        input_channels = channels * (2 if self_condition else 1)
        self.init_conv = nn.Conv2d(input_channels, dim, 7, padding=3)

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.codepoint_emb = nn.Embedding(num_codepoints, time_dim)
        # Codepoint (time_dim) + flat style vector (context_dim), concatenated
        # and injected as the "class" conditioning.  Decoupled from ``dim`` the
        # same way fontid decouples its style embedding width.
        cond_dim = time_dim + context_dim
        self.cond_dim = cond_dim
        # Learned null conditioning for classifier-free guidance: replaces the
        # whole (codepoint, style) conditioning on dropped samples.
        self.null_cond = nn.Parameter(torch.randn(cond_dim))

        # Geometry head: predicts the five em-unit labels so the crop-to-ink
        # glyph can be placed back on the baseline.  Conditioning = the full
        # (codepoint, style) vector + the glyph's own "mode" vector.  These
        # labels are global properties, so the style enters as the (already
        # global) flat vector.
        self.mode_dim = time_dim
        self.glyph_encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, self.mode_dim),
            nn.GELU(),
            nn.Linear(self.mode_dim, self.mode_dim),
        )
        self.geometry_head = nn.Sequential(
            nn.Linear(cond_dim + self.mode_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, len(GEOMETRY_SPEC)),
        )

        attn_res = set(attn_resolutions)

        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            res = image_size // (2**ind)
            has_attn = res in attn_res
            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            dim_in,
                            dim_in,
                            time_emb_dim=time_dim,
                            classes_emb_dim=cond_dim,
                        ),
                        ResnetBlock(
                            dim_in,
                            dim_in,
                            time_emb_dim=time_dim,
                            classes_emb_dim=cond_dim,
                        ),
                        (
                            Residual(
                                PreNorm(
                                    dim_in,
                                    Attention(
                                        dim_in, heads=attn_heads, dim_head=attn_dim_head
                                    ),
                                )
                            )
                            if has_attn
                            else nn.Identity()
                        ),
                        (
                            Downsample(dim_in, dim_out)
                            if not is_last
                            else nn.Conv2d(dim_in, dim_out, 3, padding=1)
                        ),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(
            mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=cond_dim
        )
        self.mid_attn = Residual(
            PreNorm(
                mid_dim, Attention(mid_dim, heads=attn_heads, dim_head=attn_dim_head)
            )
        )
        self.mid_block2 = ResnetBlock(
            mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=cond_dim
        )

        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            res = image_size // (2 ** (num_resolutions - 1 - ind))
            has_attn = res in attn_res
            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            dim_out + dim_in,
                            dim_out,
                            time_emb_dim=time_dim,
                            classes_emb_dim=cond_dim,
                        ),
                        ResnetBlock(
                            dim_out + dim_in,
                            dim_out,
                            time_emb_dim=time_dim,
                            classes_emb_dim=cond_dim,
                        ),
                        (
                            Residual(
                                PreNorm(
                                    dim_out,
                                    Attention(
                                        dim_out,
                                        heads=attn_heads,
                                        dim_head=attn_dim_head,
                                    ),
                                )
                            )
                            if has_attn
                            else nn.Identity()
                        ),
                        (
                            Upsample(dim_out, dim_in)
                            if not is_last
                            else nn.Conv2d(dim_out, dim_in, 3, padding=1)
                        ),
                    ]
                )
            )

        self.final_res_block = ResnetBlock(
            dim * 2, dim, time_emb_dim=time_dim, classes_emb_dim=cond_dim
        )
        self.final_conv = nn.Conv2d(dim, channels, 1)

    def _cond(
        self,
        codepoint: torch.Tensor,
        style: torch.Tensor,
        cond_drop: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Concatenated ``(codepoint, style)`` conditioning ``(B, cond_dim)``.

        ``cond_drop`` is an optional ``(B,)`` bool mask; where ``True`` the
        entire conditioning is replaced with the learned null embedding (used
        for classifier-free guidance).
        """
        c = torch.cat([self.codepoint_emb(codepoint), style], dim=-1)
        if cond_drop is not None:
            c = torch.where(cond_drop.unsqueeze(-1), self.null_cond, c)
        return c

    def predict_geometry(
        self,
        codepoint: torch.Tensor,
        style: torch.Tensor,
        glyph: torch.Tensor,
        snap: bool = True,
    ) -> torch.Tensor:
        """Predict the five geometry labels (em units) for ``glyph``.

        ``glyph`` is the crop-to-ink normalized glyph ``(B, 1, H, W)`` in
        ``[0, 1]``; ``style`` is the flat ``(B, context_dim)`` style vector.
        """
        cond = self._cond(codepoint, style)
        mode = self.glyph_encoder(glyph)  # (B, mode_dim)
        raw = self.geometry_head(torch.cat([cond, mode], dim=-1))
        out = _decode_geometry(raw)
        if snap:
            idx = GEOMETRY_NAMES.index("descender_depth")
            dd = out[:, idx : idx + 1]
            snapped = torch.where(
                torch.abs(dd) < DESCENDER_SNAP_EPSILON, torch.zeros_like(dd), dd
            )
            out = torch.cat([out[:, :idx], snapped, out[:, idx + 1 :]], dim=1)
        return out

    def geometry_loss(
        self,
        codepoint: torch.Tensor,
        style: torch.Tensor,
        glyph: torch.Tensor,
        gt_geometry: torch.Tensor,
    ) -> torch.Tensor:
        """Per-label variance-normalised MSE in em units (see ``fontid``)."""
        pred = self.predict_geometry(codepoint, style, glyph, snap=False)
        se = (pred - gt_geometry) ** 2
        return (se / self.geometry_std.square()).mean()

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        style: torch.Tensor,
        codepoint: torch.Tensor,
        x_self_cond: torch.Tensor | None = None,
        cond_drop: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)
        c = self._cond(codepoint, style, cond_drop=cond_drop)

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, time_emb=t, class_emb=c)
            h.append(x)
            x = block2(x, time_emb=t, class_emb=c)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, time_emb=t, class_emb=c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb=t, class_emb=c)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, time_emb=t, class_emb=c)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, time_emb=t, class_emb=c)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, time_emb=t, class_emb=c)
        return self.final_conv(x)


# ---------------------------------------------------------------------------
# Diffusion process (training + DDIM sampling), conditioning-agnostic
# ---------------------------------------------------------------------------


class ExemplarDiffusion(nn.Module):
    """Standard DDPM/DDIM process threading ``(style, codepoint)`` to the UNet."""

    def __init__(
        self,
        model: ExemplarConditionalUnet,
        *,
        image_size: int,
        timesteps: int = 250,
        sampling_timesteps: int | None = 50,
        beta_schedule: str = "cosine",
        ddim_sampling_eta: float = 0.0,
        self_condition: bool = False,
        min_snr_gamma: float = 5.0,
        cond_drop_prob: float = 0.0,
        cond_scale: float = 1.0,
    ) -> None:
        super().__init__()
        assert model.channels == model.out_dim
        self.model = model
        self.channels = model.channels
        self.image_size = image_size
        self.self_condition = self_condition
        self.min_snr_gamma = min_snr_gamma
        self.cond_drop_prob = cond_drop_prob
        self.cond_scale = cond_scale

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        self.ddim_sampling_eta = ddim_sampling_eta

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )
        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

    @property
    def device(self):
        return self.betas.device

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_losses(self, x_start, t, style, codepoint, noise=None):
        """Return just the diffusion loss (convenience over ``p_losses_with_x0``)."""
        return self.p_losses_with_x0(x_start, t, style, codepoint, noise)[0]

    def p_losses_with_x0(self, x_start, t, style, codepoint, noise=None):
        """Return ``(diffusion_loss, predicted_x0)`` (see ``fontid``)."""
        noise = default(noise, lambda: torch.randn_like(x_start))
        x = self.q_sample(x_start, t, noise)

        # Classifier-free guidance dropout: the *same* mask is used for the
        # self-conditioning estimate and the main pass.
        cond_drop = (
            torch.rand(x_start.shape[0], device=x_start.device) < self.cond_drop_prob
        )

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                first_pred = self.model(
                    x, t, style, codepoint, cond_drop=cond_drop
                )
                x_self_cond = (
                    self.predict_start_from_noise(x, t, first_pred)
                    .clamp(-1.0, 1.0)
                    .detach()
                )

        pred = self.model(
            x, t, style, codepoint, x_self_cond=x_self_cond, cond_drop=cond_drop
        )
        x0_pred = self.predict_start_from_noise(x, t, pred).detach()

        if self.min_snr_gamma is None or self.min_snr_gamma <= 0:
            loss = F.mse_loss(pred, noise)
        else:
            loss = F.mse_loss(pred, noise, reduction="none")
            loss = loss.mean(dim=(1, 2, 3))
            snr = self.alphas_cumprod[t] / (1.0 - self.alphas_cumprod[t])
            weight = snr.clamp(max=self.min_snr_gamma) / snr
            loss = (loss * weight).mean()

        return loss, unnormalize_to_zero_to_one(x0_pred).clamp(0.0, 1.0)

    def forward(self, img, style, codepoint, times=None):
        return self.forward_with_x0(img, style, codepoint, times)[0]

    def forward_with_x0(self, img, style, codepoint, times=None):
        b = img.shape[0]
        img = normalize_to_neg_one_to_one(img)
        times = default(
            times,
            lambda: torch.randint(
                0, self.num_timesteps, (b,), device=img.device
            ).long(),
        )
        return self.p_losses_with_x0(img, times, style, codepoint)

    @torch.no_grad()
    def sample(self, style, codepoint):
        b = codepoint.shape[0]
        shape = (b, self.channels, self.image_size, self.image_size)
        return self.ddim_sample(style, codepoint, shape)

    @torch.no_grad()
    def ddim_sample(self, style, codepoint, shape):
        b = shape[0]
        device = self.device
        total = self.num_timesteps
        steps = min(self.sampling_timesteps, total)
        eta = self.ddim_sampling_eta

        times = torch.linspace(-1, total - 1, steps=steps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=device)
        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((b,), time, device=device, dtype=torch.long)
            pred_noise = self.model(
                img, time_cond, style, codepoint, x_self_cond=x_start
            )
            if self.cond_scale != 1.0:
                uncond = self.model(
                    img,
                    time_cond,
                    style,
                    codepoint,
                    x_self_cond=x_start,
                    cond_drop=torch.ones(b, dtype=torch.bool, device=device),
                )
                pred_noise = uncond + self.cond_scale * (pred_noise - uncond)
            x_start = self.predict_start_from_noise(img, time_cond, pred_noise)
            x_start = x_start.clamp(-1.0, 1.0)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return unnormalize_to_zero_to_one(img)


class ExemplarDiffusionModel(SaveLoadModel):
    """Facade for the exemplar-conditional diffusion model.

    Owns the style encoder (evidence glyphs -> flat style vector), the per-font
    **memory bank** (one learned flat vector per font instance, warm-started
    from the encoder and coupled to it), and the denoiser/diffusion process.
    """

    def __init__(self, config: ExemplarDiffusionConfig) -> None:
        super().__init__()
        if config.num_codepoints <= 0:
            raise ValueError("ExemplarDiffusionConfig.num_codepoints must be set.")
        if config.num_instances <= 0:
            raise ValueError("ExemplarDiffusionConfig.num_instances must be set.")
        self.config = config

        self.style_encoder = StyleEncoder(
            in_channels=config.channels,
            base_channels=config.style_encoder_base_channels,
            context_dim=config.context_dim,
            image_size=config.image_size,
            out_res=config.style_out_res,
        )
        # The per-font memory bank: one flat style vector per font *instance*
        # (roman and italic are distinct instances with distinct evidence, so
        # each gets its own vector).  Warm-started from the encoder, trained by
        # the diffusion gradient, and held near the encoder by the coupling loss.
        self.memory = nn.Embedding(config.num_instances, config.context_dim)

        unet = ExemplarConditionalUnet(
            dim=config.dim,
            num_codepoints=config.num_codepoints,
            context_dim=config.context_dim,
            image_size=config.image_size,
            dim_mults=config.dim_mults,
            channels=config.channels,
            attn_dim_head=config.attn_dim_head,
            attn_heads=config.attn_heads,
            attn_resolutions=config.attn_resolutions,
            self_condition=config.self_condition,
            geometry_std=config.geometry_std,
        )
        self.diffusion = ExemplarDiffusion(
            unet,
            image_size=config.image_size,
            timesteps=config.timesteps,
            sampling_timesteps=config.sampling_timesteps,
            beta_schedule=config.beta_schedule,
            ddim_sampling_eta=config.ddim_sampling_eta,
            self_condition=config.self_condition,
            min_snr_gamma=config.min_snr_gamma,
            cond_drop_prob=config.cond_drop_prob,
            cond_scale=config.cond_scale,
        )

    def encode_style(self, evidence: torch.Tensor) -> torch.Tensor:
        """Encode evidence glyphs ``(B, N, C, H, W)`` to a flat ``(B, context_dim)``
        style vector (the *live* read, used for the coupling loss)."""
        return self.style_encoder(evidence)

    def lookup_style(self, instance_ids: torch.Tensor) -> torch.Tensor:
        """Look up the per-font memory vector(s) ``(B, context_dim)``.

        This is the conditioning used for both training and inference."""
        return self.memory(instance_ids)

    def coupling_loss(self, style_mem: torch.Tensor, style_live: torch.Tensor) -> torch.Tensor:
        """Pull the memory vector and the live encoder read toward each other.

        Symmetric MSE: the memory is nudged toward what the shared encoder reads
        from the glyphs (so it stays exemplar-derived and rare fonts inherit the
        encoder), and the encoder is nudged toward the memory (so it learns to
        produce the diffusion-useful summary).  Without this term the memory is
        just a fontid-style free parameter and the encoder receives no gradient.
        """
        return F.mse_loss(style_mem, style_live.to(style_mem.dtype))

    @torch.no_grad()
    def warm_start_memory(self, instances, probe: str = WARM_START_PROBE) -> None:
        """Initialise the memory bank from the style encoder.

        Each instance's memory row is set to the encoder's flat summary of a
        broad, fixed probe glyph set (``WARM_START_PROBE``), so training starts
        from exemplar-derived style vectors rather than random rows.  This is a
        one-time, cheap pass over the (already materialised) instances.
        """
        probe_cps = [ord(c) for c in probe]
        device = self.memory.weight.device
        image_size = self.config.image_size
        self.eval()
        for i, inst in enumerate(instances):
            cps = [cp for cp in probe_cps if cp in inst.codepoints]
            if not cps:
                # Fall back to whatever the instance can draw (rare symbol/guide
                # fonts); the memory still gets *a* style summary.
                cps = inst.codepoints[: self.config.num_evidence_glyphs]
            if not cps:
                continue
            evidence = []
            for cp in cps:
                ev_img, _ = render(
                    inst.font,
                    codepoint=cp,
                    size=image_size,
                    axis_position=inst.axis_position,
                )
                evidence.append(ev_img.unsqueeze(0))  # (1, H, W)
            evidence = torch.stack(evidence).unsqueeze(0)  # (1, N, 1, H, W)
            style = self.style_encoder(evidence.to(device))  # (1, context_dim)
            self.memory.weight.data[i] = style[0].to(self.memory.weight.dtype)

    def forward(
        self, img: torch.Tensor, style: torch.Tensor, codepoint: torch.Tensor
    ) -> torch.Tensor:
        return self.diffusion(img, style, codepoint)

    def forward_with_x0(
        self, img: torch.Tensor, style: torch.Tensor, codepoint: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.diffusion.forward_with_x0(img, style, codepoint)

    @torch.no_grad()
    def sample(self, style: torch.Tensor, codepoint: torch.Tensor) -> torch.Tensor:
        return self.diffusion.sample(style, codepoint)

    def predict_geometry(
        self,
        codepoint: torch.Tensor,
        style: torch.Tensor,
        glyph: torch.Tensor,
        snap: bool = True,
    ) -> torch.Tensor:
        return self.diffusion.model.predict_geometry(
            codepoint, style, glyph, snap=snap
        )

    def geometry_loss(
        self,
        codepoint: torch.Tensor,
        style: torch.Tensor,
        glyph: torch.Tensor,
        gt_geometry: torch.Tensor,
    ) -> torch.Tensor:
        return self.diffusion.model.geometry_loss(
            codepoint, style, glyph, gt_geometry
        )


def build_exemplar_model(config: ExemplarDiffusionConfig) -> ExemplarDiffusionModel:
    """Construct an ``ExemplarDiffusionModel`` from a config."""
    return ExemplarDiffusionModel(config)
