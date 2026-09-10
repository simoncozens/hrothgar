"""Factorized (codepoint, font-ID) conditional diffusion — VecFusion's
"incomplete fonts" recipe.

For the in-corpus "fill-in-the-library" task, the font style is a *discrete
name*, not pixels.  We therefore condition the denoiser on two learned lookup
embeddings, both injected into the time embedding:

* one-hot codepoint -> ``g``
* one-hot font style -> ``f``

There is no exemplar encoder and no cross-attention.  The font's style is a
*discrete identity* that is factorized into a family lookup (shared by every
weight of the family, and therefore collapse-proof) plus additive weight and
style (upright/italic) offsets.  This ties a family's members together while
keeping them distinct, and lets the model learn "bold is regular but heavier"
across families rather than memorizing each weight independently.

The key difference from Phase 1's ``(codepoint x ROND)`` single class is the
**factorization**: codepoint and style are separate embeddings, so the model
learns "style -> family + weight + italic" and "codepoint -> content" as
reusable directions and composes them, rather than memorizing each pair.
"""

from __future__ import annotations

from random import random

import torch
import torch.nn.functional as F
from denoising_diffusion_pytorch.classifier_free_guidance import (
    Attention,
    Downsample,
    LinearAttention,
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

from hrothgar.diffusion.config import FontIdDiffusionConfig
from hrothgar.glyph_rendering import GEOMETRY_SPEC
from hrothgar.utils import SaveLoadModel


def _apply_activation(raw: torch.Tensor, activation: str) -> torch.Tensor:
    """Apply the geometry head's per-output activation."""
    return torch.sigmoid(raw) if activation == "sigmoid" else torch.tanh(raw)


def _decode_geometry(raw: torch.Tensor) -> torch.Tensor:
    """Decode ``(B, 5)`` raw head logits to em-unit geometry values."""
    cols = []
    for i, (_, activation, scale) in enumerate(GEOMETRY_SPEC):
        cols.append(_apply_activation(raw[:, i], activation) * scale)
    return torch.stack(cols, dim=1)


class FontIdConditionalUnet(nn.Module):
    """Denoiser UNet conditioned on codepoint and font embeddings.

    Mirrors the library's class-conditional UNet, but with *two* learned
    embeddings concatenated as the "class" conditioning alongside the time
    embedding.
    """

    def __init__(
        self,
        dim: int,
        num_codepoints: int,
        num_families: int,
        num_style_buckets: int = 2,
        dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        channels: int = 1,
        attn_dim_head: int = 32,
        attn_heads: int = 4,
        self_condition: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.out_dim = channels
        self.self_condition = self_condition

        # With self-conditioning the denoiser also receives its own predicted
        # x0 from the previous step, concatenated channel-wise with x.
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
        # Style is a discrete family identity (collapse-proof, shared across a
        # family's weights) plus an italic one-hot, plus a *continuous* weight
        # modulation along a single learned direction.  The linear weight keeps
        # the construction fixed (bold = regular + more along one direction)
        # rather than letting weight change the skeleton.
        self.family_emb = nn.Embedding(num_families, time_dim)
        self.weight_direction = nn.Parameter(torch.randn(time_dim))
        self.style_emb = nn.Embedding(num_style_buckets, time_dim)
        cond_dim = time_dim * 2  # codepoint + style, concatenated
        # A glyph encoder condenses the *specific* generated glyph (the one the
        # denoiser actually produced) into a low-dimensional "mode" vector, so
        # the geometry head can answer "what bbox/advance does THIS glyph need
        # in this font?" rather than trying to guess the construction from the
        # conditioning alone.  The glyph image is the crop-to-ink normalized
        # rendering in [0, 1], so its aspect ratio / em extent are *not* in the
        # image — those magnitudes still come from the font conditioning; the
        # mode vector only disambiguates which construction was drawn.
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
        # Per-(codepoint, font, glyph-mode) geometry regression head: predicts
        # the five em-unit labels (scale_x, scale_y, left_sidebearing,
        # baseline_offset, advance) needed to place the generated glyph back on
        # the baseline.
        self.geometry_head = nn.Sequential(
            nn.Linear(cond_dim + self.mode_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, len(GEOMETRY_SPEC)),
        )

        resnet_block = lambda din, dout: ResnetBlock(
            din, dout, time_emb_dim=time_dim, classes_emb_dim=cond_dim
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        resnet_block(dim_in, dim_in),
                        resnet_block(dim_in, dim_in),
                        Residual(
                            PreNorm(
                                dim_in,
                                LinearAttention(
                                    dim_in, heads=attn_heads, dim_head=attn_dim_head
                                ),
                            )
                        ),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = Residual(
            PreNorm(
                mid_dim, Attention(mid_dim, heads=attn_heads, dim_head=attn_dim_head)
            )
        )
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        resnet_block(dim_out + dim_in, dim_out),
                        resnet_block(dim_out + dim_in, dim_out),
                        Residual(
                            PreNorm(
                                dim_out,
                                LinearAttention(
                                    dim_out, heads=attn_heads, dim_head=attn_dim_head
                                ),
                            )
                        ),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.final_res_block = resnet_block(dim * 2, dim)
        self.final_conv = nn.Conv2d(dim, channels, 1)

    def _cond(self, codepoint: torch.Tensor, font_meta: torch.Tensor) -> torch.Tensor:
        """Concatenated codepoint + factorized style conditioning embedding.

        ``font_meta`` is ``(B, 3)`` float32: ``[family_id, weight, style]`` where
        ``weight`` is the normalized scalar (regular = 0) and ``family_id`` /
        ``style`` are integer indices.
        """
        family_id = font_meta[:, 0].long()
        weight = font_meta[:, 1]
        style = font_meta[:, 2].long()
        f = (
            self.family_emb(family_id)
            + weight[:, None] * self.weight_direction[None, :]
            + self.style_emb(style)
        )
        return torch.cat([self.codepoint_emb(codepoint), f], dim=-1)

    def predict_geometry(
        self,
        codepoint: torch.Tensor,
        font_meta: torch.Tensor,
        glyph: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the five geometry labels (em units) for ``glyph``.

        ``glyph`` is the crop-to-ink normalized glyph ``(B, 1, H, W)`` in
        ``[0, 1]`` — either the generated glyph at inference, or the ground-truth
        glyph during training.  The font conditioning supplies the em-scale
        conventions; the glyph disambiguates which construction was drawn.
        """
        cond = self._cond(codepoint, font_meta)  # (B, cond_dim)
        mode = self.glyph_encoder(glyph)  # (B, mode_dim)
        raw = self.geometry_head(torch.cat([cond, mode], dim=-1))
        return _decode_geometry(raw)

    def geometry_loss(
        self,
        codepoint: torch.Tensor,
        font_meta: torch.Tensor,
        glyph: torch.Tensor,
        gt_geometry: torch.Tensor,
    ) -> torch.Tensor:
        """Per-value MSE in em units.

        ``gt_geometry`` is ``(B, 5)`` in em units.  The loss is the mean over the
        batch of the squared error, averaged over the five labels — so each
        label contributes equally regardless of its em magnitude.
        """
        pred = self.predict_geometry(codepoint, font_meta, glyph)  # (B, 5) em units
        return ((pred - gt_geometry) ** 2).mean()

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        codepoint: torch.Tensor,
        font_meta: torch.Tensor,
        x_self_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)
        c = self._cond(codepoint, font_meta)

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


class FontIdDiffusion(nn.Module):
    """Standard DDPM/DDIM process threading ``(codepoint, font_id)`` to the UNet."""

    def __init__(
        self,
        model: FontIdConditionalUnet,
        *,
        image_size: int,
        timesteps: int = 1000,
        sampling_timesteps: int | None = 50,
        beta_schedule: str = "cosine",
        ddim_sampling_eta: float = 0.0,
        self_condition: bool = False,
        min_snr_gamma: float = 5.0,
    ) -> None:
        super().__init__()
        assert model.channels == model.out_dim
        self.model = model
        self.channels = model.channels
        self.image_size = image_size
        self.self_condition = self_condition
        self.min_snr_gamma = min_snr_gamma

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

    def p_losses(self, x_start, t, codepoint, font_meta, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x = self.q_sample(x_start, t, noise)

        # Self-conditioning: ~50% of the time run a first pass under no_grad to
        # estimate x0, then feed it back as a conditioning channel.  This slows
        # training ~25% but improves sample quality.
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                first_pred = self.model(x, t, codepoint, font_meta)
                x_self_cond = (
                    self.predict_start_from_noise(x, t, first_pred)
                    .clamp(-1.0, 1.0)
                    .detach()
                )

        pred = self.model(x, t, codepoint, font_meta, x_self_cond=x_self_cond)

        if self.min_snr_gamma is None or self.min_snr_gamma <= 0:
            return F.mse_loss(pred, noise)

        # Min-SNR weighting for the noise (eps) prediction objective:
        #   w(t) = min(SNR(t), gamma) / SNR(t) = min(gamma / SNR(t), 1)
        # (raw min(SNR, gamma) is the signal/x0-prediction weight; the noise
        # objective divides by SNR).  This downweights the high-SNR (low-noise)
        # steps, whose noise-prediction gradient is ill-conditioned.
        loss = F.mse_loss(pred, noise, reduction="none")
        loss = loss.mean(dim=(1, 2, 3))  # (B,) per-sample MSE
        snr = self.alphas_cumprod[t] / (1.0 - self.alphas_cumprod[t])  # (B,)
        weight = snr.clamp(max=self.min_snr_gamma) / snr
        return (loss * weight).mean()

    def forward(self, img, codepoint, font_meta, times=None):
        b = img.shape[0]
        img = normalize_to_neg_one_to_one(img)
        times = default(
            times,
            lambda: torch.randint(
                0, self.num_timesteps, (b,), device=img.device
            ).long(),
        )
        return self.p_losses(img, times, codepoint, font_meta)

    @torch.no_grad()
    def sample(self, codepoint, font_meta):
        b = codepoint.shape[0]
        shape = (b, self.channels, self.image_size, self.image_size)
        return self.ddim_sample(codepoint, font_meta, shape)

    @torch.no_grad()
    def ddim_sample(self, codepoint, font_meta, shape):
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
                img, time_cond, codepoint, font_meta, x_self_cond=x_start
            )
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


class FontIdDiffusionModel(SaveLoadModel):
    """Facade for the factorized (codepoint, font-ID) diffusion model."""

    def __init__(self, config: FontIdDiffusionConfig) -> None:
        super().__init__()
        if config.num_codepoints <= 0 or config.num_families <= 0:
            raise ValueError(
                "FontIdDiffusionConfig requires num_codepoints and num_families."
            )
        self.config = config

        unet = FontIdConditionalUnet(
            dim=config.dim,
            num_codepoints=config.num_codepoints,
            num_families=config.num_families,
            num_style_buckets=config.num_style_buckets,
            dim_mults=config.dim_mults,
            channels=config.channels,
            attn_dim_head=config.attn_dim_head,
            attn_heads=config.attn_heads,
            self_condition=config.self_condition,
        )
        self.diffusion = FontIdDiffusion(
            unet,
            image_size=config.image_size,
            timesteps=config.timesteps,
            sampling_timesteps=config.sampling_timesteps,
            beta_schedule=config.beta_schedule,
            ddim_sampling_eta=config.ddim_sampling_eta,
            self_condition=config.self_condition,
            min_snr_gamma=config.min_snr_gamma,
        )

    def forward(
        self, img: torch.Tensor, codepoint: torch.Tensor, font_meta: torch.Tensor
    ) -> torch.Tensor:
        return self.diffusion(img, codepoint, font_meta)

    @torch.no_grad()
    def sample(self, codepoint: torch.Tensor, font_meta: torch.Tensor) -> torch.Tensor:
        return self.diffusion.sample(codepoint, font_meta)

    def predict_geometry(
        self, codepoint: torch.Tensor, font_meta: torch.Tensor, glyph: torch.Tensor
    ) -> torch.Tensor:
        """Predict the five geometry labels (em units) for ``glyph``."""
        return self.diffusion.model.predict_geometry(codepoint, font_meta, glyph)

    def geometry_loss(
        self,
        codepoint: torch.Tensor,
        font_meta: torch.Tensor,
        glyph: torch.Tensor,
        gt_geometry: torch.Tensor,
    ) -> torch.Tensor:
        """Per-value MSE in em units between predicted and target geometry labels."""
        return self.diffusion.model.geometry_loss(
            codepoint, font_meta, glyph, gt_geometry
        )


def build_fontid_model(config: FontIdDiffusionConfig) -> FontIdDiffusionModel:
    """Construct a ``FontIdDiffusionModel`` from a config."""
    return FontIdDiffusionModel(config)
