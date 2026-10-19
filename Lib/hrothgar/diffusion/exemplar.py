"""Phase 2: exemplar-conditional diffusion glyph generator.

Phase 1 conditioned on a class id (codepoint x ROND) and, as we found, could not
localize fine style detail — the conditioning was a single global embedding and
the reconstruction objective was swamped.  Phase 2 replaces that with the
VecFusion / latent-diffusion recipe:

* **Style encoder** — a set of *evidence glyphs* (the font's style references)
  is encoded into a coarse spatial feature map ``(context_dim, H', W')`` by
  mean-pooling a shared CNN over the evidence set (permutation-invariant).
* **Cross-attention** — the denoiser UNet's spatial features attend to that
  feature map, so each location of the output can query "what is the style
  *here*?" instead of reading one global vector.
* **Codepoint conditioning** — the target codepoint is a learned embedding
  concatenated with the time embedding (VecFusion's recipe).

The diffusion process is reimplemented here (training + DDIM sampling) rather
than reusing ``GaussianDiffusion``, because that class threads a single
``classes`` tensor and we now thread ``(context, codepoint)``.  The math is the
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
from einops import rearrange
from torch import einsum, nn

from hrothgar.diffusion.config import ExemplarDiffusionConfig
from hrothgar.render import (
    DESCENDER_SNAP_EPSILON,
    GEOMETRY_NAMES,
    GEOMETRY_SPEC,
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


class StyleEncoder(nn.Module):
    """Encode a set of evidence glyphs into a coarse spatial style feature map.

    Each evidence glyph passes through the same CNN, then the results are
    mean-pooled across the evidence set (so the order of evidence glyphs is
    irrelevant).  Output shape: ``(B, context_dim, out_res, out_res)``.
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
        """``evidence`` is ``(B, N, C, H, W)``; returns ``(B, C_ctx, H', W')``."""
        b, n, c, h, w = evidence.shape
        x = evidence.reshape(b * n, c, h, w)
        x = self.net(x)  # (B*N, context_dim, out_res, out_res)
        x = x.reshape(b, n, self.context_dim, self.out_res, self.out_res)
        return x.mean(dim=1)  # (B, context_dim, out_res, out_res)


# ---------------------------------------------------------------------------
# Cross attention
# ---------------------------------------------------------------------------


class CrossAttention(nn.Module):
    """Multi-head cross-attention from UNet tokens to style tokens."""

    def __init__(
        self, query_dim: int, context_dim: int, heads: int = 4, dim_head: int = 32
    ) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim, bias=False)
        # Post-softmax attention weights from the most recent forward pass
        # (detached), for collapse diagnostics.
        self.last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (B, N, query_dim); context: (B, M, context_dim)
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads),
            (q, k, v),
        )
        q = q * self.scale
        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        self.last_attn = attn.detach()
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class CrossAttentionBlock(nn.Module):
    """LayerNorm -> cross-attention -> residual, over a 4D feature map."""

    def __init__(
        self, dim: int, context_dim: int, heads: int = 4, dim_head: int = 32
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, context_dim, heads=heads, dim_head=dim_head)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W); context: (B, M, context_dim)
        b, c, h, w = x.shape
        x_flat = rearrange(x, "b c h w -> b (h w) c")
        out = self.attn(self.norm(x_flat), context)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        return out + x


class _IdentityCross(nn.Module):
    """Placeholder cross-attention for resolutions without attention."""

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return x


# ---------------------------------------------------------------------------
# Denoiser UNet
# ---------------------------------------------------------------------------


class ExemplarConditionalUnet(nn.Module):
    """Denoiser UNet conditioned on a style context map and a codepoint.

    Structure mirrors the library's class-conditional UNet (ResNet blocks with
    time/class embeddings), but adds cross-attention to the style context at a
    chosen set of resolutions.
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

        # Geometry head: predicts the five em-unit labels so the crop-to-ink
        # glyph can be placed back on the baseline.  Conditioning = codepoint
        # embedding + a globally-pooled style summary (mean over the context
        # feature map) + the glyph's own "mode" vector.  Global pooling is the
        # right call here: the geometry labels (scale / advance / lsb) are
        # global properties, unlike the rendering, which needs the spatial
        # context via cross-attention.
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
            nn.Linear(time_dim + context_dim + self.mode_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, len(GEOMETRY_SPEC)),
        )

        attn_res = set(attn_resolutions)
        self._cross_blocks: list[CrossAttentionBlock] = []

        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            res = image_size // (2**ind)
            has_attn = res in attn_res
            cross = (
                CrossAttentionBlock(
                    dim_in, context_dim, heads=attn_heads, dim_head=attn_dim_head
                )
                if has_attn
                else _IdentityCross()
            )
            if has_attn:
                self._cross_blocks.append(cross)
            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            dim_in,
                            dim_in,
                            time_emb_dim=time_dim,
                            classes_emb_dim=time_dim,
                        ),
                        ResnetBlock(
                            dim_in,
                            dim_in,
                            time_emb_dim=time_dim,
                            classes_emb_dim=time_dim,
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
                        cross,
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
            mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=time_dim
        )
        self.mid_attn = Residual(
            PreNorm(
                mid_dim, Attention(mid_dim, heads=attn_heads, dim_head=attn_dim_head)
            )
        )
        self.mid_cross = CrossAttentionBlock(
            mid_dim, context_dim, heads=attn_heads, dim_head=attn_dim_head
        )
        self._cross_blocks.append(self.mid_cross)
        self.mid_block2 = ResnetBlock(
            mid_dim, mid_dim, time_emb_dim=time_dim, classes_emb_dim=time_dim
        )

        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            res = image_size // (2 ** (num_resolutions - 1 - ind))
            has_attn = res in attn_res
            cross = (
                CrossAttentionBlock(
                    dim_out, context_dim, heads=attn_heads, dim_head=attn_dim_head
                )
                if has_attn
                else _IdentityCross()
            )
            if has_attn:
                self._cross_blocks.append(cross)
            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            dim_out + dim_in,
                            dim_out,
                            time_emb_dim=time_dim,
                            classes_emb_dim=time_dim,
                        ),
                        ResnetBlock(
                            dim_out + dim_in,
                            dim_out,
                            time_emb_dim=time_dim,
                            classes_emb_dim=time_dim,
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
                        cross,
                        (
                            Upsample(dim_out, dim_in)
                            if not is_last
                            else nn.Conv2d(dim_out, dim_in, 3, padding=1)
                        ),
                    ]
                )
            )

        self.final_res_block = ResnetBlock(
            dim * 2, dim, time_emb_dim=time_dim, classes_emb_dim=time_dim
        )
        self.final_conv = nn.Conv2d(dim, channels, 1)

    def attention_weights(self) -> list[torch.Tensor]:
        """Return post-softmax cross-attention weights from the last forward.

        One ``(B, heads, nq, M)`` tensor per cross-attention layer (down, mid,
        up), in forward order.  Empty if no forward has run yet.
        """
        ws = [b.attn.last_attn for b in self._cross_blocks]
        return [w for w in ws if w is not None]

    def predict_geometry(
        self,
        codepoint: torch.Tensor,
        context: torch.Tensor,
        glyph: torch.Tensor,
        snap: bool = True,
    ) -> torch.Tensor:
        """Predict the five geometry labels (em units) for ``glyph``.

        ``glyph`` is the crop-to-ink normalized glyph ``(B, 1, H, W)`` in
        ``[0, 1]``; ``context`` is the style feature map ``(B, C, h, w)``.  The
        context is globally pooled, because the geometry labels are global
        properties of the glyph-in-font, not localized detail.
        """
        c = self.codepoint_emb(codepoint)
        style = context.mean(dim=1)  # (B, context_dim), pooled over tokens
        mode = self.glyph_encoder(glyph)  # (B, mode_dim)
        raw = self.geometry_head(torch.cat([c, style, mode], dim=-1))
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
        context: torch.Tensor,
        glyph: torch.Tensor,
        gt_geometry: torch.Tensor,
    ) -> torch.Tensor:
        """Per-label variance-normalised MSE in em units (see ``fontid``)."""
        pred = self.predict_geometry(codepoint, context, glyph, snap=False)
        se = (pred - gt_geometry) ** 2
        return (se / self.geometry_std.square()).mean()

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        context: torch.Tensor,
        codepoint: torch.Tensor,
        x_self_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)
        c = self.codepoint_emb(codepoint)

        h = []
        for block1, block2, attn, cross, downsample in self.downs:
            x = block1(x, time_emb=t, class_emb=c)
            h.append(x)
            x = block2(x, time_emb=t, class_emb=c)
            x = attn(x)
            x = cross(x, context)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, time_emb=t, class_emb=c)
        x = self.mid_attn(x)
        x = self.mid_cross(x, context)
        x = self.mid_block2(x, time_emb=t, class_emb=c)

        for block1, block2, attn, cross, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, time_emb=t, class_emb=c)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, time_emb=t, class_emb=c)
            x = attn(x)
            x = cross(x, context)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, time_emb=t, class_emb=c)
        return self.final_conv(x)


# ---------------------------------------------------------------------------
# Diffusion process (training + DDIM sampling), conditioning-agnostic
# ---------------------------------------------------------------------------


class ExemplarDiffusion(nn.Module):
    """Standard DDPM/DDIM process threading ``(context, codepoint)`` to the UNet."""

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

    def p_losses(self, x_start, t, context, codepoint, noise=None):
        """Return just the diffusion loss (convenience over ``p_losses_with_x0``)."""
        return self.p_losses_with_x0(x_start, t, context, codepoint, noise)[0]

    def p_losses_with_x0(self, x_start, t, context, codepoint, noise=None):
        """Return ``(diffusion_loss, predicted_x0)`` (see ``fontid``)."""
        noise = default(noise, lambda: torch.randn_like(x_start))
        x = self.q_sample(x_start, t, noise)

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                first_pred = self.model(x, t, context, codepoint)
                x_self_cond = (
                    self.predict_start_from_noise(x, t, first_pred)
                    .clamp(-1.0, 1.0)
                    .detach()
                )

        pred = self.model(x, t, context, codepoint, x_self_cond=x_self_cond)
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

    def forward(self, img, context, codepoint, times=None):
        return self.forward_with_x0(img, context, codepoint, times)[0]

    def forward_with_x0(self, img, context, codepoint, times=None):
        b = img.shape[0]
        img = normalize_to_neg_one_to_one(img)
        times = default(
            times,
            lambda: torch.randint(
                0, self.num_timesteps, (b,), device=img.device
            ).long(),
        )
        return self.p_losses_with_x0(img, times, context, codepoint)

    @torch.no_grad()
    def sample(self, context, codepoint):
        b = codepoint.shape[0]
        shape = (b, self.channels, self.image_size, self.image_size)
        return self.ddim_sample(context, codepoint, shape)

    @torch.no_grad()
    def ddim_sample(self, context, codepoint, shape):
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
                img, time_cond, context, codepoint, x_self_cond=x_start
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


class ExemplarDiffusionModel(SaveLoadModel):
    """Facade for the exemplar-conditional diffusion model.

    Owns the style encoder (evidence glyphs -> style context map) plus the
    denoiser/diffusion process, mirroring :class:`FontIdDiffusionModel`'s
    surface so the training loop and generator can treat the two models alike.
    """

    def __init__(self, config: ExemplarDiffusionConfig) -> None:
        super().__init__()
        if config.num_codepoints <= 0:
            raise ValueError("ExemplarDiffusionConfig.num_codepoints must be set.")
        self.config = config

        self.style_encoder = StyleEncoder(
            in_channels=config.channels,
            base_channels=config.style_encoder_base_channels,
            context_dim=config.context_dim,
            image_size=config.image_size,
            out_res=config.style_out_res,
        )
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
        )

    def encode_context(self, evidence: torch.Tensor) -> torch.Tensor:
        """Encode evidence glyphs ``(B, N, C, H, W)`` to style context tokens.

        Returns ``(B, H'*W', context_dim)`` — the style encoder's spatial feature
        map flattened into tokens, which is the form the cross-attention consumes.
        """
        feat = self.style_encoder(evidence)  # (B, context_dim, H', W')
        return rearrange(feat, "b c h w -> b (h w) c")

    def attention_weights(self) -> list[torch.Tensor]:
        """Cross-attention weights from the most recent forward/sample pass."""
        return self.diffusion.model.attention_weights()

    def forward(
        self, img: torch.Tensor, context: torch.Tensor, codepoint: torch.Tensor
    ) -> torch.Tensor:
        return self.diffusion(img, context, codepoint)

    def forward_with_x0(
        self, img: torch.Tensor, context: torch.Tensor, codepoint: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.diffusion.forward_with_x0(img, context, codepoint)

    @torch.no_grad()
    def sample(self, context: torch.Tensor, codepoint: torch.Tensor) -> torch.Tensor:
        return self.diffusion.sample(context, codepoint)

    def predict_geometry(
        self,
        codepoint: torch.Tensor,
        context: torch.Tensor,
        glyph: torch.Tensor,
        snap: bool = True,
    ) -> torch.Tensor:
        return self.diffusion.model.predict_geometry(
            codepoint, context, glyph, snap=snap
        )

    def geometry_loss(
        self,
        codepoint: torch.Tensor,
        context: torch.Tensor,
        glyph: torch.Tensor,
        gt_geometry: torch.Tensor,
    ) -> torch.Tensor:
        return self.diffusion.model.geometry_loss(
            codepoint, context, glyph, gt_geometry
        )


def build_exemplar_model(config: ExemplarDiffusionConfig) -> ExemplarDiffusionModel:
    """Construct an ``ExemplarDiffusionModel`` from a config."""
    return ExemplarDiffusionModel(config)
