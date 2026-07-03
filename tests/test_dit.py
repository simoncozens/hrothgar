"""Tests for the DiT-based glyph generation model."""

import pytest
import torch
from hrothgar.ar.dit import (
    ConditioningEmbedder,
    DiTBlock,
    DiTConfig,
    FinalLayer,
    GlyphDiT,
    NoiseScheduler,
    TimestepEmbedder,
    ddim_sample,
    get_beta_schedule,
    modulate,
)
from hrothgar.ar.losses import GlyphGenLossWeights, compute_glyph_gen_loss
from hrothgar.ar.model import GlyphGenConfig, GlyphGenerator
from hrothgar.gtok.model import GtokConfig, GtokModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_small_gtok() -> GtokModel:
    """Return a tiny G-Tok model for unit tests (128 px, 64 tokens)."""
    return GtokModel(
        GtokConfig(
            image_size=128,
            cnn_base_channels=32,
            cnn_latent_channels=64,
            vit_hidden_dim=128,
            vit_num_layers=2,
            vit_num_heads=4,
            vit_mlp_dim=256,
        )
    )


def _make_dit_model(freeze_gtok: bool = True) -> GlyphGenerator:
    """Return a GlyphGenerator with a small DiT backbone."""
    gtok = _make_small_gtok()
    cfg = GlyphGenConfig(
        image_size=128,
        encoder_feature_dim=64,
        style_encoder_base_channels=32,
        aggregator_num_heads=4,
        dit_hidden_size=128,
        dit_depth=2,
        dit_num_heads=4,
        diffusion_steps=100,
        ddim_steps=10,
        freeze_gtok=freeze_gtok,
    )
    return GlyphGenerator(cfg, gtok_model=gtok)


# ---------------------------------------------------------------------------
# Modulate helper
# ---------------------------------------------------------------------------


def test_modulate_shape() -> None:
    x = torch.randn(2, 16, 64)
    shift = torch.randn(2, 64)
    scale = torch.randn(2, 64)
    out = modulate(x, shift, scale)
    assert out.shape == x.shape


def test_modulate_values() -> None:
    x = torch.ones(2, 4, 8)
    shift = torch.zeros(2, 8)
    scale = torch.zeros(2, 8)
    out = modulate(x, shift, scale)
    assert torch.allclose(out, torch.ones_like(out))


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------


def test_cosine_beta_schedule_shape() -> None:
    betas = get_beta_schedule("squaredcos_cap_v2", 1000)
    assert betas.shape == (1000,)
    assert (betas >= 0).all() and (betas <= 0.999).all()


def test_linear_beta_schedule_shape() -> None:
    betas = get_beta_schedule("linear", 1000)
    assert betas.shape == (1000,)


def test_noise_scheduler_q_sample() -> None:
    betas = get_beta_schedule("squaredcos_cap_v2", 100)
    scheduler = NoiseScheduler(betas)
    x0 = torch.randn(2, 64, 16)
    t = torch.tensor([10, 50])
    xt = scheduler.q_sample(x0, t)
    assert xt.shape == x0.shape


def test_noise_scheduler_predict_x0() -> None:
    betas = get_beta_schedule("squaredcos_cap_v2", 100)
    scheduler = NoiseScheduler(betas)
    x0 = torch.randn(2, 64, 16)
    t = torch.tensor([10, 50])
    noise = torch.randn_like(x0)
    xt = scheduler.q_sample(x0, t, noise=noise)
    x0_pred = scheduler.predict_x0_from_eps(xt, t, noise)
    assert torch.allclose(x0, x0_pred, atol=1e-5)


# ---------------------------------------------------------------------------
# Embedding layers
# ---------------------------------------------------------------------------


def test_timestep_embedder_shape() -> None:
    embedder = TimestepEmbedder(hidden_size=128)
    t = torch.tensor([0, 50, 99])
    out = embedder(t)
    assert out.shape == (3, 128)


def test_conditioning_embedder_shape() -> None:
    embedder = ConditioningEmbedder(
        hidden_size=128,
        codepoint_embedding_dim=64,
        style_feature_dim=64,
        style_dropout_prob=0.0,
    )
    cp = torch.randn(2, 64)
    style = torch.randn(2, 64)
    out = embedder(cp, style)
    assert out.shape == (2, 128)


def test_conditioning_embedder_style_dropout() -> None:
    embedder = ConditioningEmbedder(
        hidden_size=128,
        codepoint_embedding_dim=64,
        style_feature_dim=64,
        style_dropout_prob=1.0,
    )
    embedder.train()
    cp = torch.randn(2, 64)
    style = torch.randn(2, 64)

    # With dropout_prob=1.0, two calls should give different results
    # because the null_style is used (which is fixed), but the codepoint
    # projection differs per input.
    out1 = embedder(cp, style)
    cp2 = torch.randn(2, 64)
    out2 = embedder(cp2, style)
    assert not torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# DiT blocks
# ---------------------------------------------------------------------------


def test_dit_block_shape() -> None:
    block = DiTBlock(hidden_size=128, num_heads=4)
    x = torch.randn(2, 64, 128)
    c = torch.randn(2, 128)
    out = block(x, c)
    assert out.shape == x.shape


def test_final_layer_shape() -> None:
    layer = FinalLayer(hidden_size=128, token_dim=16)
    x = torch.randn(2, 64, 128)
    c = torch.randn(2, 128)
    out = layer(x, c)
    assert out.shape == (2, 64, 16)


# ---------------------------------------------------------------------------
# GlyphDiT model
# ---------------------------------------------------------------------------


def test_glyph_dit_forward() -> None:
    config = DiTConfig(
        hidden_size=128,
        depth=2,
        num_heads=4,
        num_tokens=64,
        token_dim=16,
        codepoint_embedding_dim=64,
        style_feature_dim=64,
    )
    model = GlyphDiT(config)

    B = 2
    x_t = torch.randn(B, 64, 16)
    t = torch.tensor([10, 50])
    cp = torch.randn(B, 64)
    style = torch.randn(B, 64)

    out = model(x_t, t, cp, style)
    assert out.shape == x_t.shape


def test_glyph_dit_cfg_mode() -> None:
    """Classifier-free guidance: null-style output shouldn't raise errors."""
    config = DiTConfig(
        hidden_size=128,
        depth=2,
        num_heads=4,
        num_tokens=64,
        token_dim=16,
        codepoint_embedding_dim=64,
        style_feature_dim=64,
    )
    model = GlyphDiT(config)
    model.eval()

    B = 2
    x_t = torch.randn(B, 64, 16)
    t = torch.tensor([10, 50])
    cp = torch.randn(B, 64)
    style = torch.randn(B, 64)

    # Both should produce valid outputs without errors.
    out_cond = model(x_t, t, cp, style, force_style_drop=False)
    out_uncond = model(x_t, t, cp, style, force_style_drop=True)

    assert out_cond.shape == x_t.shape
    assert out_uncond.shape == x_t.shape
    # Note: adaLN-Zero init means both start as zeros, so they're
    # identical until training.  CFG becomes effective after training.


# ---------------------------------------------------------------------------
# GlyphGenerator (full model) tests
# ---------------------------------------------------------------------------


def test_glyph_generator_forward() -> None:
    model = _make_dit_model()
    model.train()

    B = 2
    content = torch.randn(B, 3, 128, 128)
    style = torch.randn(B, 3, 3, 128, 128)
    target = torch.randn(B, 3, 128, 128)
    cp = torch.tensor([65, 66])

    output = model(content, style, target_images=target, target_codepoints=cp)

    assert output.noise_pred.shape == (B, 64, 16)
    assert output.noise_target.shape == (B, 64, 16)
    assert output.reconstructed_images.shape == target.shape
    assert output.perceptual_recon.shape == target.shape


def test_glyph_generator_generate() -> None:
    model = _make_dit_model()
    model.eval()

    B = 2
    content = torch.randn(B, 3, 128, 128)
    style = torch.randn(B, 3, 3, 128, 128)
    cp = torch.tensor([65, 66])

    with torch.no_grad():
        gen = model.generate(
            content_images=content,
            style_reference_images=style,
            target_codepoints=cp,
        )

    assert gen.reconstructed_images.shape == (B, 3, 128, 128)
    assert gen.token_indices.shape == (B, 64)


def test_glyph_generator_loss_and_backward() -> None:
    model = _make_dit_model()
    model.train()

    from hrothgar.upstream.lpips import LPIPS

    B = 2
    content = torch.randn(B, 3, 128, 128)
    style = torch.randn(B, 3, 3, 128, 128)
    target = torch.randn(B, 3, 128, 128)
    cp = torch.tensor([65, 66])

    output = model(content, style, target_images=target, target_codepoints=cp)

    lpips = LPIPS()
    loss, terms = compute_glyph_gen_loss(
        output, target, weights=GlyphGenLossWeights(), lpips_metric=lpips
    )

    assert loss.ndim == 0
    assert "noise_mse" in terms
    assert "pixel_l1" in terms

    loss.backward()

    # adaLN-Zero init: only output projections get gradient initially.
    # At minimum, the final layer should have gradient.
    final_linear = model.dit.final_layer.linear.weight
    assert final_linear.grad is not None
    assert final_linear.grad.abs().sum() > 0


def test_glyph_generator_parameter_counts() -> None:
    model = _make_dit_model()
    counts = model.parameter_counts()

    assert "dit" in counts
    assert "style_encoder" in counts
    assert "aggregator" in counts
    assert counts["dit"] > 0
    assert counts["total_trainable"] > counts["content_encoder"]


# ---------------------------------------------------------------------------
# DDIM sampling
# ---------------------------------------------------------------------------


def test_ddim_sample_shape() -> None:
    config = DiTConfig(
        hidden_size=128,
        depth=2,
        num_heads=4,
        num_tokens=64,
        token_dim=16,
        codepoint_embedding_dim=64,
        style_feature_dim=64,
        ddim_steps=10,
    )
    model = GlyphDiT(config)
    model.eval()

    betas = get_beta_schedule("squaredcos_cap_v2", 100)
    scheduler = NoiseScheduler(betas)

    B = 2
    cp = torch.randn(B, 64)
    style = torch.randn(B, 64)

    x0 = ddim_sample(
        model=model,
        scheduler=scheduler,
        shape=(B, 64, 16),
        codepoint_emb=cp,
        style_features=style,
        ddim_steps=10,
    )
    assert x0.shape == (B, 64, 16)
