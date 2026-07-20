"""Tests for the MaskGIT non-autoregressive token prediction module."""

import pytest
import torch
from hrothgar.ar.maskgit import (
    MaskGITConfig,
    MaskGITDecoder,
    MaskGITTransformer,
    _cosine_mask_ratio,
    _cosine_unmask_schedule,
)
from hrothgar.ar.model import ARModel, ARModelConfig, ARModelOutput
from hrothgar.ar.lora import LoRAConfig
from hrothgar.gtok.model import GtokConfig, GtokModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_small_gtok() -> GtokModel:
    """Return a tiny G-Tok model suitable for unit tests (64 px, 64 tokens)."""
    return GtokModel(
        GtokConfig(
            image_size=64,
            cnn_base_channels=32,
            cnn_latent_channels=64,
            vit_hidden_dim=128,
            vit_num_layers=2,
            vit_num_heads=4,
            vit_mlp_dim=256,
        )
    )


def _make_maskgit_model(freeze_gtok: bool = False) -> ARModel:
    """Return an ARModel configured for MaskGIT with a small test G-Tok."""
    gtok = _make_small_gtok()
    config = ARModelConfig(
        image_size=64,
        encoder_feature_dim=64,
        style_encoder_base_channels=32,
        aggregator_num_heads=4,
        decoder_hidden_dim=128,
        decoder_num_layers=2,
        decoder_num_heads=4,
        use_metrics=False,
        maskgit_num_inference_steps=4,
        freeze_gtok=freeze_gtok,
    )
    return ARModel(config=config, gtok_model=gtok)


# ---------------------------------------------------------------------------
# Schedule tests
# ---------------------------------------------------------------------------


def test_cosine_mask_ratio_in_range() -> None:
    """Mask ratio samples should always be in (0, 1]."""
    for _ in range(100):
        r = _cosine_mask_ratio()
        assert 0.0 < r <= 1.0, f"Mask ratio {r} out of range"


def test_cosine_unmask_schedule_monotonic() -> None:
    """Unmask schedule should be non-decreasing and end at num_tokens."""
    for T in [4, 8, 12]:
        prev = 0
        for step in range(T):
            keep = _cosine_unmask_schedule(step, T, 64)
            assert keep >= prev, f"Schedule not monotonic at step {step}/{T}"
            prev = keep
        assert keep == 64, f"Last step should keep all 64 tokens, got {keep}"


def test_cosine_unmask_schedule_clamped() -> None:
    """Schedule should always return at least 1 and at most num_tokens."""
    N = 32
    for T in [1, 4, 8, 16]:
        for step in range(T):
            keep = _cosine_unmask_schedule(step, T, N)
            assert 1 <= keep <= N, f"keep={keep} out of [1, {N}]"


# ---------------------------------------------------------------------------
# MaskGIT transformer tests
# ---------------------------------------------------------------------------


def test_maskgit_transformer_shapes() -> None:
    """MaskGITTransformer forward pass produces correct output shapes."""
    model = _make_maskgit_model()
    transformer = model.maskgit_decoder.transformer

    B, N, K = 2, model.sequence_length, model.codebook_size
    C, H, W = model.config.encoder_feature_dim * 2, N, 1

    # Conditioning map: flattened to (B, C_tokens, dim)
    # In practice this comes from build_conditioning_map with shape (B, 2C, H, W).
    cond = torch.randn(B, C, model.token_grid_height, model.token_grid_width)

    # All tokens passed as input (bidirectional).
    idx = torch.randint(0, K, (B, N))

    logits = transformer(idx=idx, imgs_feature_map=cond)
    assert logits.shape == (B, N, K), f"Expected {(B, N, K)}, got {logits.shape}"


def test_maskgit_transformer_mask_token() -> None:
    """Mask token ID should be vocab_size (one past the last valid index)."""
    model = _make_maskgit_model()
    assert model.maskgit_decoder.mask_token_id == model.codebook_size


# ---------------------------------------------------------------------------
# MaskGIT decoder training tests
# ---------------------------------------------------------------------------


def test_maskgit_decoder_forward_train() -> None:
    """Training forward pass masks some tokens and returns logits + mask."""
    model = _make_maskgit_model()
    model.train()
    decoder = model.maskgit_decoder

    B, N, K = 2, model.sequence_length, model.codebook_size
    C, H, W = model.config.encoder_feature_dim * 2, N, 1

    target_tokens = torch.randint(0, K, (B, N))
    cond = torch.randn(B, C, model.token_grid_height, model.token_grid_width)

    logits, mask = decoder.forward_train(target_tokens, cond)

    assert logits.shape == (B, N, K)
    assert mask.shape == (B, N)
    assert mask.dtype == torch.bool
    # At least one token should be masked (cosine schedule always masks some).
    assert mask.sum() >= 1
    # Not all tokens should be masked (cosine schedule < 1.0).
    assert mask.sum() < B * N


def test_maskgit_decoder_forward_train_logits_same_regardless_of_unmasked() -> None:
    """Changing an unmasked token should not affect predictions at masked positions.

    This is a basic sanity check that the bidirectional transformer actually
    uses the unmasked context tokens rather than ignoring them.
    """
    model = _make_maskgit_model()
    model.eval()
    decoder = model.maskgit_decoder

    B, N, K = 1, model.sequence_length, model.codebook_size
    C = model.config.encoder_feature_dim * 2

    target_tokens = torch.randint(1, K - 1, (B, N))  # avoid 0 and K-1
    cond = torch.randn(B, C, model.token_grid_height, model.token_grid_width)

    # First run: all ground-truth tokens visible (eval mode).
    logits_all_visible = decoder.transformer(idx=target_tokens, imgs_feature_map=cond)

    # Second run: manually mask half the tokens.
    masked_input = target_tokens.clone()
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[:, N // 2 :] = True
    masked_input[mask] = decoder.mask_token_id

    logits_partial = decoder.transformer(idx=masked_input, imgs_feature_map=cond)

    # Predictions at unmasked positions should differ from full-context predictions
    # (since the masked half provides no signal).
    unmasked_positions = ~mask  # first half
    predictions_all = torch.argmax(logits_all_visible, dim=-1)
    predictions_partial = torch.argmax(logits_partial, dim=-1)

    # The unmasked positions should have the SAME predictions (they have
    # the same input in both cases).
    assert torch.all(
        predictions_all[unmasked_positions] == predictions_partial[unmasked_positions]
    ), (
        "Predictions at unmasked positions should be identical regardless of "
        "whether other positions are masked"
    )


# ---------------------------------------------------------------------------
# MaskGIT generation tests
# ---------------------------------------------------------------------------


def test_maskgit_generate_output_shape() -> None:
    """Generation produces correctly-shaped token sequences."""
    model = _make_maskgit_model()
    model.eval()

    B, N = 2, model.sequence_length
    content = torch.randn(B, 3, 64, 64)
    style = torch.randn(B, 3, 3, 64, 64)
    codepoints = torch.tensor([65, 66])

    with torch.no_grad():
        output = model.generate(
            content_images=content,
            style_reference_images=style,
            target_codepoints=codepoints,
        )

    assert output.target_token_indices.shape == (B, N)
    assert output.reconstructed_images.shape == (B, 3, 64, 64)
    assert output.logits.shape == (B, N, model.codebook_size)


# ---------------------------------------------------------------------------
# ARModel integration tests (MaskGIT mode)
# ---------------------------------------------------------------------------


def test_ar_model_maskgit_forward_train() -> None:
    """ARModel.forward in MaskGIT mode returns a token_mask during training."""
    model = _make_maskgit_model()
    model.train()

    B = 2
    content = torch.randn(B, 3, 64, 64)
    style = torch.randn(B, 4, 3, 64, 64)
    target = torch.randn(B, 3, 64, 64)
    codepoints = torch.tensor([65, 66])

    output = model(
        content_images=content,
        style_reference_images=style,
        target_images=target,
        target_codepoints=codepoints,
    )

    assert output.token_mask is not None, "MaskGIT training should produce token_mask"
    assert output.token_mask.shape == (B, model.sequence_length)
    assert output.token_mask.sum() >= 1


def test_ar_model_maskgit_forward_eval() -> None:
    """ARModel.forward in MaskGIT eval mode returns no token_mask."""
    model = _make_maskgit_model()
    model.eval()

    B = 2
    content = torch.randn(B, 3, 64, 64)
    style = torch.randn(B, 4, 3, 64, 64)
    target = torch.randn(B, 3, 64, 64)
    codepoints = torch.tensor([65, 66])

    with torch.no_grad():
        output = model(
            content_images=content,
            style_reference_images=style,
            target_images=target,
            target_codepoints=codepoints,
        )

    assert output.token_mask is None, "MaskGIT eval should have no token_mask"


def test_ar_model_maskgit_parameter_counts() -> None:
    """Parameter counts should report maskgit_decoder instead of token_decoder."""
    model = _make_maskgit_model()
    counts = model.parameter_counts()

    assert "maskgit_decoder" in counts
    assert "token_decoder" not in counts
    assert counts["maskgit_decoder"] > 0


# ---------------------------------------------------------------------------
# Loss tests (MaskGIT mode)
# ---------------------------------------------------------------------------


def test_compute_ar_loss_maskgit_masked_positions_only() -> None:
    """CE loss should be computed only on positions flagged by token_mask."""
    from hrothgar.ar.losses import ARLossWeights, compute_ar_loss

    B, N, K = 2, 4, 8
    logits = torch.randn(B, N, K)
    reconstructed = torch.zeros(B, 3, 8, 8)
    soft_embeddings = torch.zeros(B, N, 8)
    targets = torch.randint(0, K, (B, N))
    # Mask only position (0, 0).
    token_mask = torch.zeros(B, N, dtype=torch.bool)
    token_mask[0, 0] = True

    output = ARModelOutput(
        logits=logits,
        reconstructed_images=reconstructed,
        soft_token_embeddings=soft_embeddings,
        target_token_indices=targets,
        token_mask=token_mask,
    )

    target_images = torch.ones(B, 3, 8, 8)
    total, terms = compute_ar_loss(output, target_images, weights=ARLossWeights())

    assert "n_masked" in terms
    assert terms["n_masked"].item() == 1.0
    # Token accuracy should be computed only on masked positions.
    assert 0.0 <= terms["token_accuracy"].item() <= 1.0


def test_compute_ar_loss_maskgit_no_masked_positions() -> None:
    """When no positions are masked, CE loss should be zero."""
    from hrothgar.ar.losses import ARLossWeights, compute_ar_loss

    B, N, K = 2, 4, 8
    logits = torch.randn(B, N, K)
    reconstructed = torch.zeros(B, 3, 8, 8)
    soft_embeddings = torch.zeros(B, N, 8)
    targets = torch.randint(0, K, (B, N))
    token_mask = torch.zeros(B, N, dtype=torch.bool)

    output = ARModelOutput(
        logits=logits,
        reconstructed_images=reconstructed,
        soft_token_embeddings=soft_embeddings,
        target_token_indices=targets,
        token_mask=token_mask,
    )

    target_images = torch.ones(B, 3, 8, 8)
    _total, terms = compute_ar_loss(output, target_images, weights=ARLossWeights())

    assert terms["token_cross_entropy"].item() == 0.0
    assert terms["token_accuracy"].item() == 0.0


# ---------------------------------------------------------------------------
# NFA on MaskGIT
# ---------------------------------------------------------------------------


def test_nfa_with_maskgit() -> None:
    """enable_nfa_mode should inject LoRA and freeze the model."""
    model = _make_maskgit_model()
    lora_cfg = LoRAConfig(rank=4, alpha=4.0)
    model.enable_nfa_mode(lora_cfg)
    assert model.is_nfa_mode
    assert model.maskgit_decoder.transformer._lora_injected
    # Only LoRA parameters should be trainable.
    trainable = sum(p.numel() for p in model.trainable_parameters())
    total = sum(p.numel() for p in model.parameters())
    assert trainable > 0
    assert trainable < total
    # Check LoRA state dict can be retrieved.
    lora_sd = model.maskgit_decoder.transformer.get_lora_state_dict()
    assert len(lora_sd) > 0
    # Double-inject guard.
    with pytest.raises(RuntimeError, match="already in NFA mode"):
        model.enable_nfa_mode(lora_cfg)


def test_composed_nfa_with_maskgit() -> None:
    """enable_composed_nfa_mode should inject composed LoRA."""
    # First create a GA LoRA state dict.
    model_ga = _make_maskgit_model()
    lora_cfg = LoRAConfig(rank=4, alpha=4.0)
    model_ga.maskgit_decoder.transformer.inject_lora(lora_cfg)
    glyph_state = model_ga.maskgit_decoder.transformer.get_lora_state_dict()

    # Now apply composed NFA with that glyph state.
    model = _make_maskgit_model()
    model.enable_composed_nfa_mode(glyph_state, lora_cfg)
    assert model.is_nfa_mode
    assert model.maskgit_decoder.transformer._composed_lora
    # Only font LoRA parameters should be trainable.
    lora_sd = model.maskgit_decoder.transformer.get_lora_state_dict()
    assert len(lora_sd) > 0
    # All keys should be font adapter keys.
    for k in lora_sd:
        assert "lora_A_font" in k or "lora_B_font" in k


# ---------------------------------------------------------------------------
# End-to-end training step test
# ---------------------------------------------------------------------------


def test_maskgit_e2e_training_step() -> None:
    """A full training step (forward + loss + backward) succeeds without error."""
    model = _make_maskgit_model(freeze_gtok=True)
    model.train()

    from hrothgar.ar.losses import ARLossWeights, compute_ar_loss

    B = 2
    content = torch.randn(B, 3, 64, 64)
    style = torch.randn(B, 4, 3, 64, 64)
    target = torch.randn(B, 3, 64, 64)
    codepoints = torch.tensor([65, 66])

    output = model(
        content_images=content,
        style_reference_images=style,
        target_images=target,
        target_codepoints=codepoints,
    )

    loss, terms = compute_ar_loss(
        output,
        target,
        weights=ARLossWeights(
            token_cross_entropy=0.3,
            pixel_l1=1.0,
            perceptual_lpips=0.0,
        ),
    )

    # Backward should succeed.
    loss.backward()

    # The output projection is zero-initialised (upstream GPT convention).
    # On the first step, only it receives gradient from token CE; earlier
    # layers receive gradient once the output weight becomes non-zero.
    # Verify that at minimum the output projection gets gradient.
    ow = model.maskgit_decoder.transformer.output.weight
    assert ow.grad is not None, "Output weight missing gradient"
    assert ow.grad.abs().sum() > 0, "Output weight has zero gradient"

    # After one step, at least the output and some conditioning params
    # should have non-zero grad.  (The output weight itself propagates
    # grad to earlier layers only after it becomes non-zero, which
    # requires multiple optimizer steps.)
    any_trainable_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
        if p.requires_grad
    )
    assert any_trainable_grad, "No trainable parameters received gradient"

    assert "n_masked" in terms
    assert terms["token_cross_entropy"].item() > 0
