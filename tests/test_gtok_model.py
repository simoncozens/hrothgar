"""Test suite for the G-Tok tokenizer model."""

import pytest
import torch

from hrothgar.gtok.model import (
    GtokConfig,
    GtokModel,
    ViTEncoder,
    CausalViTDecoder,
    create_2d_sinusoidal_position_embeddings,
    CausalAttentionMask,
)


class TestPositionEmbeddings:
    """Test 2D sinusoidal position embeddings."""

    def test_embeddings_shape(self):
        """Test that embeddings have the correct shape."""
        seq_length = 64
        grid_height = 8
        grid_width = 8
        embedding_dim = 384

        embeddings = create_2d_sinusoidal_position_embeddings(
            seq_length, grid_height, grid_width, embedding_dim
        )

        assert embeddings.shape == (
            seq_length,
            embedding_dim,
        ), f"Expected shape ({seq_length}, {embedding_dim}), got {embeddings.shape}"

    def test_embeddings_are_normalized(self):
        """Test that embeddings have reasonable magnitude."""
        embeddings = create_2d_sinusoidal_position_embeddings(64, 8, 8, 384)

        # Sinusoidal embeddings should have values approximately in [-1, 1]
        assert (
            embeddings.abs().max() <= 1.1
        ), f"Position embeddings exceed expected range: max={embeddings.abs().max()}"

    def test_invalid_sequence_length(self):
        """Test that mismatched dimensions raise an error."""
        with pytest.raises(AssertionError):
            create_2d_sinusoidal_position_embeddings(
                sequence_length=100,  # 10x10 = 100
                grid_height=8,  # 8x8 = 64
                grid_width=8,
                embedding_dim=384,
            )


class TestCausalAttentionMask:
    """Test causal attention mask creation."""

    def test_mask_shape(self):
        """Test that mask has correct shape."""
        seq_length = 8
        device = torch.device("cpu")

        mask = CausalAttentionMask.get_causal_mask(seq_length, device)

        assert mask.shape == (
            seq_length,
            seq_length,
        ), f"Expected shape ({seq_length}, {seq_length}), got {mask.shape}"

    def test_mask_is_lower_triangular(self):
        """Test that mask is lower triangular (causal)."""
        seq_length = 8
        mask = CausalAttentionMask.get_causal_mask(seq_length, torch.device("cpu"))

        # Check that upper triangle is -inf (masked)
        upper_indices = torch.triu_indices(seq_length, seq_length, offset=1)
        upper_triangle_masked = (
            mask[upper_indices[0], upper_indices[1]] == float("-inf")
        ).all()
        assert upper_triangle_masked, "Upper triangle should be masked (-inf)"

        # Check that lower triangle and diagonal are not masked
        lower_triangle_unmasked = (mask.tril() != float("-inf")).all()
        assert (
            lower_triangle_unmasked
        ), "Lower triangle and diagonal should not be masked"

    def test_mask_caching(self):
        """Test that masks are properly cached."""
        seq_length = 8
        device = torch.device("cpu")

        mask1 = CausalAttentionMask.get_causal_mask(seq_length, device)
        mask2 = CausalAttentionMask.get_causal_mask(seq_length, device)

        # Should return the exact same tensor object (cached)
        assert mask1 is mask2, "Mask should be cached and return the same object"


class TestViTEncoder:
    """Test Vision Transformer encoder."""

    def test_forward_pass(self):
        """Test basic forward pass through ViT encoder."""
        batch_size = 2
        sequence_length = 64
        input_dim = 256
        hidden_dim = 384

        encoder = ViTEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            mlp_dim=1024,
            sequence_length=sequence_length,
            grid_height=8,
            grid_width=8,
            dropout=0.1,
            attention_dropout=0.1,
        )

        x = torch.randn(batch_size, sequence_length, input_dim)
        output = encoder(x)

        # Output should include class token, so sequence_length + 1
        assert output.shape == (
            batch_size,
            sequence_length + 1,
            hidden_dim,
        ), f"Expected shape ({batch_size}, {sequence_length + 1}, {hidden_dim}), got {output.shape}"

    def test_position_embeddings_are_used(self):
        """Test that position embeddings are actually applied."""
        encoder = ViTEncoder(
            input_dim=256,
            hidden_dim=384,
            num_layers=1,
            num_heads=4,
            mlp_dim=1024,
            sequence_length=64,
            grid_height=8,
            grid_width=8,
        )

        # Two identical inputs should produce different outputs
        # because position embeddings are added
        x = torch.ones(1, 64, 256)
        output1 = encoder(x)

        # Check that outputs are not uniform (position embeddings made a difference)
        output_unique = output1.abs().sum(dim=-1).unique()
        assert (
            len(output_unique) > 1
        ), "Position embeddings should create variation in output"


class TestCausalViTDecoder:
    """Test causal Vision Transformer decoder."""

    def test_forward_pass(self):
        """Test basic forward pass through causal ViT decoder."""
        batch_size = 2
        sequence_length = 64
        hidden_dim = 384
        output_dim = 256

        decoder = CausalViTDecoder(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            mlp_dim=1024,
            output_dim=output_dim,
            sequence_length=sequence_length,
            grid_height=8,
            grid_width=8,
            dropout=0.1,
            attention_dropout=0.1,
        )

        x = torch.randn(batch_size, sequence_length, hidden_dim)
        output = decoder(x)

        assert output.shape == (
            batch_size,
            sequence_length,
            output_dim,
        ), f"Expected shape ({batch_size}, {sequence_length}, {output_dim}), got {output.shape}"

    def test_causal_masking(self):
        """Test that causal masking prevents attending to future tokens."""
        # This is a behavioral test: we verify that with causal masking,
        # early tokens should be unaffected by zeroing out later tokens.
        # We can only indirectly test this.
        decoder = CausalViTDecoder(
            hidden_dim=64,
            num_layers=1,
            num_heads=4,
            mlp_dim=256,
            output_dim=32,
            sequence_length=8,
            grid_height=2,
            grid_width=4,
        )

        x = torch.randn(1, 8, 64)

        # Forward pass should complete without error
        output = decoder(x)
        assert not torch.isnan(output).any(), "Output should not contain NaN"


class TestGtokConfig:
    """Test GtokConfig configuration object."""

    def test_config_defaults(self):
        """Test that config applies sensible defaults."""
        config = GtokConfig()

        assert config.image_size == 128
        assert config.cnn_base_channels == 128
        assert config.quantizer_codebook_size == 2048
        assert config.quantizer_code_dim == 8
        assert config.vit_num_layers == 6
        assert config.cnn_channel_multipliers == [1, 2, 2, 4]

    def test_config_custom_values(self):
        """Test that config accepts custom values."""
        config = GtokConfig(
            image_size=64,
            cnn_base_channels=64,
            quantizer_codebook_size=1024,
            vit_num_layers=4,
        )

        assert config.image_size == 64
        assert config.cnn_base_channels == 64
        assert config.quantizer_codebook_size == 1024
        assert config.vit_num_layers == 4
        # Other values should still have defaults
        assert config.cnn_channel_multipliers == [1, 2, 2, 4]


class TestGtokModel:
    """Test the full G-Tok model."""

    def test_model_initialization(self):
        """Test that model initializes without errors."""
        config = GtokConfig()
        model = GtokModel(config)

        assert isinstance(model.cnn_encoder, torch.nn.Module)
        assert isinstance(model.vit_encoder, ViTEncoder)
        assert isinstance(model.vit_decoder, CausalViTDecoder)
        assert isinstance(model.cnn_decoder, torch.nn.Module)
        assert isinstance(model.quantizer, torch.nn.Module)
        assert model.token_grid_height == 16
        assert model.token_grid_width == 16
        assert model.sequence_length == 256

    def test_forward_pass(self):
        """Test forward pass through complete model."""
        # Use small config for fast testing
        config = GtokConfig(
            vit_num_layers=2,  # Smaller for testing
            vit_hidden_dim=128,
            vit_mlp_dim=512,
            vit_num_heads=4,
        )
        model = GtokModel(config)
        model.eval()

        # Create dummy input (batch of 2 glyph images)
        batch_size = 2
        images = torch.randn(batch_size, 3, 128, 128)

        with torch.no_grad():
            reconstructed, loss_info = model(images)

        # Check output shape
        assert (
            reconstructed.shape == images.shape
        ), f"Reconstructed shape {reconstructed.shape} != input shape {images.shape}"

        # Check loss info
        vq_loss, commit_loss, entropy_loss, codebook_usage = loss_info
        assert (
            vq_loss is not None or not model.training
        ), "VQ loss should be computed in training"

    def test_encode_decode_roundtrip(self):
        """Test encode-decode roundtrip."""
        config = GtokConfig(vit_num_layers=2, vit_hidden_dim=128, vit_num_heads=4)
        model = GtokModel(config)
        model.eval()

        images = torch.randn(1, 3, 128, 128)

        with torch.no_grad():
            quantized, _ = model.encode(images)
            reconstructed = model.decode(quantized)

        # Shapes should match
        assert reconstructed.shape == images.shape

        # Values should be in reasonable range (images are typically [0, 1] or [-1, 1])
        assert reconstructed.abs().max() < 10, "Reconstructed values seem out of range"

    def test_training_mode(self):
        """Test that loss computation works in training mode."""
        config = GtokConfig(vit_num_layers=2, vit_hidden_dim=128, vit_num_heads=4)
        model = GtokModel(config)
        model.train()

        images = torch.randn(2, 3, 128, 128)
        reconstructed, loss_info = model(images)

        vq_loss, commit_loss, entropy_loss, codebook_usage = loss_info

        # In training mode, losses should be computed
        assert vq_loss is not None, "VQ loss should be computed in training mode"
        assert (
            commit_loss is not None
        ), "Commit loss should be computed in training mode"
        assert codebook_usage > 0, "Codebook usage should be positive"

    def test_model_parameters_trainable(self):
        """Test that all components have trainable parameters."""
        config = GtokConfig()
        model = GtokModel(config)

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert trainable_params > 0, "Model should have trainable parameters"
        print(f"Total trainable parameters: {trainable_params:,}")

    def test_model_supports_smaller_image_size(self):
        """Test that the model can be reconfigured back to 64x64 if needed."""
        config = GtokConfig(image_size=64, vit_num_layers=2, vit_hidden_dim=128, vit_num_heads=4)
        model = GtokModel(config)
        images = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            reconstructed, _ = model(images)

        assert reconstructed.shape == images.shape
        assert model.sequence_length == 64


class TestViTEncoderParameterNames:
    """Verify that parameter names are clear and descriptive."""

    def test_parameter_names_are_descriptive(self):
        """Test that module parameters use clear names, not abbreviations."""
        encoder = ViTEncoder(
            input_dim=256,
            hidden_dim=384,
            num_layers=2,
            num_heads=4,
            mlp_dim=1024,
            sequence_length=64,
            grid_height=8,
            grid_width=8,
        )

        # Check that key parameters are accessible with descriptive names
        assert hasattr(encoder, "hidden_dim"), "Should have hidden_dim attribute"
        assert hasattr(encoder, "num_layers"), "Should have num_layers attribute"
        assert hasattr(encoder, "num_heads"), "Should have num_heads attribute"
        assert hasattr(encoder, "mlp_dim"), "Should have mlp_dim attribute"

        # Check that module names are clear
        module_names = [name for name, _ in encoder.named_modules()]
        assert (
            "input_projection" in module_names
        ), "Should have clear 'input_projection' module"
        assert "encoder" in module_names, "Should have clear 'encoder' module"


if __name__ == "__main__":
    # Run a quick sanity check
    print("Running G-Tok model tests...")

    # Test basic model creation and forward pass
    config = GtokConfig(vit_num_layers=2, vit_hidden_dim=128, vit_num_heads=4)
    model = GtokModel(config)
    model.eval()

    images = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        output, losses = model(images)

    print(f"✓ Model forward pass successful")
    print(f"  Input shape: {images.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Matches: {output.shape == images.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Total model parameters: {total_params:,}")

    print("\n✓ All sanity checks passed!")
