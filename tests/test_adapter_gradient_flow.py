"""Test that TextStyleAdapter gradients flow properly with alignment loss.

This test verifies the fix for the gradient vanishing issue where zero-initialized
output projection prevented learning even with alignment loss enabled.
"""

import torch
from hrothgar.ar.multimodal import TextStyleAdapter, TextStyleAdapterConfig
from hrothgar.ar.losses import ARAdaptationLossWeights, ARAdaptationOutput
import torch.nn.functional as F


def test_adapter_gradient_flow_alignment_loss():
    """Verify that alignment loss produces non-zero gradients through adapter."""
    batch_size = 2
    style_token_count = 256
    style_token_dim = 256
    text_embedding_dim = 512
    
    config = TextStyleAdapterConfig(
        style_token_dim=style_token_dim,
        text_embedding_dim=text_embedding_dim,
        adapter_hidden_dim=256,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
    )
    
    adapter = TextStyleAdapter(config)
    
    # Create synthetic visual and text inputs
    visual_style_tokens = torch.randn(batch_size, style_token_count, style_token_dim, requires_grad=True)
    text_embeddings = torch.randn(batch_size, 32, text_embedding_dim)
    
    # Forward pass through adapter
    multimodal_style_tokens = adapter(visual_style_tokens, text_embeddings)
    
    # Compute alignment loss (MSE between visual and multimodal)
    alignment_loss = F.mse_loss(multimodal_style_tokens, visual_style_tokens)
    
    # Backward pass
    alignment_loss.backward()
    
    # Verify loss is not exactly zero (would indicate gradient vanishing)
    assert alignment_loss.item() > 0.0, \
        f"Alignment loss should be > 0 at initialization (got {alignment_loss.item()})"
    
    # Verify adapter parameters have non-zero gradients
    adapter_params_have_grads = []
    for name, param in adapter.named_parameters():
        if param.grad is not None:
            adapter_params_have_grads.append((name, param.grad.abs().max().item() > 1e-8))
    
    assert len(adapter_params_have_grads) > 0, "Adapter should have gradients"
    
    # At least some parameters should have meaningful gradients
    has_meaningful_grads = any(has_grad for _, has_grad in adapter_params_have_grads)
    assert has_meaningful_grads, \
        f"Adapter should have non-zero gradients. Gradient norms: {adapter_params_have_grads}"
    
    # Specifically check style_out layer (the fixed layer)
    style_out_grad = adapter.style_out.weight.grad
    assert style_out_grad is not None, "style_out.weight should have gradient"
    assert style_out_grad.abs().max().item() > 1e-8, \
        f"style_out.weight should have non-zero gradient (max={style_out_grad.abs().max().item()})"


def test_adapter_initialization_scale():
    """Verify that adapter output initialization produces meaningful perturbations."""
    batch_size = 4
    style_token_count = 256
    style_token_dim = 256
    text_embedding_dim = 512
    
    config = TextStyleAdapterConfig(
        style_token_dim=style_token_dim,
        text_embedding_dim=text_embedding_dim,
        adapter_hidden_dim=256,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
    )
    
    adapter = TextStyleAdapter(config)
    
    # Create synthetic inputs
    visual_style_tokens = torch.randn(batch_size, style_token_count, style_token_dim)
    text_embeddings = torch.randn(batch_size, 32, text_embedding_dim)
    
    # Forward pass
    with torch.no_grad():
        multimodal_style_tokens = adapter(visual_style_tokens, text_embeddings)
    
    # Compute perturbation from visual tokens
    delta = multimodal_style_tokens - visual_style_tokens
    delta_norm = delta.norm(dim=-1).mean()
    visual_norm = visual_style_tokens.norm(dim=-1).mean()
    
    # Delta should be non-zero (allowing small-scale random initialization to produce
    # meaningful gradients without overly disrupting the frozen model's input)
    assert delta_norm.item() > 0.0, \
        f"Delta perturbation should be non-zero (got {delta_norm.item()})"
    
    # Delta should not be massive relative to the visual tokens (would indicate
    # initialization is too aggressive)
    delta_ratio = delta_norm.item() / visual_norm.item()
    assert delta_ratio < 2.0, \
        f"Delta perturbation ({delta_norm.item()}) too large relative to visual norm ({visual_norm.item()}, ratio={delta_ratio})"


if __name__ == "__main__":
    test_adapter_gradient_flow_alignment_loss()
    test_adapter_initialization_scale()
    print("✓ All tests passed")
