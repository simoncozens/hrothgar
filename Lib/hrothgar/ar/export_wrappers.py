"""Core ML export wrappers for the MaskGIT generation model.

Each wrapper re-wraps a sub-component of ``ARModel`` with a fixed-signature
``forward()`` suitable for ``torch.jit.trace`` → ``coremltools.convert``.

All wrappers avoid tensor→int casts (``aten::Int``) that coremltools
cannot convert, by pre-computing known dimensions and inlining loops.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Patch helpers — avoid tensor→int casts in shared sub-modules
# ---------------------------------------------------------------------------


class _PatchedAttnBlock(nn.Module):
    """``AttnBlock`` without ``int(c)`` or shape-index ops."""

    def __init__(self, original: nn.Module) -> None:
        super().__init__()
        self.norm = original.norm
        self.q = original.q
        self.k = original.k
        self.v = original.v
        self.proj_out = original.proj_out
        c = original.q.weight.shape[0]
        self.register_buffer("scale", torch.tensor(c ** (-0.5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q = self.q(h_).flatten(2).permute(0, 2, 1)
        k = self.k(h_).flatten(2)
        v = self.v(h_).flatten(2)

        w_ = torch.bmm(q, k) * self.scale
        w_ = torch.softmax(w_, dim=2).permute(0, 2, 1)

        h_ = torch.bmm(v, w_).reshape(x.shape)
        return self.proj_out(h_) + x


class _PatchedStyleAttentionBlock(nn.Module):
    """``StyleAttentionBlock`` without shape-index ops."""

    def __init__(self, original: nn.Module, L: int, K_len: int) -> None:
        super().__init__()
        self.L = L
        self.K_len = K_len
        self.n_heads = original.n_heads
        self.head_dim = original.head_dim
        self.z_channel = original.z_channel
        self.q_proj = original.q_proj
        self.k_proj = original.k_proj
        self.v_proj = original.v_proj
        self.out_proj = original.out_proj
        self.norm = original.norm

    def forward(self, query_feat, key_feat, value_feat):
        residual = query_feat
        Q = self.q_proj(query_feat)
        K = self.k_proj(key_feat)
        V = self.v_proj(value_feat)

        Q_ = Q.reshape(1, self.L, self.n_heads, self.head_dim).transpose(1, 2)
        K_ = K.reshape(1, self.K_len, self.n_heads, self.head_dim).transpose(1, 2)
        V_ = V.reshape(1, self.K_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q_, K_.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, V_)
        out = out.transpose(1, 2).contiguous().reshape(1, self.L, self.z_channel)
        return self.norm(self.out_proj(out) + residual)


class _PatchedFeatureFusionModule(nn.Module):
    """``FeatureFusionModule`` without shape-index ops."""

    def __init__(self, original: nn.Module, C: int, H: int, W: int, K: int) -> None:
        super().__init__()
        self.C = C
        self.H = H
        self.W = W
        self.HW = H * W
        self.n_ref_HW = K * H * W
        self.style_blocks = nn.ModuleList([
            _PatchedStyleAttentionBlock(b, self.HW, self.n_ref_HW)
            for b in original.style_blocks
        ])

    def forward(self, content_feat, style_feats):
        content_seq = content_feat.reshape(1, self.C, self.HW).transpose(1, 2)
        style_seq = (
            style_feats.reshape(1, -1, self.C, self.HW)
            .permute(0, 1, 3, 2)
            .reshape(1, self.n_ref_HW, self.C)
        )
        fused_seq = content_seq
        for block in self.style_blocks:
            fused_seq = block(fused_seq, style_seq, style_seq)
        return fused_seq.transpose(1, 2).reshape(1, self.C, self.H, self.W)


# ---------------------------------------------------------------------------
# Wrapper 1: Conditioning Map Builder
# ---------------------------------------------------------------------------


class _EncoderExport(nn.Module):
    """Export the full conditioning-map pipeline.

    Inputs:  ``content_image``  (1, 3, H, W)
             ``style_refs``     (1, K, 3, H, W)
             ``latincore_idx``  (1,) long

    Output:  ``conditioning_map``  (1, 2*C, h, w)
    """

    def __init__(self, model: nn.Module, K: int) -> None:
        super().__init__()
        # Flatten content encoder to avoid for-loop tracing.
        ce = model.content_encoder
        self.ce_conv_in = ce.conv_in
        self.ce_blocks = _flatten_cnn_encoder_blocks(ce)
        self.ce_norm_out = ce.norm_out
        self.ce_conv_out = ce.conv_out

        self.style_encoder = model.style_encoder
        self.K = K
        self.feature_dim = model.config.encoder_feature_dim
        self.grid_h = model.token_grid_height
        self.grid_w = model.token_grid_width

        self.aggregator = _PatchedFeatureFusionModule(
            model.aggregator, C=self.feature_dim,
            H=self.grid_h, W=self.grid_w, K=K,
        )
        self.codepoint_embedding = model.codepoint_embedding

    def forward(self, content_image, style_refs, latincore_idx):
        # Content encoding — explicit unrolled forward.
        h = self.ce_conv_in(content_image)
        for block in self.ce_blocks:
            h = block(h)
        h = self.ce_norm_out(h)
        h = torch.nn.functional.silu(h)
        content_features = self.ce_conv_out(h)

        # Style encoding.
        style_flat = style_refs.squeeze(0)
        style_flat = (style_flat - 0.5) / 0.5
        encoded = self.style_encoder(style_flat)
        style_features = encoded.reshape(
            1, self.K, self.feature_dim, self.grid_h, self.grid_w
        )

        # Aggregation.
        fused = self.aggregator(content_features, style_features)

        # Codepoint embedding.
        codepoint_emb = self.codepoint_embedding(latincore_idx)
        codepoint_map = codepoint_emb[:, :, None, None].expand(
            -1, -1, self.grid_h, self.grid_w
        )

        return torch.cat([codepoint_map, fused], dim=1)


# ---------------------------------------------------------------------------
# Helper: flatten CNN encoder loops
# ---------------------------------------------------------------------------


def _flatten_cnn_encoder_blocks(encoder: nn.Module) -> nn.ModuleList:
    """Flatten a LlamaGen CNN Encoder's nested loops into a linear sequence."""
    blocks: list[nn.Module] = []
    for i_level, block in enumerate(encoder.conv_blocks):
        for i_block in range(encoder.num_res_blocks):
            blocks.append(block.res[i_block])
            if i_block < len(block.attn):
                attn = block.attn[i_block]
                if type(attn).__name__ == "AttnBlock":
                    attn = _PatchedAttnBlock(attn)
                blocks.append(attn)
        if i_level != encoder.num_resolutions - 1:
            blocks.append(block.downsample)
    for mid_block in encoder.mid:
        if type(mid_block).__name__ == "AttnBlock":
            mid_block = _PatchedAttnBlock(mid_block)
        blocks.append(mid_block)
    return nn.ModuleList(blocks)


# ---------------------------------------------------------------------------
# Wrapper 2: MaskGIT Transformer
# ---------------------------------------------------------------------------


class _MaskGITTransformerExport(nn.Module):
    """Export the MaskGIT transformer.

    Patches ``ImgFeatureMapEmbedder`` to replace ``einops.rearrange``
    (which creates ``aten::Int``) with ``flatten+transpose``.

    Inputs:  ``token_indices``     (1, N)
             ``conditioning_map``  (1, C, h, w)
    Output:  ``logits``            (1, N, vocab_size)
    """

    def __init__(self, transformer: nn.Module) -> None:
        super().__init__()
        self.img_feature_proj = transformer.img_feature_embedding.img_feature_proj
        self.tok_embeddings = transformer.tok_embeddings
        self.tok_dropout = transformer.tok_dropout
        self.layers = transformer.layers
        self.norm = transformer.norm
        self.output = transformer.output
        self.img_feature_code_len = transformer.img_feature_code_len
        self.total_len = transformer.img_feature_code_len + transformer.target_token_len
        self.register_buffer("freqs_cis", transformer.freqs_cis.clone())

    def forward(self, token_indices, conditioning_map):
        device = token_indices.device

        # Embed conditioning (patched: flatten+transpose replaces rearrange).
        img_feature = conditioning_map.flatten(2).transpose(1, 2).contiguous()
        img_embeddings = self.img_feature_proj(img_feature)

        # Embed tokens.
        tar_embeddings = self.tok_embeddings(token_indices)

        token_embeddings = torch.cat((img_embeddings, tar_embeddings), dim=1)
        h = self.tok_dropout(token_embeddings)

        # Use full freqs_cis for bidirectional attention.
        freqs = self.freqs_cis.to(device)[:self.total_len]
        mask = torch.zeros(self.total_len, self.total_len, device=device)

        for layer in self.layers:
            h = layer(h, freqs, start_pos=None, mask=mask)

        h = self.norm(h)
        logits = self.output(h).float()
        return logits[:, self.img_feature_code_len:].contiguous()


# ---------------------------------------------------------------------------
# Wrapper 3: Soft Decoder
# ---------------------------------------------------------------------------


class _SoftDecoderExport(nn.Module):
    """Export the G-Tok soft decoder (logits → image).

    Patches ``AttnBlock`` and ``CausalAttention`` to avoid
    ``aten::Int`` from shape indexing and ``einops.rearrange``.

    Inputs:  ``logits``  (1, N, vocab_size)
    Output:  ``images``  (1, 3, H, W)
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        emb = model.codebook_embeddings().detach().clone()
        self.codebook_embeddings = nn.Parameter(emb, requires_grad=False)

        g = model.gtok
        self.quantizer_to_vit_decoder = g.quantizer_to_vit_decoder

        # Patch the ViT decoder to avoid CausalAttention shape/reshape issues.
        self.vit_decoder = _PatchedViTDecoder(g.vit_decoder)

        self.proj_unpatch = g.proj_unpatch

        # Flatten CNN decoder.
        cd = g.cnn_decoder
        self.cd_conv_in = cd.conv_in
        self.cd_mid_blocks = _flatten_cnn_decoder_mid(cd)
        self.cd_up_blocks = _flatten_cnn_decoder_upsample(cd)
        self.cd_norm_out = cd.norm_out
        self.cd_conv_out = cd.conv_out

        self.token_grid_height = g.token_grid_height
        self.token_grid_width = g.token_grid_width
        self.seq_len = g.sequence_length

    def forward(self, logits):
        probs = torch.softmax(logits, dim=-1)
        soft_emb = torch.matmul(probs, self.codebook_embeddings)

        vit_input = self.quantizer_to_vit_decoder(soft_emb)
        vit_tokens = self.vit_decoder(vit_input)
        # vit_tokens: (1, H*W, C_viT) → (1, H, W, C_viT) → (1, C_viT, H, W)
        vit_tokens = vit_tokens.reshape(
            1, self.token_grid_height, self.token_grid_width, -1
        ).permute(0, 3, 1, 2)
        cnn_input = self.proj_unpatch(vit_tokens)

        # CNN decoder — explicit unrolled forward.
        h = self.cd_conv_in(cnn_input)
        for block in self.cd_mid_blocks:
            h = block(h)
        for block in self.cd_up_blocks:
            h = block(h)
        h = self.cd_norm_out(h)
        h = torch.nn.functional.silu(h)
        return self.cd_conv_out(h)


# ---------------------------------------------------------------------------
# Helpers: flatten CNN decoder & patch ViT decoder
# ---------------------------------------------------------------------------


class _PatchedViTDecoder(nn.Module):
    """Patched ``ViTDecoder`` avoiding shape-index + einops in ``CausalAttention``."""

    def __init__(self, original: nn.Module) -> None:
        super().__init__()
        self.de_pos_embedding = original.de_pos_embedding
        self.layers = original.transformer.layers
        self.norm = original.transformer.norm
        first_attn = self.layers[0][0].fn
        self.n_heads = first_attn.heads
        self.dim_head = first_attn.to_qkv.in_features // first_attn.heads
        # Pre-compute the sequence length from the position embedding.
        self.seq_len = original.de_pos_embedding.shape[1]

    def forward(self, token):
        N = self.seq_len
        x = token + self.de_pos_embedding[:, :N]
        for attn_wrapper, ff_wrapper in self.layers:
            attn = attn_wrapper.fn
            ff = ff_wrapper.fn
            # CausalAttention inlined (no shape extraction, no einops).
            qkv = attn.to_qkv(attn_wrapper.norm(x)).chunk(3, dim=-1)
            qkv_shaped = [
                t.reshape(1, N, self.n_heads, self.dim_head).transpose(1, 2)
                for t in qkv
            ]
            q, k, v = qkv_shaped

            scores = torch.matmul(q, k.transpose(-1, -2)) * attn.scale
            causal = torch.tril(torch.ones(N, N, device=x.device)).bool()
            scores = scores.masked_fill(~causal, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            out = torch.matmul(weights, v)
            out = out.transpose(1, 2).contiguous().reshape(1, N, -1)
            x = attn.to_out(out) + x

            x = ff(ff_wrapper.norm(x)) + x

        return self.norm(x)


def _flatten_cnn_decoder_mid(decoder: nn.Module) -> nn.ModuleList:
    """Flatten CNN decoder mid blocks, patching AttnBlock."""
    blocks: list[nn.Module] = []
    for mid_block in decoder.mid:
        if type(mid_block).__name__ == "AttnBlock":
            mid_block = _PatchedAttnBlock(mid_block)
        blocks.append(mid_block)
    return nn.ModuleList(blocks)


def _flatten_cnn_decoder_upsample(decoder: nn.Module) -> nn.ModuleList:
    """Flatten CNN decoder upsample blocks, patching AttnBlock."""
    blocks: list[nn.Module] = []
    for i_level, block in enumerate(decoder.conv_blocks):
        for i_block in range(decoder.num_res_blocks + 1):
            blocks.append(block.res[i_block])
            if i_block < len(block.attn):
                attn = block.attn[i_block]
                if type(attn).__name__ == "AttnBlock":
                    attn = _PatchedAttnBlock(attn)
                blocks.append(attn)
        if i_level != decoder.num_resolutions - 1:
            blocks.append(block.upsample)
    return nn.ModuleList(blocks)


__all__ = [
    "_EncoderExport",
    "_MaskGITTransformerExport",
    "_SoftDecoderExport",
]
