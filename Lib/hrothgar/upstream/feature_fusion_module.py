import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleAttentionBlock(nn.Module):
    def __init__(self, z_channel, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = z_channel // n_heads
        self.z_channel = z_channel

        self.q_proj = nn.Linear(z_channel, z_channel)
        self.k_proj = nn.Linear(z_channel, z_channel)
        self.v_proj = nn.Linear(z_channel, z_channel)
        self.out_proj = nn.Linear(z_channel, z_channel)
        self.norm = nn.LayerNorm(z_channel)

    def _attention(self, Q, K, V):
        len_q = Q.shape[1]

        Q_ = Q.view(-1, len_q, self.n_heads, self.head_dim).transpose(1, 2)
        K_ = K.view(-1, K.shape[1], self.n_heads, self.head_dim).transpose(1, 2)
        V_ = V.view(-1, V.shape[1], self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q_, K_.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, V_)

        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(-1, len_q, self.z_channel)
        )
        return attn_output

    def forward(self, query_feat, key_feat, value_feat):
        residual = query_feat

        Q = self.q_proj(query_feat)
        K = self.k_proj(key_feat)
        V = self.v_proj(value_feat)

        attn_out = self._attention(Q, K, V)
        out = self.norm(self.out_proj(attn_out) + residual)
        return out


class FeatureFusionModule(nn.Module):
    def __init__(self, z_channel, n_heads=8, n_style_blocks=3):
        super().__init__()
        assert z_channel % n_heads == 0, "z_channel must be divisible by n_heads"

        self.z_channel = z_channel
        self.n_heads = n_heads
        self.head_dim = z_channel // n_heads

        self.style_blocks = nn.ModuleList(
            [StyleAttentionBlock(z_channel, n_heads) for _ in range(n_style_blocks)]
        )

    def forward(self, content_feat, style_feats):
        B, C, H, W = content_feat.shape
        _, n_ref, _, _, _ = style_feats.shape

        content_seq = content_feat.view(B, C, H * W).transpose(1, 2)
        style_seq = (
            style_feats.view(B, n_ref, C, H * W)
            .permute(0, 1, 3, 2)
            .reshape(B, n_ref * H * W, C)
        )  # [B, n_ref*HW, C]

        fused_seq = content_seq
        for block in self.style_blocks:
            fused_seq = block(
                query_feat=fused_seq,  # [B, HW, C]
                key_feat=style_seq,  # [B, n_ref*HW, C]
                value_feat=style_seq,  # [B, n_ref*HW, C]
            )

        fused_feat = fused_seq.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        return fused_feat
