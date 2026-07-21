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


class StyleAttentionPool(nn.Module):
    """Pool per-position style features into a small set of global style tokens.

    A set of learned query tokens attend to all style positions across all
    reference glyphs via multi-head cross-attention.  The output is a fixed-size
    set of global style tokens that summarise the style feature map.

    This replaces per-position cross-attention between content and style: every
    content position attends to the *same* small set of pooled tokens, giving
    the model a globally-consistent style signal without the spatial aliasing
    problems of per-position matching.
    """

    def __init__(self, z_channel: int, n_tokens: int = 16, n_heads: int = 8):
        super().__init__()
        assert z_channel % n_heads == 0, "z_channel must be divisible by n_heads"

        self.z_channel = z_channel
        self.n_tokens = n_tokens
        self.n_heads = n_heads
        self.head_dim = z_channel // n_heads

        # Learned query tokens — one set shared across all batch items.
        self.query_tokens = nn.Parameter(torch.randn(1, n_tokens, z_channel) * 0.02)

        self.q_proj = nn.Linear(z_channel, z_channel)
        self.k_proj = nn.Linear(z_channel, z_channel)
        self.v_proj = nn.Linear(z_channel, z_channel)
        self.out_proj = nn.Linear(z_channel, z_channel)
        self.norm = nn.LayerNorm(z_channel)

    def forward(self, style_seq: torch.Tensor) -> torch.Tensor:
        """Attend learned queries to all style positions.

        Args:
            style_seq: ``(B, N_style, C)`` — flattened style features
                from all reference glyphs.

        Returns:
            ``(B, n_tokens, C)`` pooled global style tokens.
        """
        B, N_style, C = style_seq.shape
        assert C == self.z_channel, (
            f"Channel mismatch: style_seq has {C}, pool expects {self.z_channel}"
        )

        # Expand learned queries to batch.
        queries = self.query_tokens.expand(B, -1, -1)  # (B, n_tokens, C)

        Q = self.q_proj(queries)        # (B, n_tokens, C)
        K = self.k_proj(style_seq)       # (B, N_style, C)
        V = self.v_proj(style_seq)       # (B, N_style, C)

        # Multi-head attention.
        Q_ = Q.view(B, self.n_tokens, self.n_heads, self.head_dim).transpose(1, 2)
        K_ = K.view(B, N_style, self.n_heads, self.head_dim).transpose(1, 2)
        V_ = V.view(B, N_style, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q_, K_.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V_)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(B, self.n_tokens, C)
        )

        pooled = self.norm(self.out_proj(attn_output) + queries)
        return pooled  # (B, n_tokens, C)


class FeatureFusionModule(nn.Module):
    def __init__(
        self,
        z_channel,
        n_heads=8,
        n_style_blocks=3,
        n_style_tokens: int = 0,
    ):
        super().__init__()
        assert z_channel % n_heads == 0, "z_channel must be divisible by n_heads"

        self.z_channel = z_channel
        self.n_heads = n_heads
        self.head_dim = z_channel // n_heads
        self.n_style_tokens = n_style_tokens

        if n_style_tokens > 0:
            self.style_pool = StyleAttentionPool(
                z_channel, n_tokens=n_style_tokens, n_heads=n_heads
            )
        else:
            self.style_pool = None

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

        # Pool style features into global tokens (if enabled).
        if self.style_pool is not None:
            style_kv = self.style_pool(style_seq)  # [B, n_tokens, C]
        else:
            style_kv = style_seq  # [B, n_ref*HW, C]  (legacy per-position)

        fused_seq = content_seq
        for block in self.style_blocks:
            fused_seq = block(
                query_feat=fused_seq,   # [B, HW, C]
                key_feat=style_kv,      # [B, n_tokens|n_ref*HW, C]
                value_feat=style_kv,
            )

        fused_feat = fused_seq.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        return fused_feat
