# Modified from:
# llamagen/vq_model.py

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from hrothgar.upstream.blocks import *
import numpy as np

@dataclass
class TokenizerModelArgs:
    codebook_embed_num: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0

    mid_ch : int = 128
    z_channels: int = 256
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    dropout_p: float = 0.0

    vit_dim: int = 256
    vit_depth: int = 6
    vit_heads_num: int = 8
    vit_heads_dim: int = 64
    vit_mlp_dim: int = 512

    patch_num: int = 8


from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_size = (grid_size, grid_size) if type(grid_size) != tuple else grid_size
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h) 
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        w = m.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64) -> None:
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)
    
class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for idx in range(depth):
            layer = nn.ModuleList([PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head)),
                                   PreNorm(dim, FeedForward(dim, mlp_dim))])
            self.layers.append(layer)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class CausalAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # [b, heads, n, n]
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        causal_mask = torch.tril(torch.ones(n, n, device=x.device)).bool()
        attn = attn.masked_fill(~causal_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class CausalTransformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            layer = nn.ModuleList([
                PreNorm(dim, CausalAttention(dim, heads=heads, dim_head=dim_head)),
                PreNorm(dim, FeedForward(dim, mlp_dim))
            ])
            self.layers.append(layer)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
    
class ViTEncoder(nn.Module):
    def __init__(self, patch_num:int, dim: int, depth: int, heads: int, mlp_dim: int, dim_head: int = 64) -> None:
        super().__init__()

        en_pos_embedding = get_2d_sincos_pos_embed(dim, (patch_num,patch_num))

        self.en_pos_embedding = nn.Parameter(torch.from_numpy(en_pos_embedding).float().unsqueeze(0), requires_grad=False)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.apply(init_weights)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = x + self.en_pos_embedding
        x = self.transformer(x)
        return x


class ViTDecoder(nn.Module):
    def __init__(self, patch_num: int, dim: int, depth: int, heads: int, mlp_dim: int, dim_head: int = 64) -> None:
        super().__init__()
        
        de_pos_embedding =  get_2d_sincos_pos_embed(dim, (patch_num,patch_num))

        self.transformer = CausalTransformer(dim, depth, heads, dim_head, mlp_dim)
        self.de_pos_embedding = nn.Parameter(torch.from_numpy(de_pos_embedding).float().unsqueeze(0), requires_grad=False)

        self.apply(init_weights)

    def forward(self, token: torch.FloatTensor) -> torch.FloatTensor:
        x = token + self.de_pos_embedding
        x = self.transformer(x)
        return x


class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, ch=128, ch_mult=(1,1,2,2,4), num_res_blocks=2, 
                 norm_type='group', dropout=0.0, z_channels=256):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks  

        self.conv_in = ConvBlock(in_channels, ch, kernel_size=3, stride=1, padding=1,
                                 norm='none', activ="relu", pad_type="zero", dropout=dropout)
        # Downsampling
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down_blocks = nn.ModuleList()
        
        for i_level in range(self.num_resolutions):
            down_block = nn.Module()
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            curr_in_ch = ch*in_ch_mult[i_level]
            curr_out_ch = ch* ch_mult[i_level]
            
            # ResBlocks
            for _ in range(self.num_res_blocks):
                res_block.append(ResBlock(curr_in_ch, curr_out_ch, norm=norm_type, activ="relu", pad_type="zero", dropout=dropout))
                curr_in_ch = curr_out_ch
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttentionBlock(curr_in_ch, norm_type=norm_type))

            down_block.res = res_block
            down_block.attn = attn_block
            
            if i_level != self.num_resolutions-1:
                down_block.downsample = ConvBlock(curr_in_ch, curr_in_ch, kernel_size=3, stride=1, padding=1,
                                       norm=norm_type, activ="relu", pad_type="zero", dropout=dropout,downsample=True)
            
            self.down_blocks.append(down_block)
            
        self.mid = nn.Sequential(
            ResBlock(curr_in_ch, curr_in_ch, norm=norm_type, activ="relu", pad_type="zero", dropout=dropout),
            AttentionBlock(curr_in_ch, norm_type=norm_type),
            ResBlock(curr_in_ch, curr_in_ch, norm=norm_type, activ="relu", pad_type="zero", dropout=dropout)
        )
        
        norm_out = norm_dispatch(norm_type)(num_channels = curr_in_ch)
        self.norm_out = norm_out
        self.conv_out = ConvBlock(curr_in_ch, z_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self,x):
        h  =self.conv_in(x) # initial convolution
        
        # Downsampling:
        for i_level, block in enumerate(self.down_blocks):
            for i_block in range(self.num_res_blocks):
                h = block.res[i_block](h)
                if len(block.attn)>0:
                    h = block.attn[i_block](h)
            
            if i_level != self.num_resolutions - 1:
                h = block.downsample(h)
                
        for mid_block in self.mid:
            h  = mid_block(h)
            
        h = self.norm_out(h)
        h = nonlinearity(h)  
        h = self.conv_out(h) 
        return h
    
class CNNDecoder(nn.Module):
    def __init__(self, z_channels=256, ch=128, ch_mult=(1,1,2,2,4), num_res_blocks=2, 
                 norm_type="group", dropout=0.0, out_channels=3):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        
        ch_mult = list(ch_mult)
        block_in = ch * ch_mult[-1]
        self.conv_in = ConvBlock(z_channels, block_in, kernel_size=3, stride=1, padding=1,
                                 norm=norm_type, activ="relu", pad_type="zero", dropout=dropout)
        # Middle
        self.mid = nn.Sequential(
            ResBlock(block_in, block_in, norm=norm_type, activ="relu", pad_type="zero", dropout=dropout),
            AttentionBlock(block_in, norm_type=norm_type),
            ResBlock(block_in, block_in, norm=norm_type, activ="relu", pad_type="zero", dropout=dropout)
        )
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)): 
            up_block = nn.Module()
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks+1):
                res_block.append(ResBlock(block_in, block_out, norm=norm_type, activ="relu", pad_type="zero", dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions-1:
                    attn_block.append(AttentionBlock(block_in, norm_type=norm_type))
            
            up_block.res = res_block
            up_block.attn = attn_block
            
            if i_level != 0:
                up_block.upsample = ConvBlock(block_in, block_in, kernel_size=3, stride=1, padding=1,
                                              norm=norm_type, activ="relu", pad_type="zero", dropout=dropout, upsample=True)
            self.up_blocks.append(up_block)

        norm_out = norm_dispatch(norm_type)(num_channels=block_in)
        self.norm_out = norm_out
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    @property
    def last_layer(self):
        return self.conv_out.weight
    
    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid(h)
        for i_level, block in enumerate(self.up_blocks):
            for i_block in range(self.num_res_blocks+1):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions-1:
                h = block.upsample(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
    

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, entropy_loss_ratio, l2_norm, show_usage):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage
        
        # Initialize codebook
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        
        if self.show_usage:
            self.register_buffer("codebook_used", torch.zeros(65536, dtype=torch.long))
            
    def forward(self, z):
        # z: [B, N, C]
        z_flattened = z.view(-1, self.e_dim) # [B*N, C]
        
        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight
        
        # Compute distances
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) \
          + torch.sum(embedding ** 2, dim=1) \
          - 2 * torch.matmul(z_flattened, embedding.t())
        
        min_encoding_indices = torch.argmin(d, dim=1) # [B*N]
        
        # Quantize
        z_q = embedding[min_encoding_indices].view(z.shape) # [B, N, C]
        
        perplexity = None
        min_encodings = None
        vq_loss = None
        commit_loss = None
        entropy_loss = None
        codebook_usage = 0
        
        if self.show_usage and self.training:
            cur_len = min_encoding_indices.shape[0]
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = min_encoding_indices
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e
            
        if self.training:
            vq_loss = torch.mean((z_q - z.detach()) ** 2)
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2)
            entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)
            
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        
        return z_q, (vq_loss, commit_loss, entropy_loss, codebook_usage), (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices):
        embedding = F.normalize(self.embedding.weight, p=2, dim=-1) if self.l2_norm else self.embedding.weight
        z_q = embedding[indices]
        return z_q

def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    
    if loss_type != "softmax":
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    
    avg_probs = torch.mean(probs, dim=0)
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = - torch.mean(torch.sum(probs * log_probs, dim=-1))
    
    loss = sample_entropy - avg_entropy
    return loss
            


class Tokenizer(nn.Module):
    def __init__(self, config: TokenizerModelArgs):
        super().__init__()
        self.config = config
        
        self.encoder = CNNEncoder(in_channels=3, ch=config.mid_ch, ch_mult=config.encoder_ch_mult, num_res_blocks=2,
                                 norm_type='group', dropout=config.dropout_p, z_channels=config.z_channels)
        self.decoder = CNNDecoder(z_channels=config.z_channels, ch=config.mid_ch, ch_mult=config.decoder_ch_mult,
                                 num_res_blocks=2, norm_type='group', dropout=config.dropout_p, out_channels=3)

        self.proj_patch = nn.Conv2d(config.z_channels, config.vit_dim, 1)
        self.proj_unpatch = nn.Conv2d(config.vit_dim, config.z_channels, 1)
        self.vit_encoder = ViTEncoder(config.patch_num, config.vit_dim, config.vit_depth, config.vit_heads_num,config.vit_mlp_dim,config.vit_heads_dim)
        self.vit_decoder = ViTDecoder(config.patch_num, config.vit_dim, config.vit_depth, config.vit_heads_num,config.vit_mlp_dim,config.vit_heads_dim)

        self.quantizer = VectorQuantizer(
            n_e=config.codebook_embed_num,
            e_dim=config.codebook_embed_dim,
            beta=config.commit_loss_beta,
            entropy_loss_ratio=config.entropy_loss_ratio,
            l2_norm=config.codebook_l2_norm,
            show_usage=config.codebook_show_usage,
        )
        self.proj_quantize = nn.Linear(config.vit_dim, config.codebook_embed_dim)
        self.proj_dequantize = nn.Linear(config.codebook_embed_dim, config.vit_dim)

    def encode(self, x):
        feat = self.encoder(x)  # [B, C, H, W]
        h, w = feat.shape[2], feat.shape[3]
        
        tokens = self.proj_patch(feat).flatten(2).transpose(1, 2)  # [B, N, D]
        tokens = self.vit_encoder(tokens)
        
        q_tokens_input = self.proj_quantize(tokens)  # [B, N, e_dim]
        quant, emb_loss, info = self.quantizer(q_tokens_input)
        
        return quant, emb_loss, info, (h, w)

    def decode(self, quant, hw):
        h, w = hw
        
        tokens = self.proj_dequantize(quant)  # [B, N, D]
        tokens = self.vit_decoder(tokens)
        
        feat = tokens.transpose(1, 2).view(-1, tokens.shape[-1], h, w)
        feat = self.proj_unpatch(feat)
        
        dec = self.decoder(feat)
        return dec

    def forward(self, x):
        quant, emb_loss, _, hw = self.encode(x)
        dec = self.decode(quant, hw)
        return dec, emb_loss
