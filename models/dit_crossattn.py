# A modified version of DiT (Diffusion Transformer) to support directly dealing with 3D primitives with shape of [batch_size, sequence_length, dim_feat]

# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# DiT: https://github.dev/facebookresearch/DiT
# --------------------------------------------------------

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import numpy as np
import math
from itertools import repeat
import collections.abc
from .attention import MemEffCrossAttention, MemEffAttention
from .utils import TimestepEmbedder, Mlp, modulate


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, cross_attn_cond_dim, num_heads, mlp_ratio=4.0, proj_bias=False, gradient_checkpointing=False, **block_kwargs):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.crossattn = MemEffCrossAttention(dim=hidden_size, dim_q=hidden_size, dim_k=cross_attn_cond_dim, dim_v=cross_attn_cond_dim, num_heads=num_heads, qkv_bias=True, proj_bias=proj_bias, attn_drop=0.0, proj_drop=0.0, gradient_checkpointing=gradient_checkpointing, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MemEffAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=True, proj_bias=proj_bias, attn_drop=0.0, proj_drop=0.0, gradient_checkpointing=gradient_checkpointing, **block_kwargs)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, cross_attn_cond, mod_cond):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, x, cross_attn_cond, mod_cond, use_reentrant=False)
        else:
            return self._forward(x, cross_attn_cond, mod_cond)

    def _forward(self, x, cross_attn_cond, mod_cond):
        # cross_attn_cond: conditions that use cross attention to cond, would be image tokens typically [B, L_cond, D_cond]
        # mod_cond: conditions that uses modulation to cond, would be timestep typically [B, D_mod]
        shift_mca, scale_mca, gate_mca, shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod_cond).chunk(9, dim=1)
        x = x + gate_mca.unsqueeze(1) * self.crossattn(modulate(self.norm1(x), shift_mca, scale_mca), cross_attn_cond, cross_attn_cond)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm2(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, seq_length, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim+3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum('bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings
    
    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2)) # B x N x C
        return embed

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        seq_length=2,
        in_channels=4,
        condition_channels=512,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        cond_drop_prob=0.0,
        attn_proj_bias=False,
        learn_sigma=True,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.seq_length = seq_length
        self.num_heads = num_heads
        self.cond_drop_prob = cond_drop_prob
        if self.cond_drop_prob > 0:
            self.null_cond_embedding = nn.Parameter(torch.randn(condition_channels))

        # no need to patchify as prim representation is already patch-wise
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        # self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, condition_channels, num_heads, mlp_ratio=mlp_ratio, proj_bias=attn_proj_bias, gradient_checkpointing=gradient_checkpointing) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, seq_length, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y, precision_dtype=torch.float32, enable_amp=False):
        """
        Forward pass of DiT.
        x: (N, T, D)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x)
        t = self.t_embedder(t)                   # (N, D)
        if self.cond_drop_prob > 0 and self.training:
            drop_mask = torch.rand(y.shape[0], device=y.device) < self.cond_drop_prob
            null_cond_embed = self.null_cond_embedding[None, None, :]
            y = torch.where(drop_mask[:, None, None], null_cond_embed, y)
        with torch.autocast(device_type='cuda', dtype=precision_dtype, enabled=enable_amp):
            for block in self.blocks:
                x = block(x=x, cross_attn_cond=y, mod_cond=t)                      # (N, T, D)
            #TODO: final layer only has timestep conditions, no sure if could be better
            x = self.final_layer(x, t)                # (N, T, D)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale=0.0, precision_dtype=torch.float32, enable_amp=False):
        combined = torch.cat([x, x], dim=0)
        combined_t = torch.cat([t, t], dim=0)
        y_null = self.null_cond_embedding.expand_as(y)
        combined_y = torch.cat([y, y_null], dim=0)
        model_out = self.forward(combined, combined_t, combined_y, precision_dtype, enable_amp)
        eps = model_out
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        return half_eps

class DiTAdditivePosEmb(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        seq_length=2,
        in_channels=4,
        condition_channels=512,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        attn_proj_bias=False,
        learn_sigma=True,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.seq_length = seq_length
        self.num_heads = num_heads

        # no need to patchify as prim representation is already patch-wise
        self.point_emb = PointEmbed(hidden_dim=48, dim=hidden_size)
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        # self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, condition_channels, num_heads, mlp_ratio=mlp_ratio, proj_bias=attn_proj_bias, gradient_checkpointing=gradient_checkpointing) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, seq_length, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y, precision_dtype=torch.float32, enable_amp=False):
        """
        Forward pass of DiT.
        x: (N, T, D)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        point = x[:, :, 1:4]
        point_emb = self.point_emb(point)
        x = self.x_embedder(x) + point_emb
        t = self.t_embedder(t)                   # (N, D)
        with torch.autocast(device_type='cuda', dtype=precision_dtype, enabled=enable_amp):
            for block in self.blocks:
                x = block(x=x, cross_attn_cond=y, mod_cond=t)                      # (N, T, D)
            #TODO: final layer only has timestep conditions, no sure if could be better
            x = self.final_layer(x, t)                # (N, T, D)
        return x