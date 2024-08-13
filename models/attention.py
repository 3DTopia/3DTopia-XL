# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import os
import warnings

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from xformers.ops import memory_efficient_attention, unbind


class MemEffAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.gradient_checkpointing = gradient_checkpointing

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, x, attn_bias, use_reentrant=False)
        else:
            return self._forward(x, attn_bias)

    def _forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_q: int,
        dim_k: int,
        dim_v: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.gradient_checkpointing = gradient_checkpointing

        self.to_q = nn.Linear(dim_q, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim_k, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim_v, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_bias=None) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, q, k, v, attn_bias, use_reentrant=False)
        else:
            return self._forward(q, k, v, attn_bias)

    def _forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_bias=None) -> torch.Tensor:
        # q: [B, N, Cq]
        # k: [B, M, Ck]
        # v: [B, M, Cv]
        # return: [B, N, C]

        B, N, _ = q.shape
        M = k.shape[1]

        q = self.scale * self.to_q(q).reshape(B, N, self.num_heads, self.dim // self.num_heads) # [B, N, nh, C/nh]
        k = self.to_k(k).reshape(B, M, self.num_heads, self.dim // self.num_heads) # [B, M, nh, C/nh]
        v = self.to_v(v).reshape(B, M, self.num_heads, self.dim // self.num_heads) # [B, M, nh, C/nh]

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x