import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from utils.typing import *
from .attention import MemEffAttention

class VolumeAttention(nn.Module):
    def __init__(
        self, 
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        groups: int = 32,
        eps: float = 1e-5,
        residual: bool = True,
        skip_scale: float = 1,
    ):
        super().__init__()

        self.residual = residual
        self.skip_scale = skip_scale

        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim, eps=eps, affine=True)
        self.attn = MemEffAttention(dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop)

    def forward(self, x):
        # x: [B, C, H, W, D]
        B, C, H, W, D = x.shape
        
        res = x
        x = self.norm(x)

        x = x.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
        x = self.attn(x)
        x = x.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3).reshape(B, C, H, W, D)

        if self.residual:
            x = (x + res) * self.skip_scale
            
        return x

class DiagonalGaussianDistribution:
    def __init__(self, parameters, deterministic=False):
        # parameters: [B, 2C, ...]
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean, device=self.parameters.device, dtype=self.parameters.dtype)
        
    def sample(self):
        sample = torch.randn(self.mean.shape, device=self.parameters.device, dtype=self.parameters.dtype)
        x = self.mean + self.std * sample
        return x

    def kl(self, other=None, dims=[1, 2, 3, 4]):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=dims)
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=dims,
                )

    def nll(self, sample, dims=[1, 2, 3, 4]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):
        return self.mean
    

class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resample: Literal['default', 'up', 'down'] = 'default',
        groups: int = 32,
        eps: float = 1e-5,
        skip_scale: float = 1, # multiplied to output
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_scale = skip_scale

        self.norm1 = nn.GroupNorm(num_groups=min(groups, in_channels), num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=min(groups, out_channels), num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.act = F.silu

        self.resample = None
        if resample == 'up':
            self.resample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
        elif resample == 'down':
            self.resample = nn.AvgPool3d(kernel_size=2, stride=2)
        
        self.shortcut = nn.Identity()
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True)

    
    def forward(self, x):
        res = x

        x = self.norm1(x)
        x = self.act(x)

        if self.resample:
            res = self.resample(res)
            x = self.resample(x)
        
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x = (x + self.shortcut(res)) * self.skip_scale

        return x

class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        downsample: bool = True,
        skip_scale: float = 1,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
 
        nets = []
        for i in range(num_layers):
            cin = in_channels if i == 0 else out_channels
            nets.append(ResnetBlock(cin, out_channels, skip_scale=skip_scale))
        self.nets = nn.ModuleList(nets)

        self.downsample = None
        if downsample:
            self.downsample = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)

    def _forward(self, x):
        
        for net in self.nets:
            x = net(x)
        
        if self.downsample:
            x = self.downsample(x)
        
        return x


class MidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        attention: bool = True,
        attention_heads: int = 8,
        skip_scale: float = 1,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing

        nets = []
        attns = []
        # first layer
        nets.append(ResnetBlock(in_channels, in_channels, skip_scale=skip_scale))
        # more layers
        for i in range(num_layers):
            nets.append(ResnetBlock(in_channels, in_channels, skip_scale=skip_scale))
            if attention:
                attns.append(VolumeAttention(in_channels, attention_heads, skip_scale=skip_scale))
            else:
                attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)
        
    def forward(self, x):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)

    def _forward(self, x):
        x = self.nets[0](x)
        for attn, net in zip(self.attns, self.nets[1:]):
            if attn:
                x = attn(x)
            x = net(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        upsample: bool = True,
        skip_scale: float = 1,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing

        nets = []
        for i in range(num_layers):
            cin = in_channels if i == 0 else out_channels
            nets.append(ResnetBlock(cin, out_channels, skip_scale=skip_scale))
            
        self.nets = nn.ModuleList(nets)

        self.upsample = None
        if upsample:
            self.upsample = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)

    def _forward(self, x):

        for net in self.nets:
            x = net(x)
            
        if self.upsample:
            x = self.upsample(x)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2 * 16, # double_z
        down_channels: Tuple[int, ...] = (8, 16, 32, 64),
        mid_attention: bool = True,
        layers_per_block: int = 2,
        skip_scale: float = np.sqrt(0.5),
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
      
        # input (first downsample)
        self.conv_in = nn.Conv3d(in_channels, down_channels[0], kernel_size=3, stride=1, padding=1)

        # down
        down_blocks = []
        cout = down_channels[0]
        for i in range(len(down_channels)):
            cin = cout
            cout = down_channels[i]

            down_blocks.append(DownBlock(
                cin, cout, 
                num_layers=layers_per_block, 
                downsample=(i != len(down_channels) - 1), # not final layer
                skip_scale=skip_scale,
                gradient_checkpointing=gradient_checkpointing,
            ))
        self.down_blocks = nn.ModuleList(down_blocks)

        # mid
        self.mid_block = MidBlock(down_channels[-1], attention=mid_attention, skip_scale=skip_scale)

        # last
        self.norm_out = nn.GroupNorm(num_channels=down_channels[-1], num_groups=32, eps=1e-5)
        self.conv_out = nn.Conv3d(down_channels[-1], out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        # x: [B, Cin, H, W, D]

        # first
        x = self.conv_in(x)
        
        # down
        for block in self.down_blocks:
            x = block(x)
        
        # mid
        x = self.mid_block(x)

        # last
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)

        return x
    

class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 1,
        up_channels: Tuple[int, ...] = (64, 32, 16, 8),
        mid_attention: bool = True,
        layers_per_block: int = 2,
        skip_scale: float = np.sqrt(0.5),
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
     
        # first
        self.conv_in = nn.Conv3d(in_channels, up_channels[0], kernel_size=3, stride=1, padding=1)

        # mid
        self.mid_block = MidBlock(up_channels[0], attention=mid_attention, skip_scale=skip_scale)

        # up
        up_blocks = []
        cout = up_channels[0]
        for i in range(len(up_channels)):
            cin = cout
            cout = up_channels[i]
            
            up_blocks.append(UpBlock(
                cin, cout, 
                num_layers=layers_per_block,
                upsample=(i != len(up_channels) - 1), # not final layer
                skip_scale=skip_scale,
                gradient_checkpointing=gradient_checkpointing,
            ))
        self.up_blocks = nn.ModuleList(up_blocks)

        # last (upsample)
        self.norm_out = nn.GroupNorm(num_channels=up_channels[-1], num_groups=min(32, up_channels[-1]), eps=1e-5)
        self.conv_out = nn.ConvTranspose3d(up_channels[-1], out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x: [B, Cin, H, W, D]

        # first
        x = self.conv_in(x)

        # mid
        x = self.mid_block(x)
        
        # up
        for block in self.up_blocks:
            x = block(x)

        # last
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)

        return x


class VAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 16,
        out_channels: int = 1,
        down_channels: Tuple[int, ...] = (16, 32, 64, 128, 256),
        mid_attention: bool = True,
        up_channels: Tuple[int, ...] = (256, 128, 64, 32, 16),
        layers_per_block: int = 2,
        skip_scale: float = np.sqrt(0.5),
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        # encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=2 * latent_channels, # double_z
            down_channels=down_channels,
            mid_attention=mid_attention,
            layers_per_block=layers_per_block,
            skip_scale=skip_scale,
            gradient_checkpointing=gradient_checkpointing,
        )

        # decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_channels=up_channels,
            mid_attention=mid_attention,
            layers_per_block=layers_per_block,
            skip_scale=skip_scale,
            gradient_checkpointing=gradient_checkpointing,
        )

        # quant
        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, 1)

    def encode(self, x):
        x = self.encoder(x)
        x = self.quant_conv(x)
        posterior = DiagonalGaussianDistribution(x)
        return posterior
    
    def decode(self, x):
        x = self.post_quant_conv(x)
        x = self.decoder(x)
        return x

    def forward(self, x, sample=True):
        # x: [B, Cin, H, W, D]

        p = self.encode(x)

        if sample:
            z = p.sample()
        else:
            z = p.mode()
        
        x = self.decode(z)

        return x, p