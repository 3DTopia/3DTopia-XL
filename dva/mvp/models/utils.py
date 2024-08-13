# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""PyTorch utilities"""
from collections import OrderedDict
from itertools import islice
import math
import operator
from typing import Optional, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def xaviermultiplier(m, gain):
    if isinstance(m, nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // m.stride[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // m.stride[0] // m.stride[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] // m.stride[0] // m.stride[1] // m.stride[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * math.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * math.sqrt(2.0 / (n1 + n2))
    else:
        return None

    return std

### normal initialization routines
def xavier_uniform_(m, gain):
    std = xaviermultiplier(m, gain)
    m.weight.data.uniform_(-std * math.sqrt(3.0), std * math.sqrt(3.0))

def initmod(m, gain=1.0, weightinitfunc=xavier_uniform_):
    validclasses = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
    if any([isinstance(m, x) for x in validclasses]):
        weightinitfunc(m, gain)
        if hasattr(m, 'bias') and isinstance(m.bias, torch.Tensor):
            m.bias.data.zero_()

    # blockwise initialization for transposed convs
    if isinstance(m, nn.ConvTranspose2d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]

    if isinstance(m, nn.ConvTranspose3d):
        # hardcoded for stride=2 for now
        m.weight.data[:, :, 0::2, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 0::2, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        m.weight.data[:, :, 1::2, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]

    if isinstance(m, Conv2dWNUB) or isinstance(m, Conv2dWN) or isinstance(m, ConvTranspose2dWN) or \
            isinstance(m, ConvTranspose2dWNUB) or isinstance(m, LinearWN):
        norm = np.sqrt(torch.sum(m.weight.data[:] ** 2))
        m.g.data[:] = norm

def initseq(s):
    for a, b in zip(s[:-1], s[1:]):
        if isinstance(b, nn.ReLU):
            initmod(a, nn.init.calculate_gain('relu'))
        elif isinstance(b, nn.LeakyReLU):
            initmod(a, nn.init.calculate_gain('leaky_relu', b.negative_slope))
        elif isinstance(b, nn.Sigmoid):
            initmod(a)
        elif isinstance(b, nn.Softplus):
            initmod(a)
        else:
            initmod(a)

    initmod(s[-1])

### custom modules
class LinearWN(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearWN, self).__init__(in_features, out_features, bias)
        self.g = nn.Parameter(torch.ones(out_features))
        self.fused = False

    def fuse(self):
        wnorm = torch.sqrt(torch.sum(self.weight ** 2))
        self.weight.data = self.weight.data * self.g.data[:, None] / wnorm
        self.fused = True

    def forward(self, input):
        if self.fused:
            return F.linear(input, self.weight, self.bias)
        else:
            wnorm = torch.sqrt(torch.sum(self.weight ** 2))
            return F.linear(input, self.weight * self.g[:, None] / wnorm, self.bias)

class LinearELR(nn.Module):
    """Linear layer with equalized learning rate from stylegan2"""
    def __init__(self, inch, outch, lrmult=1., norm : Optional[str]=None, act=None):
        super(LinearELR, self).__init__()

        # compute gain from activation fn
        try:
            if isinstance(act, nn.LeakyReLU):
                actgain = nn.init.calculate_gain("leaky_relu", act.negative_slope)
            elif isinstance(act, nn.ReLU):
                actgain = nn.init.calculate_gain("relu")
            else:
                actgain = nn.init.calculate_gain(act)
        except:
            actgain = 1.

        initgain = 1. / math.sqrt(inch)

        self.weight = nn.Parameter(torch.randn(outch, inch) / lrmult)
        self.weightgain = actgain

        if norm == None:
            self.weightgain = self.weightgain * initgain * lrmult

        self.bias = nn.Parameter(torch.full([outch], 0.))

        self.norm : Optional[str] = norm
        self.act = act

        self.fused = False

    def extra_repr(self):
        return 'inch={}, outch={}, norm={}, act={}'.format(
            self.weight.size(1), self.weight.size(0), self.norm, self.act
        )

    def getweight(self):
        if self.fused:
            return self.weight
        else:
            weight = self.weight
            if self.norm is not None:
                if self.norm == "demod":
                    weight = F.normalize(weight, dim=1)
            return weight

    def fuse(self):
        if not self.fused:
            with torch.no_grad():
                self.weight.data = self.getweight() * self.weightgain
        self.fused = True

    def forward(self, x):
        if self.fused:
            weight = self.getweight()

            out = torch.addmm(self.bias[None], x, weight.t())
            if self.act is not None:
                out = self.act(out)
            return out
        else:
            weight = self.getweight()

            if self.act is None:
                out = torch.addmm(self.bias[None], x, weight.t(), alpha=self.weightgain)
                return out
            else:
                out = F.linear(x, weight * self.weightgain, bias=self.bias)
                out = self.act(out)
                return out

class Downsample2d(nn.Module):
    def __init__(self, nchannels, stride=1, padding=0):
        super(Downsample2d, self).__init__()

        self.nchannels = nchannels
        self.stride = stride
        self.padding = padding

        blurkernel = torch.tensor([1., 6., 15., 20., 15., 6., 1.])
        blurkernel = (blurkernel[:, None] * blurkernel[None, :])
        blurkernel = blurkernel / torch.sum(blurkernel)
        blurkernel = blurkernel[None, None, :, :].repeat(nchannels, 1, 1, 1)
        self.register_buffer('kernel', blurkernel)

    def forward(self, x):
        if self.padding == "reflect":
            x = F.pad(x, (3, 3, 3, 3), mode='reflect')
            return F.conv2d(x, weight=self.kernel, stride=self.stride, padding=0, groups=self.nchannels)
        else:
            return F.conv2d(x, weight=self.kernel, stride=self.stride, padding=self.padding, groups=self.nchannels)

class Dilate2d(nn.Module):
    def __init__(self, nchannels, kernelsize, stride=1, padding=0):
        super(Dilate2d, self).__init__()

        self.nchannels = nchannels
        self.kernelsize = kernelsize
        self.stride = stride
        self.padding = padding

        blurkernel = torch.ones((self.kernelsize,))
        blurkernel = (blurkernel[:, None] * blurkernel[None, :])
        blurkernel = blurkernel / torch.sum(blurkernel)
        blurkernel = blurkernel[None, None, :, :].repeat(nchannels, 1, 1, 1)
        self.register_buffer('kernel', blurkernel)

    def forward(self, x):
        return F.conv2d(x, weight=self.kernel, stride=self.stride, padding=self.padding, groups=self.nchannels).clamp(max=1.)

class Conv2dWN(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
            stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dWN, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, True)
        self.g = nn.Parameter(torch.ones(out_channels))
        
    def forward(self, x):
        wnorm = torch.sqrt(torch.sum(self.weight ** 2))
        return F.conv2d(x, self.weight * self.g[:, None, None, None] / wnorm,
                bias=self.bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups)

class Conv2dUB(nn.Conv2d):
    def __init__(self, in_channels, out_channels, height, width, kernel_size,
            stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Conv2dUB, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, False)
        self.bias = nn.Parameter(torch.zeros(out_channels, height, width))
        
    def forward(self, x):
        return F.conv2d(x, self.weight,
                bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups) + self.bias[None, ...]

class Conv2dWNUB(nn.Conv2d):
    def __init__(self, in_channels, out_channels, height, width, kernel_size,
            stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Conv2dWNUB, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, False)
        self.g = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels, height, width))
        
    def forward(self, x):
        wnorm = torch.sqrt(torch.sum(self.weight ** 2))
        return F.conv2d(x, self.weight * self.g[:, None, None, None] / wnorm,
                bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups) + self.bias[None, ...]

def blockinit(k, stride):
    dim = k.ndim - 2
    return k \
            .view(k.size(0), k.size(1), *(x for i in range(dim) for x in (k.size(i+2), 1))) \
            .repeat(1, 1, *(x for i in range(dim) for x in (1, stride))) \
            .view(k.size(0), k.size(1), *(k.size(i+2)*stride for i in range(dim)))

class ConvTranspose1dELR(nn.Module):
    def __init__(self, inch, outch, kernel_size, stride, padding, wsize=0, affinelrmult=1., norm=None, ub=None, act=None):
        super(ConvTranspose1dELR, self).__init__()

        self.inch = inch
        self.outch = outch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.wsize = wsize
        self.norm = norm
        self.ub = ub
        self.act = act

        # compute gain from activation fn
        try:
            if isinstance(act, nn.LeakyReLU):
                actgain = nn.init.calculate_gain("leaky_relu", act.negative_slope)
            elif isinstance(act, nn.ReLU):
                actgain = nn.init.calculate_gain("relu")
            else:
                actgain = nn.init.calculate_gain(act)
        except:
            actgain = 1.

        fan_in = inch * (kernel_size / (stride))

        initgain = stride ** 0.5 if norm == "demod" else 1. / math.sqrt(fan_in)

        self.weightgain = actgain * initgain

        self.weight = nn.Parameter(blockinit(
            torch.randn(inch, outch, kernel_size//self.stride), self.stride))

        if ub is not None:
            self.bias = nn.Parameter(torch.zeros(outch, ub[0]))
        else:
            self.bias = nn.Parameter(torch.zeros(outch))

        if wsize > 0:
            self.affine = LinearELR(wsize, inch, lrmult=affinelrmult)
        else:
            self.affine = None

        self.fused = False

    def extra_repr(self):
        return 'inch={}, outch={}, kernel_size={}, stride={}, padding={}, wsize={}, norm={}, ub={}, act={}'.format(
            self.inch, self.outch, self.kernel_size, self.stride, self.padding, self.wsize, self.norm, self.ub, self.act
        )

    def getweight(self, weight):
        if self.fused:
            return weight
        else:
            if self.norm is not None:
                if self.norm == "demod":
                    if weight.ndim == 5:
                        normdims = [1, 3]
                    else:
                        normdims = [0, 2]

                    if torch.jit.is_scripting():
                        # scripting doesn't support F.normalize(..., dim=list[int])
                        weight = weight / torch.linalg.vector_norm(weight, dim=normdims, keepdim=True)
                    else:
                        weight = F.normalize(weight, dim=normdims)

            weight = weight * self.weightgain

            return weight

    def fuse(self):
        if self.affine is None:
            with torch.no_grad():
                self.weight.data = self.getweight(self.weight)
            self.fused = True

    def forward(self, x, w : Optional[torch.Tensor]=None):
        b = x.size(0)

        if self.affine is not None and w is not None:
            # modulate
            affine = self.affine(w)[:, :, None, None] # [B, inch, 1, 1]
            weight = self.weight * (affine * 0.1 + 1.)
        else:
            weight = self.weight

        weight = self.getweight(weight)

        if self.affine is not None and w is not None: 
            x = x.view(1, b * self.inch, x.size(2))
            weight = weight.view(b * self.inch, self.outch, self.kernel_size)
            groups = b
        else:
            groups = 1

        out = F.conv_transpose1d(x, weight, None,
                stride=self.stride, padding=self.padding, dilation=1, groups=groups)

        if self.affine is not None and w is not None:
            out = out.view(b, self.outch, out.size(2))

        if self.bias.ndim == 1:
            bias = self.bias[None, :, None]
        else:
            bias = self.bias[None, :, :]
        out = out + bias

        if self.act is not None:
            out = self.act(out)

        return out

class ConvTranspose2dELR(nn.Module):
    def __init__(self, inch, outch, kernel_size, stride, padding, wsize=0, affinelrmult=1., norm=None, ub=None, act=None):
        super(ConvTranspose2dELR, self).__init__()

        self.inch = inch
        self.outch = outch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.wsize = wsize
        self.norm = norm
        self.ub = ub
        self.act = act

        # compute gain from activation fn
        try:
            if isinstance(act, nn.LeakyReLU):
                actgain = nn.init.calculate_gain("leaky_relu", act.negative_slope)
            elif isinstance(act, nn.ReLU):
                actgain = nn.init.calculate_gain("relu")
            else:
                actgain = nn.init.calculate_gain(act)
        except:
            actgain = 1.

        fan_in = inch * (kernel_size ** 2 / (stride ** 2))

        initgain = stride if norm == "demod" else 1. / math.sqrt(fan_in)

        self.weightgain = actgain * initgain

        self.weight = nn.Parameter(blockinit(
            torch.randn(inch, outch, kernel_size//self.stride, kernel_size//self.stride), self.stride))

        if ub is not None:
            self.bias = nn.Parameter(torch.zeros(outch, ub[0], ub[1]))
        else:
            self.bias = nn.Parameter(torch.zeros(outch))

        if wsize > 0:
            self.affine = LinearELR(wsize, inch, lrmult=affinelrmult)
        else:
            self.affine = None

        self.fused = False

    def extra_repr(self):
        return 'inch={}, outch={}, kernel_size={}, stride={}, padding={}, wsize={}, norm={}, ub={}, act={}'.format(
            self.inch, self.outch, self.kernel_size, self.stride, self.padding, self.wsize, self.norm, self.ub, self.act
        )

    def getweight(self, weight):
        if self.fused:
            return weight
        else:
            if self.norm is not None:
                if self.norm == "demod":
                    if weight.ndim == 5:
                        normdims = [1, 3, 4]
                    else:
                        normdims = [0, 2, 3]

                    if torch.jit.is_scripting():
                        # scripting doesn't support F.normalize(..., dim=list[int])
                        weight = weight / torch.linalg.vector_norm(weight, dim=normdims, keepdim=True)
                    else:
                        weight = F.normalize(weight, dim=normdims)

            weight = weight * self.weightgain

            return weight

    def fuse(self):
        if self.affine is None:
            with torch.no_grad():
                self.weight.data = self.getweight(self.weight)
            self.fused = True

    def forward(self, x, w : Optional[torch.Tensor]=None):
        b = x.size(0)

        if self.affine is not None and w is not None:
            # modulate
            affine = self.affine(w)[:, :, None, None, None] # [B, inch, 1, 1, 1]
            weight = self.weight * (affine * 0.1 + 1.)
        else:
            weight = self.weight

        weight = self.getweight(weight)

        if self.affine is not None and w is not None: 
            x = x.view(1, b * self.inch, x.size(2), x.size(3))
            weight = weight.view(b * self.inch, self.outch, self.kernel_size, self.kernel_size)
            groups = b
        else:
            groups = 1

        out = F.conv_transpose2d(x, weight, None,
                stride=self.stride, padding=self.padding, dilation=1, groups=groups)

        if self.affine is not None and w is not None:
            out = out.view(b, self.outch, out.size(2), out.size(3))

        if self.bias.ndim == 1:
            bias = self.bias[None, :, None, None]
        else:
            bias = self.bias[None, :, :, :]
        out = out + bias

        if self.act is not None:
            out = self.act(out)

        return out

class ConvTranspose3dELR(nn.Module):
    def __init__(self, inch, outch, kernel_size, stride, padding, wsize=0, affinelrmult=1., norm=None, ub=None, act=None):
        super(ConvTranspose3dELR, self).__init__()

        self.inch = inch
        self.outch = outch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.wsize = wsize
        self.norm = norm
        self.ub = ub
        self.act = act

        # compute gain from activation fn
        try:
            if isinstance(act, nn.LeakyReLU):
                actgain = nn.init.calculate_gain("leaky_relu", act.negative_slope)
            elif isinstance(act, nn.ReLU):
                actgain = nn.init.calculate_gain("relu")
            else:
                actgain = nn.init.calculate_gain(act)
        except:
            actgain = 1.

        fan_in = inch * (kernel_size ** 3 / (stride ** 3))

        initgain = stride ** 1.5 if norm == "demod" else 1. / math.sqrt(fan_in)

        self.weightgain = actgain * initgain

        self.weight = nn.Parameter(blockinit(
            torch.randn(inch, outch, kernel_size//self.stride, kernel_size//self.stride, kernel_size//self.stride), self.stride))

        if ub is not None:
            self.bias = nn.Parameter(torch.zeros(outch, ub[0], ub[1], ub[2]))
        else:
            self.bias = nn.Parameter(torch.zeros(outch))

        if wsize > 0:
            self.affine = LinearELR(wsize, inch, lrmult=affinelrmult)
        else:
            self.affine = None

        self.fused = False

    def extra_repr(self):
        return 'inch={}, outch={}, kernel_size={}, stride={}, padding={}, wsize={}, norm={}, ub={}, act={}'.format(
            self.inch, self.outch, self.kernel_size, self.stride, self.padding, self.wsize, self.norm, self.ub, self.act
        )

    def getweight(self, weight):
        if self.fused:
            return weight
        else:
            if self.norm is not None:
                if self.norm == "demod":
                    if weight.ndim == 5:
                        normdims = [1, 3, 4, 5]
                    else:
                        normdims = [0, 2, 3, 4]

                    if torch.jit.is_scripting():
                        # scripting doesn't support F.normalize(..., dim=list[int])
                        weight = weight / torch.linalg.vector_norm(weight, dim=normdims, keepdim=True)
                    else:
                        weight = F.normalize(weight, dim=normdims)

            weight = weight * self.weightgain

            return weight

    def fuse(self):
        if self.affine is None:
            with torch.no_grad():
                self.weight.data = self.getweight(self.weight)
            self.fused = True

    def forward(self, x, w : Optional[torch.Tensor]=None):
        b = x.size(0)

        if self.affine is not None and w is not None:
            # modulate
            affine = self.affine(w)[:, :, None, None, None, None] # [B, inch, 1, 1, 1, 1]
            weight = self.weight * (affine * 0.1 + 1.)
        else:
            weight = self.weight

        weight = self.getweight(weight)

        if self.affine is not None and w is not None: 
            x = x.view(1, b * self.inch, x.size(2), x.size(3), x.size(4))
            weight = weight.view(b * self.inch, self.outch, self.kernel_size, self.kernel_size, self.kernel_size)
            groups = b
        else:
            groups = 1

        out = F.conv_transpose3d(x, weight, None,
                stride=self.stride, padding=self.padding, dilation=1, groups=groups)

        if self.affine is not None and w is not None:
            out = out.view(b, self.outch, out.size(2), out.size(3), out.size(4))

        if self.bias.ndim == 1:
            bias = self.bias[None, :, None, None, None]
        else:
            bias = self.bias[None, :, :, :, :]
        out = out + bias

        if self.act is not None:
            out = self.act(out)

        return out

class Conv2dELR(nn.Module):
    def __init__(self, inch, outch, kernel_size, stride, padding, wsize=0, affinelrmult=1., norm=None, ub=None, act=None):
        super(Conv2dELR, self).__init__()

        self.inch = inch
        self.outch = outch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.wsize = wsize
        self.norm = norm
        self.ub = ub
        self.act = act

        # compute gain from activation fn
        try:
            if isinstance(act, nn.LeakyReLU):
                actgain = nn.init.calculate_gain("leaky_relu", act.negative_slope)
            elif isinstance(act, nn.ReLU):
                actgain = nn.init.calculate_gain("relu")
            else:
                actgain = nn.init.calculate_gain(act)
        except:
            actgain = 1.

        fan_in = inch * (kernel_size ** 2)

        initgain = 1. if norm == "demod" else 1. / math.sqrt(fan_in)

        self.weightgain = actgain * initgain

        self.weight = nn.Parameter(
            torch.randn(outch, inch, kernel_size, kernel_size))

        if ub is not None:
            self.bias = nn.Parameter(torch.zeros(outch, ub[0], ub[1]))
        else:
            self.bias = nn.Parameter(torch.zeros(outch))

        if wsize > 0:
            self.affine = LinearELR(wsize, inch, lrmult=affinelrmult)
        else:
            self.affine = None

        self.fused = False

    def extra_repr(self):
        return 'inch={}, outch={}, kernel_size={}, stride={}, padding={}, wsize={}, norm={}, ub={}, act={}'.format(
            self.inch, self.outch, self.kernel_size, self.stride, self.padding, self.wsize, self.norm, self.ub, self.act
        )

    def getweight(self, weight):
        if self.fused:
            return weight
        else:
            if self.norm is not None:
                if self.norm == "demod":
                    if weight.ndim == 5:
                        normdims = [2, 3, 4]
                    else:
                        normdims = [1, 2, 3]

                    if torch.jit.is_scripting():
                        # scripting doesn't support F.normalize(..., dim=list[int])
                        weight = weight / torch.linalg.vector_norm(weight, dim=normdims, keepdim=True)
                    else:
                        weight = F.normalize(weight, dim=normdims)

            weight = weight * self.weightgain

            return weight

    def fuse(self):
        if self.affine is None:
            with torch.no_grad():
                self.weight.data = self.getweight(self.weight)
            self.fused = True

    def forward(self, x, w : Optional[torch.Tensor]=None):
        b = x.size(0)

        if self.affine is not None and w is not None:
            # modulate
            affine = self.affine(w)[:, None, :, None, None] # [B, 1, inch, 1, 1]
            weight = self.weight * (affine * 0.1 + 1.)
        else:
            weight = self.weight

        weight = self.getweight(weight)

        if self.affine is not None and w is not None: 
            x = x.view(1, b * self.inch, x.size(2), x.size(3))
            weight = weight.view(b * self.outch, self.inch, self.kernel_size, self.kernel_size)
            groups = b
        else:
            groups = 1

        out = F.conv2d(x, weight, None,
                stride=self.stride, padding=self.padding, dilation=1, groups=groups)

        if self.affine is not None and w is not None:
            out = out.view(b, self.outch, out.size(2), out.size(3))

        if self.bias.ndim == 1:
            bias = self.bias[None, :, None, None]
        else:
            bias = self.bias[None, :, :, :]
        out = out + bias

        if self.act is not None:
            out = self.act(out)

        return out

class ConvTranspose2dWN(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size,
            stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ConvTranspose2dWN, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, True)
        self.g = nn.Parameter(torch.ones(out_channels))
        self.fused = False
        
    def fuse(self):
        wnorm = torch.sqrt(torch.sum(self.weight ** 2))
        self.weight.data = self.weight.data * self.g.data[None, :, None, None] / wnorm 
        self.fused = True
        
    def forward(self, x):
        bias = self.bias
        assert bias is not None
        if self.fused:
            return F.conv_transpose2d(x, self.weight,
                    bias=self.bias, stride=self.stride, padding=self.padding,
                    dilation=self.dilation, groups=self.groups)
        else:
            wnorm = torch.sqrt(torch.sum(self.weight ** 2))
            return F.conv_transpose2d(x, self.weight * self.g[None, :, None, None] / wnorm,
                    bias=self.bias, stride=self.stride, padding=self.padding,
                    dilation=self.dilation, groups=self.groups)

class ConvTranspose2dUB(nn.ConvTranspose2d):
    def __init__(self, width, height, in_channels, out_channels, kernel_size,
            stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ConvTranspose2dUB, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, False)
        self.bias_ = nn.Parameter(torch.zeros(out_channels, height, width))

    def forward(self, x):
        return F.conv_transpose2d(x, self.weight,
                bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups) + self.bias_[None, ...]

class ConvTranspose2dWNUB(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, height, width, kernel_size,
            stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ConvTranspose2dWNUB, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, False)
        self.g = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels, height, width))
        #self.biasf = nn.Parameter(torch.zeros(out_channels, height, width))
        self.fused = False

    def fuse(self):
        wnorm = torch.sqrt(torch.sum(self.weight ** 2))
        self.weight.data = self.weight.data * self.g.data[None, :, None, None] / wnorm 
        self.fused = True
        
    def forward(self, x):
        bias = self.bias
        assert bias is not None
        if self.fused:
            return F.conv_transpose2d(x, self.weight,
                    bias=None, stride=self.stride, padding=self.padding,
                    dilation=self.dilation, groups=self.groups) + bias[None, ...]
        else:
            wnorm = torch.sqrt(torch.sum(self.weight ** 2))
            return F.conv_transpose2d(x, self.weight * self.g[None, :, None, None] / wnorm,
                    bias=None, stride=self.stride, padding=self.padding,
                    dilation=self.dilation, groups=self.groups) + bias[None, ...]

class Conv3dUB(nn.Conv3d):
    def __init__(self, width, height, depth, in_channels, out_channels, kernel_size,
            stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv3dUB, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, False)
        self.bias = nn.Parameter(torch.zeros(out_channels, depth, height, width))
        
    def forward(self, x):
        return F.conv3d(x, self.weight,
                bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups) + self.bias[None, ...]

class ConvTranspose3dUB(nn.ConvTranspose3d):
    def __init__(self, width, height, depth, in_channels, out_channels, kernel_size,
            stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ConvTranspose3dUB, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, False)
        self.bias = nn.Parameter(torch.zeros(out_channels, depth, height, width))
        
    def forward(self, x):
        return F.conv_transpose3d(x, self.weight,
                bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups) + self.bias[None, ...]

class Rodrigues(nn.Module):
    def __init__(self):
        super(Rodrigues, self).__init__()

    def forward(self, rvec):
        theta = torch.sqrt(1e-5 + torch.sum(rvec ** 2, dim=1))
        rvec = rvec / theta[:, None]
        costh = torch.cos(theta)
        sinth = torch.sin(theta)
        return torch.stack((
            rvec[:, 0] ** 2 + (1. - rvec[:, 0] ** 2) * costh,
            rvec[:, 0] * rvec[:, 1] * (1. - costh) - rvec[:, 2] * sinth,
            rvec[:, 0] * rvec[:, 2] * (1. - costh) + rvec[:, 1] * sinth,

            rvec[:, 0] * rvec[:, 1] * (1. - costh) + rvec[:, 2] * sinth,
            rvec[:, 1] ** 2 + (1. - rvec[:, 1] ** 2) * costh,
            rvec[:, 1] * rvec[:, 2] * (1. - costh) - rvec[:, 0] * sinth,

            rvec[:, 0] * rvec[:, 2] * (1. - costh) - rvec[:, 1] * sinth,
            rvec[:, 1] * rvec[:, 2] * (1. - costh) + rvec[:, 0] * sinth,
            rvec[:, 2] ** 2 + (1. - rvec[:, 2] ** 2) * costh), dim=1).view(-1, 3, 3)

class Quaternion(nn.Module):
    def __init__(self):
        super(Quaternion, self).__init__()

    def forward(self, rvec):
        theta = torch.sqrt(1e-5 + torch.sum(rvec ** 2, dim=1))
        rvec = rvec / theta[:, None]
        return torch.stack((
            1. - 2. * rvec[:, 1] ** 2 - 2. * rvec[:, 2] ** 2,
            2. * (rvec[:, 0] * rvec[:, 1] - rvec[:, 2] * rvec[:, 3]),
            2. * (rvec[:, 0] * rvec[:, 2] + rvec[:, 1] * rvec[:, 3]),

            2. * (rvec[:, 0] * rvec[:, 1] + rvec[:, 2] * rvec[:, 3]),
            1. - 2. * rvec[:, 0] ** 2 - 2. * rvec[:, 2] ** 2,
            2. * (rvec[:, 1] * rvec[:, 2] - rvec[:, 0] * rvec[:, 3]),

            2. * (rvec[:, 0] * rvec[:, 2] - rvec[:, 1] * rvec[:, 3]),
            2. * (rvec[:, 0] * rvec[:, 3] + rvec[:, 1] * rvec[:, 2]),
            1. - 2. * rvec[:, 0] ** 2 - 2. * rvec[:, 1] ** 2
            ), dim=1).view(-1, 3, 3)

class BufferDict(nn.Module):
    def __init__(self, d, persistent=False):
        super(BufferDict, self).__init__()

        for k in d:
            self.register_buffer(k, d[k], persistent=False)

    def __getitem__(self, key):
        return self._buffers[key]

    def __setitem__(self, key, parameter):
        self.register_buffer(key, parameter, persistent=False)

def matrix_to_axisangle(r):
    th = torch.arccos(0.5 * (r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2] - 1.))[..., None]
    vec = 0.5 * torch.stack([
        r[..., 2, 1] - r[..., 1, 2],
        r[..., 0, 2] - r[..., 2, 0],
        r[..., 1, 0] - r[..., 0, 1]], dim=-1) / torch.sin(th)
    return th, vec

@torch.jit.script
def axisangle_to_matrix(rvec : torch.Tensor):
    theta = torch.sqrt(1e-5 + torch.sum(rvec ** 2, dim=-1))
    rvec = rvec / theta[..., None]
    costh = torch.cos(theta)
    sinth = torch.sin(theta)
    return torch.stack((
        torch.stack((rvec[..., 0] ** 2 + (1. - rvec[..., 0] ** 2) * costh,
        rvec[..., 0] * rvec[..., 1] * (1. - costh) - rvec[..., 2] * sinth,
        rvec[..., 0] * rvec[..., 2] * (1. - costh) + rvec[..., 1] * sinth), dim=-1),

        torch.stack((rvec[..., 0] * rvec[..., 1] * (1. - costh) + rvec[..., 2] * sinth,
        rvec[..., 1] ** 2 + (1. - rvec[..., 1] ** 2) * costh,
        rvec[..., 1] * rvec[..., 2] * (1. - costh) - rvec[..., 0] * sinth), dim=-1),

        torch.stack((rvec[..., 0] * rvec[..., 2] * (1. - costh) - rvec[..., 1] * sinth,
        rvec[..., 1] * rvec[..., 2] * (1. - costh) + rvec[..., 0] * sinth,
        rvec[..., 2] ** 2 + (1. - rvec[..., 2] ** 2) * costh), dim=-1)),
        dim=-2)

def rotation_interp(r0, r1, alpha):
    r0a = r0.view(-1, 3, 3)
    r1a = r1.view(-1, 3, 3)
    r = torch.bmm(r0a.permute(0, 2, 1), r1a).view_as(r0)

    th, rvec = matrix_to_axisangle(r)
    rvec = rvec * (alpha * th)

    r = axisangle_to_matrix(rvec)
    return torch.bmm(r0a, r.view(-1, 3, 3)).view_as(r0)

def fuse(trainiter=None, renderoptions={}):
    def _fuse(m):
        if hasattr(m, "fuse") and isinstance(m, torch.nn.Module):
            if m.fuse.__code__.co_argcount > 1:
                m.fuse(trainiter, renderoptions)
            else:
                m.fuse()
    return _fuse

def no_grad(m):
    for p in m.parameters():
        p.requires_grad = False
