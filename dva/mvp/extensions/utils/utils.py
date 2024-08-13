# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import time

import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F

try:
    from . import utilslib
except:
    import utilslib

class ComputeRaydirs(Function):
    @staticmethod
    def forward(self, viewpos, viewrot, focal, princpt, pixelcoords, volradius):
        for tensor in [viewpos, viewrot, focal, princpt, pixelcoords]:
            assert tensor.is_contiguous()

        N = viewpos.size(0)
        if isinstance(pixelcoords, tuple):
            W, H = pixelcoords
            pixelcoords = None
        else:
            H = pixelcoords.size(1)
            W = pixelcoords.size(2)

        raypos = torch.empty((N, H, W, 3), device=viewpos.device)
        raydirs = torch.empty((N, H, W, 3), device=viewpos.device)
        tminmax = torch.empty((N, H, W, 2), device=viewpos.device)
        utilslib.compute_raydirs_forward(viewpos, viewrot, focal, princpt,
                pixelcoords, W, H, volradius, raypos, raydirs, tminmax)

        return raypos, raydirs, tminmax

    @staticmethod
    def backward(self, grad_raydirs, grad_tminmax):
        return None, None, None, None, None, None

def compute_raydirs(viewpos, viewrot, focal, princpt, pixelcoords, volradius):
    raypos, raydirs, tminmax = ComputeRaydirs.apply(viewpos, viewrot, focal, princpt, pixelcoords, volradius)
    return raypos, raydirs, tminmax

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

def gradcheck():
    N = 2
    H = 64
    W = 64
    k3 = 4
    K = k3*k3*k3

    M = 32
    volradius = 1.

    # generate random inputs
    torch.manual_seed(1113)

    rodrigues = Rodrigues()

    _viewpos = torch.tensor([[-0.0, 0.0, -4.] for n in range(N)], device="cuda") + torch.randn(N, 3, device="cuda") * 0.1
    viewrvec = torch.randn(N, 3, device="cuda") * 0.01
    _viewrot = rodrigues(viewrvec)

    _focal = torch.tensor([[W*4.0, W*4.0] for n in range(N)], device="cuda")
    _princpt = torch.tensor([[W*0.5, H*0.5] for n in range(N)], device="cuda")
    pixely, pixelx = torch.meshgrid(torch.arange(H, device="cuda").float(), torch.arange(W, device="cuda").float())
    _pixelcoords = torch.stack([pixelx, pixely], dim=-1)[None, :, :, :].repeat(N, 1, 1, 1)

    _viewpos = _viewpos.contiguous().detach().clone()
    _viewpos.requires_grad = True
    _viewrot = _viewrot.contiguous().detach().clone()
    _viewrot.requires_grad = True
    _focal = _focal.contiguous().detach().clone()
    _focal.requires_grad = True
    _princpt = _princpt.contiguous().detach().clone()
    _princpt.requires_grad = True
    _pixelcoords = _pixelcoords.contiguous().detach().clone()
    _pixelcoords.requires_grad = True

    max_len = 6.0
    _stepsize = max_len / 15.5

    params = [_viewpos, _viewrot, _focal, _princpt]
    paramnames = ["viewpos", "viewrot", "focal", "princpt"]

    ########################### run pytorch version ###########################

    viewpos = _viewpos
    viewrot = _viewrot
    focal = _focal
    princpt = _princpt
    pixelcoords = _pixelcoords

    raypos = viewpos[:, None, None, :].repeat(1, H, W, 1)

    raydir = (pixelcoords - princpt[:, None, None, :]) / focal[:, None, None, :]
    raydir = torch.cat([raydir, torch.ones_like(raydir[:, :, :, 0:1])], dim=-1)
    raydir = torch.sum(viewrot[:, None, None, :, :] * raydir[:, :, :, :, None], dim=-2)
    raydir = raydir / torch.sqrt(torch.sum(raydir ** 2, dim=-1, keepdim=True))

    t1 = (-1. - viewpos[:, None, None, :]) / raydir
    t2 = ( 1. - viewpos[:, None, None, :]) / raydir
    tmin = torch.max(torch.min(t1[..., 0], t2[..., 0]),
           torch.max(torch.min(t1[..., 1], t2[..., 1]),
                     torch.min(t1[..., 2], t2[..., 2]))).clamp(min=0.)
    tmax = torch.min(torch.max(t1[..., 0], t2[..., 0]),
           torch.min(torch.max(t1[..., 1], t2[..., 1]),
                     torch.max(t1[..., 2], t2[..., 2])))

    tminmax = torch.stack([tmin, tmax], dim=-1)

    sample0 = raydir

    torch.cuda.synchronize()
    time1 = time.time()

    sample0.backward(torch.ones_like(sample0))

    torch.cuda.synchronize()
    time2 = time.time()

    grads0 = [p.grad.detach().clone() if p.grad is not None else None for p in params]

    for p in params:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

    ############################## run cuda version ###########################

    viewpos = _viewpos
    viewrot = _viewrot
    focal = _focal
    princpt = _princpt
    pixelcoords = _pixelcoords

    niter = 1

    for p in params:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()
    t0 = time.time()
    torch.cuda.synchronize()

    sample1 = compute_raydirs(viewpos, viewrot, focal, princpt, pixelcoords, volradius)[1]

    t1 = time.time()
    torch.cuda.synchronize()

    print("-----------------------------------------------------------------")
    print("{:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format("", "maxabsdiff", "dp", "index", "py", "cuda"))
    ind = torch.argmax(torch.abs(sample0 - sample1))
    print("{:<10} {:>10.5} {:>10.5} {:>10} {:>10.5} {:>10.5}".format(
        "fwd",
        torch.max(torch.abs(sample0 - sample1)).item(),
        (torch.sum(sample0 * sample1) / torch.sqrt(torch.sum(sample0 * sample0) * torch.sum(sample1 * sample1))).item(),
        ind.item(),
        sample0.view(-1)[ind].item(),
        sample1.view(-1)[ind].item()))

    sample1.backward(torch.ones_like(sample1), retain_graph=True)

    torch.cuda.synchronize()
    t2 = time.time()


    print("{:<10} {:10.5} {:10.5} {:10.5}".format("time", tf / niter, tb / niter, (tf + tb) / niter))
    grads1 = [p.grad.detach().clone() if p.grad is not None else None for p in params]

    ############# compare results #############

    for p, g0, g1 in zip(paramnames, grads0, grads1):
        ind = torch.argmax(torch.abs(g0 - g1))
        print("{:<10} {:>10.5} {:>10.5} {:>10} {:>10.5} {:>10.5}".format(
                p,
                torch.max(torch.abs(g0 - g1)).item(),
                (torch.sum(g0 * g1) / torch.sqrt(torch.sum(g0 * g0) * torch.sum(g1 * g1))).item(),
                ind.item(),
                g0.view(-1)[ind].item(),
                g1.view(-1)[ind].item()))

if __name__ == "__main__":
    gradcheck()
