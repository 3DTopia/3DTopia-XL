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
    from . import mvpraymarchlib
except:
    import mvpraymarchlib

def build_accel(primtransfin, algo, fixedorder=False):
    """build bvh structure given primitive centers and sizes
    
    Parameters:
    ----------
    primtransfin : tuple[tensor, tensor, tensor]
        primitive transform tensors
    algo : int
        raymarching algorithm
    fixedorder : optional[str]
        True means the bvh builder will not reorder primitives and will
        use a trivial tree structure. Likely to be slow for arbitrary
        configurations of primitives.
    
    """
    primpos, primrot, primscale = primtransfin

    N = primpos.size(0)
    K = primpos.size(1)

    dev = primpos.device

    # compute and sort morton codes
    if fixedorder:
        sortedobjid = (torch.arange(N*K, dtype=torch.int32, device=dev) % K).view(N, K)
    else:
        cmax = primpos.max(dim=1, keepdim=True)[0]
        cmin = primpos.min(dim=1, keepdim=True)[0]

        centers_norm = (primpos - cmin) / (cmax - cmin).clamp(min=1e-8)

        mortoncode = torch.empty((N, K), dtype=torch.int32, device=dev)
        mvpraymarchlib.compute_morton(centers_norm, mortoncode, algo)
        sortedcode, sortedobjid_long = torch.sort(mortoncode, dim=-1)
        sortedobjid = sortedobjid_long.int()

    if fixedorder:
        nodechildren = torch.cat([
            torch.arange(1, (K - 1) * 2 + 1, dtype=torch.int32, device=dev),
            torch.div(torch.arange(-2, -(K * 2 + 1) - 1, -1, dtype=torch.int32, device=dev), 2, rounding_mode="floor")],
        dim=0).view(1, K + K - 1, 2).repeat(N, 1, 1)
        nodeparent = (
            torch.div(torch.arange(-1, K * 2 - 2, dtype=torch.int32, device=dev), 2, rounding_mode="floor")
               .view(1, -1).repeat(N, 1))
    else:
        nodechildren = torch.empty((N, K + K - 1, 2), dtype=torch.int32, device=dev)
        nodeparent = torch.full((N, K + K - 1), -1, dtype=torch.int32, device=dev)
        mvpraymarchlib.build_tree(sortedcode, nodechildren, nodeparent)

    nodeaabb = torch.empty((N, K + K - 1, 2, 3), dtype=torch.float32, device=dev)
    mvpraymarchlib.compute_aabb(*primtransfin, sortedobjid, nodechildren, nodeparent, nodeaabb, algo)

    return sortedobjid, nodechildren, nodeaabb

class MVPRaymarch(Function):
    """Custom Function for raymarching Mixture of Volumetric Primitives."""
    @staticmethod
    def forward(self, raypos, raydir, stepsize, tminmax,
            primpos, primrot, primscale,
            template, warp,
            rayterm, gradmode, options):
        algo = options["algo"]
        usebvh = options["usebvh"]
        sortprims = options["sortprims"]
        randomorder = options["randomorder"]
        maxhitboxes = options["maxhitboxes"]
        synchitboxes = options["synchitboxes"]
        chlast = options["chlast"]
        fadescale = options["fadescale"]
        fadeexp = options["fadeexp"]
        accum = options["accum"]
        termthresh = options["termthresh"]
        griddim = options["griddim"]
        if isinstance(options["blocksize"], tuple):
            blocksizex, blocksizey = options["blocksize"]
        else:
            blocksizex = options["blocksize"]
            blocksizey = 1

        assert raypos.is_contiguous() and raypos.size(3) == 3
        assert raydir.is_contiguous() and raydir.size(3) == 3
        assert tminmax.is_contiguous() and tminmax.size(3) == 2

        assert primpos is None or primpos.is_contiguous() and primpos.size(2) == 3
        assert primrot is None or primrot.is_contiguous() and primrot.size(2) == 3
        assert primscale is None or primscale.is_contiguous() and primscale.size(2) == 3

        if chlast:
            assert template.is_contiguous() and len(template.size()) == 6 and template.size(-1) == 4
            assert warp is None or (warp.is_contiguous() and warp.size(-1) == 3)
        else:
            assert template.is_contiguous() and len(template.size()) == 6 and template.size(2) == 4
            assert warp is None or (warp.is_contiguous() and warp.size(2) == 3)

        primtransfin = (primpos, primrot, primscale)

        # Build bvh
        if usebvh is not False:
            # compute radius of primitives
            sortedobjid, nodechildren, nodeaabb = build_accel(primtransfin,
                    algo, fixedorder=usebvh=="fixedorder")
            assert sortedobjid.is_contiguous()
            assert nodechildren.is_contiguous()
            assert nodeaabb.is_contiguous()

            if randomorder:
                sortedobjid = sortedobjid[torch.randperm(len(sortedobjid))]
        else:
            _, sortedobjid, nodechildren, nodeaabb = None, None, None, None

        # march through boxes
        N, H, W = raypos.size(0), raypos.size(1), raypos.size(2)
        rayrgba = torch.empty((N, H, W, 4), device=raypos.device)
        if gradmode:
            raysat = torch.full((N, H, W, 3), -1, dtype=torch.float32, device=raypos.device)
            rayterm = None
        else:
            raysat = None
            rayterm = None

        mvpraymarchlib.raymarch_forward(
                raypos, raydir, stepsize, tminmax,
                sortedobjid, nodechildren, nodeaabb,
                *primtransfin,
                template, warp,
                rayrgba, raysat, rayterm,
                algo, sortprims, maxhitboxes, synchitboxes, chlast,
                fadescale, fadeexp,
                accum, termthresh,
                griddim, blocksizex, blocksizey)

        self.save_for_backward(
                raypos, raydir, tminmax,
                sortedobjid, nodechildren, nodeaabb,
                primpos, primrot, primscale,
                template, warp,
                rayrgba, raysat, rayterm)
        self.options = options
        self.stepsize = stepsize

        return rayrgba

    @staticmethod
    def backward(self, grad_rayrgba):
        (raypos, raydir, tminmax,
            sortedobjid, nodechildren, nodeaabb,
            primpos, primrot, primscale,
            template, warp,
            rayrgba, raysat, rayterm) = self.saved_tensors
        algo = self.options["algo"]
        usebvh = self.options["usebvh"]
        sortprims = self.options["sortprims"]
        maxhitboxes = self.options["maxhitboxes"]
        synchitboxes = self.options["synchitboxes"]
        chlast = self.options["chlast"]
        fadescale = self.options["fadescale"]
        fadeexp = self.options["fadeexp"]
        accum = self.options["accum"]
        termthresh = self.options["termthresh"]
        griddim = self.options["griddim"]
        if isinstance(self.options["bwdblocksize"], tuple):
            blocksizex, blocksizey = self.options["bwdblocksize"]
        else:
            blocksizex = self.options["bwdblocksize"]
            blocksizey = 1

        stepsize = self.stepsize

        grad_primpos = torch.zeros_like(primpos)
        grad_primrot = torch.zeros_like(primrot)
        grad_primscale = torch.zeros_like(primscale)
        primtransfin = (primpos, grad_primpos, primrot, grad_primrot, primscale, grad_primscale)

        grad_template = torch.zeros_like(template)
        grad_warp = torch.zeros_like(warp) if warp is not None else None

        mvpraymarchlib.raymarch_backward(raypos, raydir, stepsize, tminmax,
                sortedobjid, nodechildren, nodeaabb,

                *primtransfin,

                template, grad_template, warp, grad_warp,

                rayrgba, grad_rayrgba.contiguous(), raysat, rayterm,

                algo, sortprims, maxhitboxes, synchitboxes, chlast,
                fadescale, fadeexp,
                accum, termthresh,
                griddim, blocksizex, blocksizey)

        return (None, None, None, None,
                grad_primpos, grad_primrot, grad_primscale,
                grad_template, grad_warp,
                None, None, None)

def mvpraymarch(raypos, raydir, stepsize, tminmax,
            primtransf,
            template, warp,
            rayterm=None,
            algo=0, usebvh="fixedorder",
            sortprims=False, randomorder=False,
            maxhitboxes=512, synchitboxes=True,
            chlast=True, fadescale=8., fadeexp=8.,
            accum=0, termthresh=0.,
            griddim=3, blocksize=(8, 16), bwdblocksize=(8, 16)):
    """Main entry point for raymarching MVP.

    Parameters:
    ----------
    raypos: N x H x W x 3 tensor of ray origins
    raydir: N x H x W x 3 tensor of ray directions
    stepsize: raymarching step size
    tminmax: N x H x W x 2 tensor of raymarching min/max bounds
    template: N x K x 4 x TD x TH x TW tensor of K RGBA primitives
    warp: N x K x 3 x TD x TH x TW tensor of K warp fields (optional)
    primpos: N x K x 3 tensor of primitive centers
    primrot: N x K x 3 x 3 tensor of primitive orientations
    primscale: N x K x 3 tensor of primitive inverse dimension lengths
    algo: algorithm for raymarching (valid values: 0, 1). algo=0 is the fastest.
        Currently algo=0 has a limit of 512 primitives per ray, so problems can
        occur if there are many more boxes. all sortprims=True options have
        this limitation, but you can use (algo=1, sortprims=False,
        usebvh="fixedorder") which works correctly and has no primitive number
        limitation (but is slightly slower).
    usebvh: True to use bvh, "fixedorder" for a simple BVH, False for no bvh
    sortprims: True to sort overlapping primitives at a sample point. Must
        be True for gradients to match the PyTorch gradients. Seems unstable
        if False but also not a big performance bottleneck.
    chlast: whether template is provided as channels last or not. True tends
        to be faster.
    fadescale: Opacity is faded at the borders of the primitives by the equation
        exp(-fadescale * x ** fadeexp) where x is the normalized coordinates of
        the primitive.
    fadeexp: Opacity is faded at the borders of the primitives by the equation
        exp(-fadescale * x ** fadeexp) where x is the normalized coordinates of
        the primitive.
    griddim: CUDA grid dimensionality.
    blocksize: blocksize of CUDA kernels. Should be 2-element tuple if
        griddim>1, or integer if griddim==1."""
    if isinstance(primtransf, tuple):
        primpos, primrot, primscale = primtransf
    else:
        primpos, primrot, primscale = (
                primtransf[:, :, 0, :].contiguous(),
                primtransf[:, :, 1:4, :].contiguous(),
                primtransf[:, :, 4, :].contiguous())
    primtransfin = (primpos, primrot, primscale)

    out = MVPRaymarch.apply(raypos, raydir, stepsize, tminmax,
            *primtransfin,
            template, warp,
            rayterm, torch.is_grad_enabled(),
            {"algo": algo, "usebvh": usebvh, "sortprims": sortprims, "randomorder": randomorder,
                "maxhitboxes": maxhitboxes, "synchitboxes": synchitboxes,
                "chlast": chlast, "fadescale": fadescale, "fadeexp": fadeexp,
                "accum": accum, "termthresh": termthresh,
                "griddim": griddim, "blocksize": blocksize, "bwdblocksize": bwdblocksize})
    return out

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

def gradcheck(usebvh=True, sortprims=True, maxhitboxes=512, synchitboxes=False,
        dowarp=False, chlast=False, fadescale=8., fadeexp=8.,
        accum=0, termthresh=0., algo=0, griddim=2, blocksize=(8, 16), bwdblocksize=(8, 16)):
    N = 2
    H = 65
    W = 65
    k3 = 4
    K = k3*k3*k3

    M = 32

    print("=================================================================")
    print("usebvh={}, sortprims={}, maxhb={}, synchb={}, dowarp={}, chlast={}, "
        "fadescale={}, fadeexp={}, accum={}, termthresh={}, algo={}, griddim={}, "
        "blocksize={}, bwdblocksize={}".format(
        usebvh, sortprims, maxhitboxes, synchitboxes, dowarp, chlast,
        fadescale, fadeexp, accum, termthresh, algo, griddim, blocksize,
        bwdblocksize))

    # generate random inputs
    torch.manual_seed(1112)

    coherent_rays = True
    if not coherent_rays:
        _raypos = torch.randn(N, H, W, 3).to("cuda")
        _raydir = torch.randn(N, H, W, 3).to("cuda")
        _raydir /= torch.sqrt(torch.sum(_raydir ** 2, dim=-1, keepdim=True))
    else:
        focal = torch.tensor([[W*4.0, W*4.0] for n in range(N)])
        princpt = torch.tensor([[W*0.5, H*0.5] for n in range(N)])
        pixely, pixelx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float())
        pixelcoords = torch.stack([pixelx, pixely], dim=-1)[None, :, :, :].repeat(N, 1, 1, 1)

        raydir = (pixelcoords - princpt[:, None, None, :]) / focal[:, None, None, :]
        raydir = torch.cat([raydir, torch.ones_like(raydir[:, :, :, 0:1])], dim=-1)
        raydir = raydir / torch.sqrt(torch.sum(raydir ** 2, dim=-1, keepdim=True))

        _raypos = torch.tensor([-0.0, 0.0, -4.])[None, None, None, :].repeat(N, H, W, 1).to("cuda")
        _raydir = raydir.to("cuda")
        _raydir /= torch.sqrt(torch.sum(_raydir ** 2, dim=-1, keepdim=True))

    max_len = 6.0
    _stepsize = max_len / 15.386928
    _tminmax = max_len*torch.arange(2, dtype=torch.float32)[None, None, None, :].repeat(N, H, W, 1).to("cuda") + \
            torch.rand(N, H, W, 2, device="cuda") * 1.

    _template = torch.randn(N, K, 4, M, M, M, requires_grad=True)
    _template.data[:, :, -1, :, :, :] -= 3.5
    _template = _template.contiguous().detach().clone()
    _template.requires_grad = True
    gridxyz = torch.stack(torch.meshgrid(
        torch.linspace(-1., 1., M//2),
        torch.linspace(-1., 1., M//2),
        torch.linspace(-1., 1., M//2))[::-1], dim=0).contiguous()
    _warp = (torch.randn(N, K, 3, M//2, M//2, M//2) * 0.01 + gridxyz[None, None, :, :, :, :]).contiguous().detach().clone()
    _warp.requires_grad = True
    _primpos = torch.randn(N, K, 3, requires_grad=True)
    _primpos = torch.randn(N, K, 3, requires_grad=True)

    coherent_centers = True
    if coherent_centers:
        ns = k3
        #assert ns*ns*ns==K
        grid3d = torch.stack(torch.meshgrid(
            torch.linspace(-1., 1., ns),
            torch.linspace(-1., 1., ns),
            torch.linspace(-1., 1., K//(ns*ns)))[::-1], dim=0)[None]
        _primpos = ((
            grid3d.permute((0, 2, 3, 4, 1)).reshape(1, K, 3).expand(N, -1, -1) +
            0.1 * torch.randn(N, K, 3, requires_grad=True)
            )).contiguous().detach().clone()
        _primpos.requires_grad = True
    scale_ws = 1.
    _primrot = torch.randn(N, K, 3)
    rodrigues = Rodrigues()
    _primrot = rodrigues(_primrot.view(-1, 3)).view(N, K, 3, 3).contiguous().detach().clone()
    _primrot.requires_grad = True

    _primscale = torch.randn(N, K, 3, requires_grad=True)
    _primscale.data *= 0.0

    if dowarp:
        params = [_template, _warp, _primscale, _primrot, _primpos]
        paramnames = ["template", "warp", "primscale", "primrot", "primpos"]
    else:
        params = [_template, _primscale, _primrot, _primpos]
        paramnames = ["template", "primscale", "primrot", "primpos"]

    termthreshorig = termthresh

    ########################### run pytorch version ###########################

    raypos = _raypos
    raydir = _raydir
    stepsize = _stepsize
    tminmax = _tminmax

    #template = F.softplus(_template.to("cuda") * 1.5)
    template = F.softplus(_template.to("cuda") * 1.5) if algo != 2 else _template.to("cuda") * 1.5
    warp = _warp.to("cuda")
    primpos = _primpos.to("cuda") * 0.3
    primrot = _primrot.to("cuda")
    primscale = scale_ws * torch.exp(0.1 * _primscale.to("cuda"))

    # python raymarching implementation
    rayrgba = torch.zeros((N, H, W, 4)).to("cuda")
    raypos = raypos + raydir * tminmax[:, :, :, 0, None]
    t = tminmax[:, :, :, 0]

    step = 0
    t0 = t.detach().clone()
    raypos0 = raypos.detach().clone()

    torch.cuda.synchronize()
    time0 = time.time()

    while (t < tminmax[:, :, :, 1]).any():
        valid2 = torch.ones_like(rayrgba[:, :, :, 3:4])

        for k in range(K):
            y0 = torch.bmm(
                    (raypos - primpos[:, k, None, None, :]).view(raypos.size(0), -1, raypos.size(3)),
                    primrot[:, k, :, :]).view_as(raypos) * primscale[:, k, None, None, :]

            fade = torch.exp(-fadescale * torch.sum(torch.abs(y0) ** fadeexp, dim=-1, keepdim=True))

            if dowarp:
                y1 = F.grid_sample(
                        warp[:, k, :, :, :, :],
                        y0[:, None, :, :, :], align_corners=True)[:, :, 0, :, :].permute(0, 2, 3, 1)
            else:
                y1 = y0

            sample = F.grid_sample(
                    template[:, k, :, :, :, :],
                    y1[:, None, :, :, :], align_corners=True)[:, :, 0, :, :].permute(0, 2, 3, 1)

            valid1 = (
                torch.prod(y0[:, :, :, :] >= -1., dim=-1, keepdim=True) *
                torch.prod(y0[:, :, :, :] <= 1., dim=-1, keepdim=True))

            valid = ((t >= tminmax[:, :, :, 0]) & (t < tminmax[:, :, :, 1])).float()[:, :, :, None]

            alpha0 = sample[:, :, :, 3:4]

            rgb = sample[:, :, :, 0:3] * valid * valid1
            alpha = alpha0 * fade * stepsize * valid * valid1

            if accum == 0:
                newalpha = rayrgba[:, :, :, 3:4] + alpha
                contrib = (newalpha.clamp(max=1.0) - rayrgba[:, :, :, 3:4]) * valid * valid1
                rayrgba = rayrgba + contrib * torch.cat([rgb, torch.ones_like(alpha)], dim=-1)
            else:
                raise

        step += 1
        t = t0 + stepsize * step
        raypos = raypos0 + raydir * stepsize * step

    print(rayrgba[..., -1].min().item(), rayrgba[..., -1].max().item())

    sample0 = rayrgba

    torch.cuda.synchronize()
    time1 = time.time()

    sample0.backward(torch.ones_like(sample0))

    torch.cuda.synchronize()
    time2 = time.time()

    print("{:<10} {:>10} {:>10} {:>10}".format("", "fwd", "bwd", "total"))
    print("{:<10} {:10.5} {:10.5} {:10.5}".format("pytime", time1 - time0, time2 - time1, time2 - time0))

    grads0 = [p.grad.detach().clone() for p in params]

    for p in params:
        p.grad.detach_()
        p.grad.zero_()

    ############################## run cuda version ###########################

    raypos = _raypos
    raydir = _raydir
    stepsize = _stepsize
    tminmax = _tminmax

    template = F.softplus(_template.to("cuda") * 1.5) if algo != 2 else _template.to("cuda") * 1.5
    warp = _warp.to("cuda")
    if chlast:
        template = template.permute(0, 1, 3, 4, 5, 2).contiguous()
        warp = warp.permute(0, 1, 3, 4, 5, 2).contiguous()
    primpos = _primpos.to("cuda") * 0.3
    primrot = _primrot.to("cuda")
    primscale = scale_ws * torch.exp(0.1 * _primscale.to("cuda"))

    niter = 1

    tf, tb = 0., 0.
    for i in range(niter):
        for p in params:
            try:
                p.grad.detach_()
                p.grad.zero_()
            except:
                pass
        t0 = time.time()
        torch.cuda.synchronize()
        sample1 = mvpraymarch(raypos, raydir, stepsize, tminmax,
                (primpos, primrot, primscale),
                template, warp if dowarp else None,
                algo=algo, usebvh=usebvh, sortprims=sortprims, 
                maxhitboxes=maxhitboxes, synchitboxes=synchitboxes,
                chlast=chlast, fadescale=fadescale, fadeexp=fadeexp,
                accum=accum, termthresh=termthreshorig,
                griddim=griddim, blocksize=blocksize, bwdblocksize=bwdblocksize)
        t1 = time.time()
        torch.cuda.synchronize()
        sample1.backward(torch.ones_like(sample1), retain_graph=True)
        torch.cuda.synchronize()
        t2 = time.time()
        tf += t1 - t0
        tb += t2 - t1

    print("{:<10} {:10.5} {:10.5} {:10.5}".format("time", tf / niter, tb / niter, (tf + tb) / niter))
    grads1 = [p.grad.detach().clone() for p in params]

    ############# compare results #############

    print("-----------------------------------------------------------------")
    print("{:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format("", "maxabsdiff", "dp", "||py||", "||cuda||", "index", "py", "cuda"))
    ind = torch.argmax(torch.abs(sample0 - sample1))
    print("{:<10} {:>10.5} {:>10.5} {:>10.5} {:>10.5} {:>10} {:>10.5} {:>10.5}".format(
        "fwd",
        torch.max(torch.abs(sample0 - sample1)).item(),
        (torch.sum(sample0 * sample1) / torch.sqrt(torch.sum(sample0 * sample0) * torch.sum(sample1 * sample1))).item(),
        torch.sqrt(torch.sum(sample0 * sample0)).item(),
        torch.sqrt(torch.sum(sample1 * sample1)).item(),
        ind.item(),
        sample0.view(-1)[ind].item(),
        sample1.view(-1)[ind].item()))

    for p, g0, g1 in zip(paramnames, grads0, grads1):
        ind = torch.argmax(torch.abs(g0 - g1))
        print("{:<10} {:>10.5} {:>10.5} {:>10.5} {:>10.5} {:>10} {:>10.5} {:>10.5}".format(
                p,
                torch.max(torch.abs(g0 - g1)).item(),
                (torch.sum(g0 * g1) / torch.sqrt(torch.sum(g0 * g0) * torch.sum(g1 * g1))).item(),
                torch.sqrt(torch.sum(g0 * g0)).item(),
                torch.sqrt(torch.sum(g1 * g1)).item(),
                ind.item(),
                g0.view(-1)[ind].item(),
                g1.view(-1)[ind].item()))

if __name__ == "__main__":
    gradcheck(usebvh="fixedorder", sortprims=False, maxhitboxes=512, synchitboxes=True,
            dowarp=False, chlast=True, fadescale=6.5, fadeexp=7.5, accum=0, algo=0, griddim=3)
    gradcheck(usebvh="fixedorder", sortprims=False, maxhitboxes=512, synchitboxes=True,
            dowarp=True, chlast=True, fadescale=6.5, fadeexp=7.5, accum=0, algo=1, griddim=3)
