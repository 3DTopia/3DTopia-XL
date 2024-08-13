# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
""" Volumetric autoencoder (image -> encoding -> volume -> image) """
import inspect
import time
from typing import Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import models.utils

from extensions.utils.utils import compute_raydirs

@torch.jit.script
def compute_raydirs_ref(pixelcoords : torch.Tensor, viewrot : torch.Tensor, focal : torch.Tensor, princpt : torch.Tensor):
    raydir = (pixelcoords - princpt[:, None, None, :]) / focal[:, None, None, :]
    raydir = torch.cat([raydir, torch.ones_like(raydir[:, :, :, 0:1])], dim=-1)
    raydir = torch.sum(viewrot[:, None, None, :, :] * raydir[:, :, :, :, None], dim=-2)
    raydir = F.normalize(raydir, dim=-1)

    return raydir

@torch.jit.script
def compute_rmbounds(viewpos : torch.Tensor, raydir : torch.Tensor, volradius : float):
    viewpos = viewpos / volradius

    # compute raymarching starting points
    with torch.no_grad():
        t1 = (-1. - viewpos[:, None, None, :]) / raydir
        t2 = ( 1. - viewpos[:, None, None, :]) / raydir
        tmin = torch.max(torch.min(t1[..., 0], t2[..., 0]),
               torch.max(torch.min(t1[..., 1], t2[..., 1]),
                         torch.min(t1[..., 2], t2[..., 2])))
        tmax = torch.min(torch.max(t1[..., 0], t2[..., 0]),
               torch.min(torch.max(t1[..., 1], t2[..., 1]),
                         torch.max(t1[..., 2], t2[..., 2])))

        intersections = tmin < tmax
        t = torch.where(intersections, tmin, torch.zeros_like(tmin)).clamp(min=0.)
        tmin = torch.where(intersections, tmin, torch.zeros_like(tmin)).clamp(min=0.)
        tmax = torch.where(intersections, tmax, torch.zeros_like(tmin))

    raypos = viewpos[:, None, None, :] + raydir * 0.
    tminmax = torch.stack([tmin, tmax], dim=-1)

    return raypos, tminmax

class Autoencoder(nn.Module):
    def __init__(self, dataset, encoder, decoder, raymarcher, colorcal,
            volradius, bgmodel=None, encoderinputs=[], topology=None,
            imagemean=0., imagestd=1., vertmask=None, cudaraydirs=True):
        super(Autoencoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.raymarcher = raymarcher
        self.colorcal = colorcal
        self.volradius = volradius
        self.bgmodel = bgmodel
        self.encoderinputs = encoderinputs

        if hasattr(dataset, 'vertmean'):
            self.register_buffer("vertmean", torch.from_numpy(dataset.vertmean), persistent=False)
            self.vertstd = dataset.vertstd
        if hasattr(dataset, 'texmean'):
            self.register_buffer("texmean", torch.from_numpy(dataset.texmean), persistent=False)
            self.texstd = dataset.texstd
        self.imagemean = imagemean
        self.imagestd = imagestd

        self.cudaraydirs = cudaraydirs

        if vertmask is not None:
            self.register_buffer("vertmask", torch.from_numpy(vertmask), persistent=False)

        self.irgbmsestart = -1

    def forward(self,
            camrot : torch.Tensor,
            campos : torch.Tensor,
            focal : torch.Tensor,
            princpt : torch.Tensor,
            camindex : Optional[torch.Tensor] = None,
            pixelcoords : Optional[torch.Tensor]=None,
            modelmatrix : Optional[torch.Tensor]=None,
            modelmatrixinv : Optional[torch.Tensor]=None,
            modelmatrix_next : Optional[torch.Tensor]=None,
            modelmatrixinv_next : Optional[torch.Tensor]=None,
            validinput : Optional[torch.Tensor]=None,
            avgtex : Optional[torch.Tensor]=None,
            avgtex_next : Optional[torch.Tensor]=None,
            verts : Optional[torch.Tensor]=None,
            verts_next : Optional[torch.Tensor]=None,
            fixedcamimage : Optional[torch.Tensor]=None,
            encoding : Optional[torch.Tensor]=None,
            image : Optional[torch.Tensor]=None,
            imagemask : Optional[torch.Tensor]=None,
            imagevalid : Optional[torch.Tensor]=None,
            bg : Optional[torch.Tensor]=None,
            renderoptions : dict ={},
            trainiter : int=-1,
            evaliter : Optional[torch.Tensor]=None,
            outputlist : list=[],
            losslist : list=[],
            **kwargs):
        """
        Parameters
        ----------
        camrot : torch.Tensor [B, 3, 3]
            Rotation matrix of target view camera
        campos : torch.Tensor [B, 3]
            Position of target view camera
        focal : torch.Tensor [B, 2]
            Focal length of target view camera
        princpt : torch.Tensor [B, 2]
            Princple point of target view camera
        camindex : torch.Tensor[int32], optional [B]
            Camera index within the list of all cameras
        pixelcoords : torch.Tensor, optional [B, H', W', 2]
            Pixel coordinates to render of the target view camera
        modelmatrix : torch.Tensor, optional [B, 3, 3]
            Relative transform from the 'neutral' pose of object
        validinput : torch.Tensor, optional [B]
            Whether the current batch element is valid (used for missing images)
        avgtex : torch.Tensor, optional [B, 3, 1024, 1024]
            Texture map averaged from all viewpoints
        verts : torch.Tensor, optional [B, 7306, 3]
            Mesh vertex positions
        fixedcamimage : torch.Tensor, optional [B, 3, 512, 334]
            Camera images from a one or more cameras that are always the same
            (i.e., unrelated to target)
        encoding : torch.Tensor, optional [B, 256]
            Direct encodings (overrides encoder)
        image : torch.Tensor, optional [B, 3, H, W]
            Target image
        imagemask : torch.Tensor, optional [B, 1, H, W]
            Target image mask for computing reconstruction loss
        imagevalid : torch.Tensor, optional [B]
        bg : torch.Tensor, optional [B, 3, H, W]
        renderoptions : dict
            Rendering/raymarching options (e.g., stepsize, whether to output debug images, etc.)
        trainiter : int
            Training iteration number
        outputlist : list
            Values to return (e.g., image reconstruction, debug output)
        losslist : list
            Losses to output (e.g., image reconstruction loss, priors)

        Returns
        -------
        result : dict
            Contains outputs specified in outputlist (e.g., image rgb
            reconstruction "irgbrec")
        losses : dict
            Losses to optimize
        """
        resultout = {}
        resultlosses = {}

        aestart = time.time()

        # encode/get encoding
        # verts [6, 7306, 3]
        # avgtex [6, 3, 256, 256]
        if encoding is None:
            if "enctime" in outputlist:
                torch.cuda.synchronize()
                encstart = time.time()
            encout, enclosses = self.encoder(
                    *[dict(verts=verts, avgtex=avgtex, fixedcamimage=fixedcamimage)[k] for k in self.encoderinputs],
                    losslist=losslist)
            if "enctime" in outputlist:
                torch.cuda.synchronize()
                encend = time.time()
                resultout["enctime"] = encend - encstart

            # encoding [6, 256]
            encoding = encout["encoding"]
            resultlosses.update(enclosses)

        # compute relative viewing position
        if modelmatrixinv is not None:
            viewrot = torch.bmm(camrot, modelmatrixinv[:, :3, :3])
            viewpos = torch.bmm((campos[:, :] - modelmatrixinv[:, :3, 3])[:, None, :], modelmatrixinv[:, :3, :3])[:, 0, :]
        else:
            viewrot = camrot
            viewpos = campos

        # decode volumetric representation
        if "dectime" in outputlist:
            torch.cuda.synchronize()
            decstart = time.time()
        if isinstance(self.decoder, torch.jit.ScriptModule):
            # torchscript requires statically typed dict
            renderoptionstyped : Dict[str, str] = {k: str(v) for k, v in renderoptions.items()}
        else:
            renderoptionstyped = renderoptions
        decout, declosses = self.decoder(
                encoding,
                viewpos,
                renderoptions=renderoptionstyped,
                trainiter=trainiter,
                evaliter=evaliter,
                losslist=losslist)
        if "dectime" in outputlist:
            torch.cuda.synchronize()
            decend = time.time()
            resultout["dectime"] = decend - decstart
        resultlosses.update(declosses)

        # compute vertex loss
        if "vertmse" in losslist:
            weight = validinput[:, None, None].expand_as(verts)

            if hasattr(self, "vertmask"):
                weight = weight * self.vertmask[None, :, None]

            vertsrecstd = (decout["verts"] - self.vertmean) / self.vertstd

            vertsqerr = weight * (verts - vertsrecstd) ** 2

            vertmse = torch.sum(vertsqerr.view(vertsqerr.size(0), -1), dim=-1)
            vertmse_weight = torch.sum(weight.view(weight.size(0), -1), dim=-1)

            resultlosses["vertmse"] = (vertmse, vertmse_weight)

        # compute texture loss
        if "trgbmse" in losslist or "trgbsqerr" in outputlist:
            weight = (validinput[:, None, None, None] * texmask[:, None, :, :].float()).expand_as(tex).contiguous()

            # re-standardize
            texrecstd = (decout["tex"] - self.texmean.to("cuda")) / self.texstd
            texstd = (tex - self.texmean.to("cuda")) / self.texstd

            texsqerr = weight * (texstd - texrecstd) ** 2

            if "trgbsqerr" in outputlist:
                resultout["trgbsqerr"] = texsqerr

            # texture rgb mean-squared-error
            if "trgbmse" in losslist:
                texmse = torch.sum(texsqerr.view(texsqerr.size(0), -1), dim=-1)
                texmse_weight = torch.sum(weight.view(weight.size(0), -1), dim=-1)

                resultlosses["trgbmse"] = (texmse, texmse_weight)

        # subsample depth, imagerec, imagerecmask
        if image is not None and pixelcoords.size()[1:3] != image.size()[2:4]:
            imagesize = torch.tensor(image.size()[3:1:-1], dtype=torch.float32, device=pixelcoords.device)
        else:
            imagesize = torch.tensor(pixelcoords.size()[2:0:-1], dtype=torch.float32, device=pixelcoords.device)

        samplecoords = pixelcoords * 2. / (imagesize[None, None, None, :] - 1.) - 1.

        # compute ray directions
        if self.cudaraydirs:
            raypos, raydir, tminmax = compute_raydirs(viewpos, viewrot, focal, princpt, pixelcoords, self.volradius)
        else:
            raydir = compute_raydirs_ref(pixelcoords, viewrot, focal, princpt)
            raypos, tminmax = compute_rmbounds(viewpos, raydir, self.volradius)

        if "dtstd" in renderoptions:
            renderoptions["dt"] = renderoptions["dt"] * \
                    torch.exp(torch.randn(1) * renderoptions.get("dtstd")).item()

        if renderoptions.get("unbiastminmax", False):
            stepsize = renderoptions["dt"] / self.volradius
            tminmax = torch.floor(tminmax / stepsize) * stepsize

        if renderoptions.get("tminmaxblocks", False):
            bx, by = renderoptions.get("blocksize", (8, 16))
            H, W = tminmax.size(1), tminmax.size(2)
            tminmax = tminmax.view(tminmax.size(0), H // by, by, W // bx, bx, 2)
            tminmax = tminmax.amin(dim=[2, 4], keepdim=True)
            tminmax = tminmax.repeat(1, 1, by, 1, bx, 1)
            tminmax = tminmax.view(tminmax.size(0), H, W, 2)

        # raymarch
        if "rmtime" in outputlist:
            torch.cuda.synchronize()
            rmstart = time.time()
        # rayrgba [6, 4, 384, 384]
        rayrgba, rmlosses = self.raymarcher(raypos, raydir, tminmax,
                decout=decout, renderoptions=renderoptions,
                trainiter=trainiter, evaliter=evaliter, losslist=losslist)
        resultlosses.update(rmlosses)
        if "rmtime" in outputlist:
            torch.cuda.synchronize()
            rmend = time.time()
            resultout["rmtime"] = rmend - rmstart

        if isinstance(rayrgba, tuple):
            rayrgb, rayalpha = rayrgba
        else:
            rayrgb, rayalpha = rayrgba[:, :3, :, :].contiguous(), rayrgba[:, 3:4, :, :].contiguous()

        # beta distribution prior on final opacity
        if "alphapr" in losslist:
            alphaprior = torch.mean(
                    torch.log(0.1 + rayalpha.view(rayalpha.size(0), -1)) +
                    torch.log(0.1 + 1. - rayalpha.view(rayalpha.size(0), -1)) - -2.20727, dim=-1)
            resultlosses["alphapr"] = alphaprior

        # color correction
        if camindex is not None and not renderoptions.get("nocolcorrect", False):
            rayrgb = self.colorcal(rayrgb, camindex)

        # background decoder
        if self.bgmodel is not None and not renderoptions.get("nobg", False):
            if "bgtime" in outputlist:
                torch.cuda.synchronize()
                bgstart = time.time()

            raypos, raydir, tminmax = compute_raydirs(campos, camrot, focal, princpt, pixelcoords, self.volradius)

            rayposbeg = raypos + raydir * tminmax[..., 0:1]
            rayposend = raypos + raydir * tminmax[..., 1:2]

            bg = self.bgmodel(bg, camindex, campos, rayposend, raydir, samplecoords, trainiter=trainiter)

        # alpha matting
        if bg is not None:
            rayrgb = rayrgb + (1. - rayalpha) * bg

            if "bg" in outputlist:
                resultout["bg"] = bg

            if "bgtime" in outputlist:
                torch.cuda.synchronize()
                bgend = time.time()
                resultout["bgtime"] = bgend - bgstart

        if "irgbrec" in outputlist:
            resultout["irgbrec"] = rayrgb
        if "irgbarec" in outputlist:
            resultout["irgbarec"] = torch.cat([rayrgb, rayalpha], dim=1)
        if "irgbflip" in outputlist:
            resultout["irgbflip"] = torch.cat([rayrgb[i:i+1] if i % 4 < 2 else image[i:i+1]
                for i in range(image.size(0))], dim=0)

        # image rgb loss
        if image is not None and trainiter > self.irgbmsestart:
            # subsample image
            if pixelcoords.size()[1:3] != image.size()[2:4]:
                image = F.grid_sample(image, samplecoords, align_corners=True)
                if imagemask is not None:
                    imagemask = F.grid_sample(imagemask, samplecoords, align_corners=True)

            # compute reconstruction loss weighting
            weight = torch.ones_like(image) * validinput[:, None, None, None]
            if imagevalid is not None:
                weight = weight * imagevalid[:, None, None, None]
            if imagemask is not None:
                weight = weight * imagemask

            if "irgbsqerr" in outputlist:
                irgbsqerr_nonorm = (weight * (image - rayrgb) ** 2).contiguous()
                resultout["irgbsqerr"] = torch.sqrt(irgbsqerr_nonorm.mean(dim=1, keepdim=True))

            # standardize
            rayrgb = (rayrgb - self.imagemean) / self.imagestd
            image = (image - self.imagemean) / self.imagestd

            irgbsqerr = (weight * (image - rayrgb) ** 2).contiguous()

            if "irgbmse" in losslist:
                irgbmse = torch.sum(irgbsqerr.view(irgbsqerr.size(0), -1), dim=-1)
                irgbmse_weight = torch.sum(weight.view(weight.size(0), -1), dim=-1)

                resultlosses["irgbmse"] = (irgbmse, irgbmse_weight)

        aeend = time.time()
        if "aetime" in outputlist:
            resultout["aetime"] = aeend - aestart

        return resultout, resultlosses
