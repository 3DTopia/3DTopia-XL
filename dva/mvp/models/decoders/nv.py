# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Neural Volumes decoder """
import math
from typing import Optional, Dict, List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import models.utils
from models.utils import LinearELR, ConvTranspose2dELR, ConvTranspose3dELR

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class ContentDecoder(nn.Module):
    def __init__(self, primsize, inch, outch, chstart=256, hstart=4,
            texwarp=False, elr=True, norm=None, mod=False, ub=True, upconv=None,
            penultch=None):
        super(ContentDecoder, self).__init__()

        assert not texwarp
        assert upconv == None

        self.primsize = primsize

        nlayers = int(math.log2(self.primsize / hstart))

        lastch = chstart
        dims = (hstart, hstart, hstart)

        layers = []
        layers.append(LinearELR(inch, chstart*dims[0]*dims[1]*dims[2], act=nn.LeakyReLU(0.2)))
        layers.append(Reshape(-1, chstart, dims[0], dims[1], dims[2]))

        for i in range(nlayers):
            nextch = lastch if i % 2 == 0 else lastch // 2

            if i == nlayers - 2 and penultch is not None:
                nextch = penultch

            layers.append(ConvTranspose3dELR(
                lastch,
                (outch if i == nlayers - 1 else nextch),
                4, 2, 1,
                ub=(dims[0]*2, dims[1]*2, dims[2]*2) if ub else None,
                norm=None if i == nlayers - 1 else norm,
                act=None if i == nlayers - 1 else nn.LeakyReLU(0.2)
                ))

            lastch = nextch
            dims = (dims[0] * 2, dims[1] * 2, dims[2] * 2)

        self.mod = nn.Sequential(*layers)

    def forward(self, enc, renderoptions : Dict[str, str], trainiter : Optional[int]=None):
        x = self.mod(enc)

        algo = renderoptions.get("algo")
        chlast = renderoptions.get("chlast")

        if chlast is not None and bool(chlast):
            # reorder channels last
            outch = x.size(1)
            x = x.permute(0, 2, 3, 4, 1)[:, None, :, :, :, :].contiguous()
        else:
            outch = x.size(1)
            x = x[:, None, :, :, :, :].contiguous()

        return x

def get_dec(dectype, **kwargs):
    if dectype == "conv":
        return ContentDecoder(**kwargs)
    else:
        raise

class Decoder(nn.Module):
    def __init__(self,
            volradius,
            dectype="conv",
            primsize=128,
            chstart=256,
            penultch=None,
            condsize=0,
            warptype="conv",
            warpprimsize=32,
            sharedrgba=False,
            norm=None,
            mod=False,
            elr=True,
            notplateact=False,
            postrainstart=-1,
            alphatrainstart=-1,
            renderoptions={},
            **kwargs):
        """
        Parameters
        ----------
        volradius : float
            radius of bounding volume of scene
        dectype : string
            type of content decoder, options are "slab2d", "slab2d3d", "slab2d3dv2"
        primsize : Tuple[int, int, int]
            size of primitive dimensions
        postrainstart : int
            training iterations to start learning position, rotation, and
            scaling (i.e., primitives stay frozen until this iteration number)
        condsize : int
            unused
        motiontype : string
            motion model, options are "linear" and "deconv"
        warptype : string
            warp model, options are "same" to use same architecture as content
            or None
        sharedrgba : bool
            True to use 1 branch to output rgba, False to use 1 branch for rgb
            and 1 branch for alpha
        """
        super(Decoder, self).__init__()

        self.volradius = volradius
        self.postrainstart = postrainstart
        self.alphatrainstart = alphatrainstart

        self.primsize = primsize
        self.warpprimsize = warpprimsize

        self.notplateact = notplateact

        self.enc = LinearELR(256 + condsize, 256)

        # slab decoder (RGBA)
        if sharedrgba:
            self.rgbadec = get_dec(dectype, primsize=primsize,
                    inch=256+3, outch=4, norm=norm, mod=mod, elr=elr,
                    penultch=penultch, **kwargs)

            if renderoptions.get("half", False):
                self.rgbadec = self.rgbadec.half()

            if renderoptions.get("chlastconv", False):
                self.rgbadec = self.rgbadec.to(memory_format=torch.channels_last)
        else:
            self.rgbdec = get_dec(dectype, primsize=primsize,
                    inch=256+3, outch=3, chstart=chstart, norm=norm, mod=mod,
                    elr=elr, penultch=penultch, **kwargs)
            self.alphadec = get_dec(dectype, primsize=primsize,
                    inch=256, outch=1, chstart=chstart, norm=norm, mod=mod,
                    elr=elr, penultch=penultch, **kwargs)
            self.rgbadec = None

            if renderoptions.get("half", False):
                self.rgbdec = self.rgbdec.half()
                self.alphadec = self.alphadec.half()

            if renderoptions.get("chlastconv", False):
                self.rgbdec = self.rgbdec.to(memory_format=torch.channels_last)
                self.alphadec = self.alphadec.to(memory_format=torch.channels_last)

        # warp field decoder
        if warptype is not None:
            self.warpdec = get_dec(warptype, primsize=warpprimsize,
                    inch=256, outch=3, chstart=chstart, norm=norm, mod=mod, elr=elr, **kwargs)
        else:
            self.warpdec = None

    def forward(self,
            encoding,
            viewpos,
            condinput : Optional[torch.Tensor]=None,
            renderoptions : Optional[Dict[str, str]]=None,
            trainiter : int=-1,
            evaliter : Optional[torch.Tensor]=None,
            losslist : Optional[List[str]]=None,
            modelmatrix : Optional[torch.Tensor]=None):
        """
        Parameters
        ----------
        encoding : torch.Tensor [B, 256]
            Encoding of current frame
        viewpos : torch.Tensor [B, 3]
            Viewing position of target camera view
        condinput : torch.Tensor [B, ?]
            Additional conditioning input (e.g., headpose)
        renderoptions : dict
            Options for rendering (e.g., rendering debug images)
        trainiter : int,
            Current training iteration
        losslist : list,
            List of losses to compute and return

        Returns
        -------
        result : dict,
            Contains predicted vertex positions, primitive contents and
            locations, scaling, and orientation, and any losses.
        """
        assert renderoptions is not None
        assert losslist is not None

        if condinput is not None:
            encoding = torch.cat([encoding, condinput], dim=1)

        encoding = self.enc(encoding)

        viewdirs = F.normalize(viewpos, dim=1)

        primpos = torch.zeros(encoding.size(0), 1, 3, device=encoding.device)
        primrot = torch.eye(3, device=encoding.device)[None, None, :, :].repeat(encoding.size(0), 1, 1, 1)
        primscale = torch.ones(encoding.size(0), 1, 3, device=encoding.device)

        # options
        algo = renderoptions.get("algo")
        chlast = renderoptions.get("chlast")
        half = renderoptions.get("half")

        if self.rgbadec is not None:
            # shared rgb and alpha branch
            scale = torch.tensor([25., 25., 25., 1.], device=encoding.device)
            bias = torch.tensor([100., 100., 100., 0.], device=encoding.device)
            if chlast is not None and bool(chlast):
                scale = scale[None, None, None, None, None, :]
                bias = bias[None, None, None, None, None, :]
            else:
                scale = scale[None, None, :, None, None, None]
                bias = bias[None, None, :, None, None, None]

            templatein = torch.cat([encoding, viewdirs], dim=1)
            if half is not None and bool(half):
                templatein = templatein.half()
            template = self.rgbadec(templatein, trainiter=trainiter, renderoptions=renderoptions)
            template = bias + scale * template
            if not self.notplateact:
                template = F.relu(template)
            if half is not None and bool(half):
                template = template.float()
        else:
            templatein = torch.cat([encoding, viewdirs], dim=1)
            if half is not None and bool(half):
                templatein = templatein.half()
            primrgb = self.rgbdec(templatein, trainiter=trainiter, renderoptions=renderoptions)
            primrgb = primrgb * 25. + 100.
            if not self.notplateact:
                primrgb = F.relu(primrgb)

            templatein = encoding
            if half is not None and bool(half):
                templatein = templatein.half()
            primalpha = self.alphadec(templatein, trainiter=trainiter, renderoptions=renderoptions)
            if not self.notplateact:
                primalpha = F.relu(primalpha)

            if trainiter <= self.alphatrainstart:
                primalpha = primalpha * 0. + 1.
        
            if algo is not None and int(algo) == 4:
                template = torch.cat([primrgb, primalpha], dim=-1)
            elif chlast is not None and bool(chlast):
                template = torch.cat([primrgb, primalpha], dim=-1)
            else:
                template = torch.cat([primrgb, primalpha], dim=2)
            if half is not None and bool(half):
                template = template.float()

        if self.warpdec is not None:
            warp = self.warpdec(encoding, trainiter=trainiter, renderoptions=renderoptions) * 0.01
            warp = warp + torch.stack(torch.meshgrid(
                torch.linspace(-1., 1., self.warpprimsize, device=encoding.device),
                torch.linspace(-1., 1., self.warpprimsize, device=encoding.device),
                torch.linspace(-1., 1., self.warpprimsize, device=encoding.device))[::-1],
                dim=-1 if chlast is not None and bool(chlast) else 0)[None, None, :, :, :, :]
            warp = warp.contiguous()
        else:
            warp = None

        losses = {}

        # prior on primitive volume
        if "primvolsum" in losslist:
            losses["primvolsum"] = torch.sum(torch.prod(1. / primscale, dim=-1), dim=-1)

        if "logprimscalevar" in losslist:
            logprimscale = torch.log(primscale)
            logprimscalemean = torch.mean(logprimscale, dim=1, keepdim=True)
            losses["logprimscalevar"] = torch.mean((logprimscale - logprimscalemean) ** 2)

        result = {
                "template": template,
                "primpos": primpos,
                "primrot": primrot,
                "primscale": primscale}
        if warp is not None:
            result["warp"] = warp
        return result, losses
