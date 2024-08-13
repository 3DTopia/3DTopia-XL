# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
""" MVP decoder """
import math
from typing import Optional, Dict, List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import models.utils
from models.utils import LinearELR, ConvTranspose2dELR, ConvTranspose3dELR

@torch.jit.script
def compute_postex(geo, idxim, barim, volradius : float):
    # compute 3d coordinates of each texel in uv map
    return (
        barim[None, :, :, 0, None] * geo[:, idxim[:, :, 0], :] +
        barim[None, :, :, 1, None] * geo[:, idxim[:, :, 1], :] +
        barim[None, :, :, 2, None] * geo[:, idxim[:, :, 2], :]
        ).permute(0, 3, 1, 2) / volradius

@torch.jit.script
def compute_tbn(v0, v1, v2, vt0, vt1, vt2):
    v01 = v1 - v0
    v02 = v2 - v0
    vt01 = vt1 - vt0
    vt02 = vt2 - vt0
    f = 1. / (vt01[None, :, :, 0] * vt02[None, :, :, 1] - vt01[None, :, :, 1] * vt02[None, :, :, 0])
    tangent = f[:, :, :, None] * torch.stack([
        v01[:, :, :, 0] * vt02[None, :, :, 1] - v02[:, :, :, 0] * vt01[None, :, :, 1],
        v01[:, :, :, 1] * vt02[None, :, :, 1] - v02[:, :, :, 1] * vt01[None, :, :, 1],
        v01[:, :, :, 2] * vt02[None, :, :, 1] - v02[:, :, :, 2] * vt01[None, :, :, 1]], dim=-1)
    tangent = F.normalize(tangent, dim=-1)
    normal = torch.cross(v01, v02, dim=3)
    normal = F.normalize(normal, dim=-1)
    bitangent = torch.cross(tangent, normal, dim=3)
    bitangent = F.normalize(bitangent, dim=-1)

    # create matrix
    primrotmesh = torch.stack((tangent, bitangent, normal), dim=-1)

    return primrotmesh

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

# RGBA decoder
class SlabContentDecoder(nn.Module):
    def __init__(self, nprims, primsize, inch, outch, chstart=256, hstart=4,
            texwarp=False, elr=True, norm=None, mod=False, ub=True, upconv=None,
            penultch=None, use3dconv=False, reduced3dch=False):
        super(SlabContentDecoder, self).__init__()

        assert not texwarp
        assert upconv == None

        self.nprims = nprims
        self.primsize = primsize

        self.nprimy = int(math.sqrt(nprims))
        self.nprimx = nprims // self.nprimy
        assert nprims == self.nprimx * self.nprimy

        self.slabw = self.nprimx * primsize[0]
        self.slabh = self.nprimy * primsize[1]
        self.slabd =               primsize[2]

        nlayers = int(math.log2(min(self.slabw, self.slabh))) - int(math.log2(hstart))
        nlayers3d = int(math.log2(self.slabd))
        nlayers2d = nlayers - nlayers3d

        lastch = chstart
        dims = (1, hstart, hstart * self.nprimx // self.nprimy)

        layers = []
        layers.append(LinearELR(inch, chstart*dims[1]*dims[2], act=nn.LeakyReLU(0.2)))
        layers.append(Reshape(-1, chstart, dims[1], dims[2]))

        for i in range(nlayers):
            nextch = lastch if i % 2 == 0 else lastch // 2

            if use3dconv and reduced3dch and i >= nlayers2d:
                nextch //= 2

            if i == nlayers - 2 and penultch is not None:
                nextch = penultch

            if use3dconv and i >= nlayers2d:
                if i == nlayers2d:
                    layers.append(Reshape(-1, lastch, 1, dims[1], dims[2]))
                layers.append(ConvTranspose3dELR(
                    lastch,
                    (outch if i == nlayers - 1 else nextch),
                    4, 2, 1,
                    ub=(dims[0]*2, dims[1]*2, dims[2]*2) if ub else None,
                    norm=None if i == nlayers - 1 else norm,
                    act=None if i == nlayers - 1 else nn.LeakyReLU(0.2)
                    ))
            else:
                layers.append(ConvTranspose2dELR(
                    lastch,
                    (outch * primsize[2] if i == nlayers - 1 else nextch),
                    4, 2, 1,
                    ub=(dims[1]*2, dims[2]*2) if ub else None,
                    norm=None if i == nlayers - 1 else norm,
                    act=None if i == nlayers - 1 else nn.LeakyReLU(0.2)
                    ))

            lastch = nextch
            dims = (dims[0] * (2 if use3dconv and i >= nlayers2d else 1), dims[1] * 2, dims[2] * 2)

        self.mod = nn.Sequential(*layers)

    def forward(self, enc, renderoptions : Dict[str, str], trainiter : Optional[int]=None):
        x = self.mod(enc)

        algo = renderoptions.get("algo")
        chlast = renderoptions.get("chlast")

        if chlast is not None and bool(chlast):
            # reorder channels last
            if len(x.size()) == 5:
                outch = x.size(1)
                x = x.view(x.size(0), outch, self.primsize[2], self.nprimy, self.primsize[1], self.nprimx, self.primsize[0])
                x = x.permute(0, 3, 5, 2, 4, 6, 1)
                x = x.reshape(x.size(0), self.nprims, self.primsize[2], self.primsize[1], self.primsize[0], outch)
            else:
                outch = x.size(1) // self.primsize[2]
                x = x.view(x.size(0), self.primsize[2], outch, self.nprimy, self.primsize[1], self.nprimx, self.primsize[0])
                x = x.permute(0, 3, 5, 1, 4, 6, 2)
                x = x.reshape(x.size(0), self.nprims, self.primsize[2], self.primsize[1], self.primsize[0], outch)
        else:
            if len(x.size()) == 5:
                outch = x.size(1)
                x = x.view(x.size(0), outch, self.primsize[2], self.nprimy, self.primsize[1], self.nprimx, self.primsize[0])
                x = x.permute(0, 3, 5, 1, 2, 4, 6)
                x = x.reshape(x.size(0), self.nprims, outch, self.primsize[2], self.primsize[1], self.primsize[0])
            else:
                outch = x.size(1) // self.primsize[2]
                x = x.view(x.size(0), self.primsize[2], outch, self.nprimy, self.primsize[1], self.nprimx, self.primsize[0])
                x = x.permute(0, 3, 5, 2, 1, 4, 6)
                x = x.reshape(x.size(0), self.nprims, outch, self.primsize[2], self.primsize[1], self.primsize[0])

        return x

def get_dec(dectype, **kwargs):
    if dectype == "slab2d":
        return SlabContentDecoder(**kwargs, use3dconv=False)
    elif dectype == "slab2d3d":
        return SlabContentDecoder(**kwargs, use3dconv=True)
    elif dectype == "slab2d3dv2":
        return SlabContentDecoder(**kwargs, use3dconv=True, reduced3dch=True)
    else:
        raise

# motion model for the delta from mesh-based position/orientation
class DeconvMotionModel(nn.Module):
    def __init__(self, nprims, inch, outch, chstart=1024,
            norm=None, mod=False, elr=True):
        super(DeconvMotionModel, self).__init__()

        self.nprims = nprims

        self.nprimy = int(math.sqrt(nprims))
        self.nprimx = nprims // int(math.sqrt(nprims))
        assert nprims == self.nprimx * self.nprimy

        nlayers = int(math.log2(min(self.nprimx, self.nprimy)))

        ch0, ch1 = chstart, chstart // 2
        layers = []

        layers.append(LinearELR(inch, ch0, norm=norm, act=nn.LeakyReLU(0.2)))

        layers.append(Reshape(-1, ch0, 1, self.nprimx // self.nprimy))
        dims = (1, 1, self.nprimx // self.nprimy)

        for i in range(nlayers):
            layers.append(ConvTranspose2dELR(
                ch0,
                (outch if i == nlayers - 1 else ch1),
                4, 2, 1,
                norm=None if i == nlayers - 1 else norm,
                act=None if i == nlayers - 1 else nn.LeakyReLU(0.2)
                ))

            if ch0 == ch1:
                ch1 = ch0 // 2
            else:
                ch0 = ch1

        self.mod = nn.Sequential(*layers)

    def forward(self, encoding):
        out = self.mod(encoding)
        out = out.view(encoding.size(0), 9, -1).permute(0, 2, 1).contiguous()

        primposdelta = out[:, :, 0:3]
        primrvecdelta = out[:, :, 3:6]
        primscaledelta = out[:, :, 6:9]
        return primposdelta, primrvecdelta, primscaledelta

def get_motion(motiontype, **kwargs):
    if motiontype == "deconv":
        return DeconvMotionModel(**kwargs)
    else:
        raise

class Decoder(nn.Module):
    def __init__(self,
            vt,
            vertmean,
            vertstd,
            idxim,
            tidxim,
            barim,
            volradius,
            dectype="slab2d",
            nprims=512,
            primsize=(32, 32, 32),
            chstart=256,
            penultch=None,
            condsize=0,
            motiontype="deconv",
            warptype=None,
            warpprimsize=None,
            sharedrgba=False,
            norm=None,
            mod=False,
            elr=True,
            scalemult=2.,
            nogeo=False,
            notplateact=False,
            postrainstart=-1,
            alphatrainstart=-1,
            renderoptions={},
            **kwargs):
        """
        Parameters
        ----------
        vt : numpy.array [V, 2]
            mesh vertex texture coordinates
        vertmean : numpy.array [V, 3]
            mesh vertex position average (average over time)
        vertstd : float
            mesh vertex position standard deviation (over time)
        idxim : torch.Tensor
            texture map of triangle indices
        tidxim : torch.Tensor
            texture map of texture triangle indices
        barim : torch.Tensor
            texture map of barycentric coordinates
        volradius : float
            radius of bounding volume of scene
        dectype : string
            type of content decoder, options are "slab2d", "slab2d3d", "slab2d3dv2"
        nprims : int
            number of primitives
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

        self.nprims = nprims
        self.primsize = primsize

        self.motiontype = motiontype
        self.nogeo = nogeo
        self.notplateact = notplateact
        self.scalemult = scalemult

        self.enc = LinearELR(256 + condsize, 256)

        # vertex output
        if not self.nogeo:
            self.geobranch = LinearELR(256, vertmean.numel(), norm=None)

        # primitive motion delta decoder
        self.motiondec = get_motion(motiontype, nprims=nprims, inch=256, outch=9,
                norm=norm, mod=mod, elr=elr, **kwargs)

        # slab decoder (RGBA)
        if sharedrgba:
            self.rgbadec = get_dec(dectype, nprims=nprims, primsize=primsize,
                    inch=256+3, outch=4, norm=norm, mod=mod, elr=elr,
                    penultch=penultch, **kwargs)

            if renderoptions.get("half", False):
                self.rgbadec = self.rgbadec.half()

            if renderoptions.get("chlastconv", False):
                self.rgbadec = self.rgbadec.to(memory_format=torch.channels_last)
        else:
            self.rgbdec = get_dec(dectype, nprims=nprims, primsize=primsize,
                    inch=256+3, outch=3, chstart=chstart, norm=norm, mod=mod,
                    elr=elr, penultch=penultch, **kwargs)
            self.alphadec = get_dec(dectype, nprims=nprims, primsize=primsize,
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
            self.warpdec = get_dec(warptype, nprims=nprims, primsize=warpprimsize,
                    inch=256, outch=3, chstart=chstart, norm=norm, mod=mod, elr=elr, **kwargs)
        else:
            self.warpdec = None

        # vertex/triangle/mesh topology data
        if vt is not None:
            vt = torch.tensor(vt) if not isinstance(vt, torch.Tensor) else vt
            self.register_buffer("vt", vt, persistent=False)

        if vertmean is not None:
            self.register_buffer("vertmean", vertmean, persistent=False)
        self.vertstd = vertstd

        idxim = torch.tensor(idxim) if not isinstance(idxim, torch.Tensor) else idxim
        tidxim = torch.tensor(tidxim) if not isinstance(tidxim, torch.Tensor) else tidxim
        barim = torch.tensor(barim) if not isinstance(barim, torch.Tensor) else barim
        self.register_buffer("idxim", idxim.long(), persistent=False)
        self.register_buffer("tidxim", tidxim.long(), persistent=False)
        self.register_buffer("barim", barim, persistent=False)

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

        if int(math.sqrt(self.nprims)) ** 2 == self.nprims:
            nprimsy = int(math.sqrt(self.nprims))
        else:
            nprimsy = int(math.sqrt(self.nprims // 2))
        nprimsx = self.nprims // nprimsy

        assert nprimsx * nprimsy == self.nprims

        if not self.nogeo:
            # decode mesh vertices
            # geo [6, 7306, 3]
            geo = self.geobranch(encoding)
            geo = geo.view(encoding.size(0), -1, 3)
            geo = geo * self.vertstd + self.vertmean

            # placement of primitives on mesh
            uvheight, uvwidth = self.barim.size(0), self.barim.size(1)
            stridey = uvheight // nprimsy
            stridex = uvwidth // nprimsx

            # get subset of vertices and texture map coordinates to compute TBN matrix
            v0 = geo[:, self.idxim[stridey//2::stridey, stridex//2::stridex, 0], :]
            v1 = geo[:, self.idxim[stridey//2::stridey, stridex//2::stridex, 1], :]
            v2 = geo[:, self.idxim[stridey//2::stridey, stridex//2::stridex, 2], :]

            vt0 = self.vt[self.tidxim[stridey//2::stridey, stridex//2::stridex, 0], :]
            vt1 = self.vt[self.tidxim[stridey//2::stridey, stridex//2::stridex, 1], :]
            vt2 = self.vt[self.tidxim[stridey//2::stridey, stridex//2::stridex, 2], :]

            # [6, 256, 3]
            primposmesh = (
                    self.barim[None, stridey//2::stridey, stridex//2::stridex, 0, None] * v0 +
                    self.barim[None, stridey//2::stridey, stridex//2::stridex, 1, None] * v1 +
                    self.barim[None, stridey//2::stridey, stridex//2::stridex, 2, None] * v2
                    ).view(v0.size(0), self.nprims, 3) / self.volradius

            # compute TBN matrix
            # primrotmesh [6, 16, 16, 3, 3]
            primrotmesh = compute_tbn(v0, v1, v2, vt0, vt1, vt2)

            # decode motion deltas [6, 256, 3]
            primposdelta, primrvecdelta, primscaledelta = self.motiondec(encoding)
            if trainiter <= self.postrainstart:
                primposdelta = primposdelta * 0.
                primrvecdelta = primrvecdelta * 0.
                primscaledelta = primscaledelta * 0.

            # compose mesh transform with deltas
            primpos = primposmesh + primposdelta * 0.01
            primrotdelta = models.utils.axisangle_to_matrix(primrvecdelta * 0.01)
            primrot = torch.bmm(
                    primrotmesh.view(-1, 3, 3),
                    primrotdelta.view(-1, 3, 3)).view(encoding.size(0), self.nprims, 3, 3)
            primscale = (self.scalemult * int(self.nprims ** (1. / 3))) * torch.exp(primscaledelta * 0.01)

            primtransf = None
        else:
            geo = None

            # decode motion deltas
            primposdelta, primrvecdelta, primscaledelta = self.motiondec(encoding)
            if trainiter <= self.postrainstart:
                primposdelta = primposdelta * 0.
                primrvecdelta = primrvecdelta * 0.
                primscaledelta = primscaledelta * 0. + 1.

            primpos = primposdelta * 0.3
            primrotdelta = models.utils.axisangle_to_matrix(primrvecdelta * 0.3)
            primrot = torch.exp(primrotdelta * 0.01)
            primscale = (self.scalemult * int(self.nprims ** (1. / 3))) * primscaledelta

            primtransf = None

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
            # primrgb [6, 256, 32, 32, 32, 3] -> [B, 256, primsize, 3]
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
                torch.linspace(-1., 1., self.primsize[2], device=encoding.device),
                torch.linspace(-1., 1., self.primsize[1], device=encoding.device),
                torch.linspace(-1., 1., self.primsize[0], device=encoding.device))[::-1],
                dim=-1 if chlast is not None and bool(chlast) else 0)[None, None, :, :, :, :]
        else:
            warp = None

        # debugging / visualization
        viewaxes = renderoptions.get("viewaxes")
        colorprims = renderoptions.get("colorprims")
        viewslab = renderoptions.get("viewslab")

        # add axes to primitives
        if viewaxes is not None and bool(viewaxes):
            template[:, :, 3, template.size(3)//2:template.size(3)//2+1, template.size(4)//2:template.size(4)//2+1, :] = 2550.
            template[:, :, 0, template.size(3)//2:template.size(3)//2+1, template.size(4)//2:template.size(4)//2+1, :] = 2550.
            template[:, :, 3, template.size(3)//2:template.size(3)//2+1, :, template.size(5)//2:template.size(5)//2+1] = 2550.
            template[:, :, 1, template.size(3)//2:template.size(3)//2+1, :, template.size(5)//2:template.size(5)//2+1] = 2550.
            template[:, :, 3, :, template.size(4)//2:template.size(4)//2+1, template.size(5)//2:template.size(5)//2+1] = 2550.
            template[:, :, 2, :, template.size(4)//2:template.size(4)//2+1, template.size(5)//2:template.size(5)//2+1] = 2550.

        # give each primitive a unique color
        if colorprims is not None and bool(colorprims):
            lightdir = -torch.tensor([1., 1., 1.], device=template.device)
            lightdir = lightdir / torch.sqrt(torch.sum(lightdir ** 2))
            zz, yy, xx = torch.meshgrid(
                torch.linspace(-1., 1., self.primsize[2], device=template.device),
                torch.linspace(-1., 1., self.primsize[1], device=template.device),
                torch.linspace(-1., 1., self.primsize[0], device=template.device))
            primnormalx = torch.where(
                    (torch.abs(xx) >= torch.abs(yy)) & (torch.abs(xx) >= torch.abs(zz)),
                    torch.sign(xx) * torch.ones_like(xx),
                    torch.zeros_like(xx))
            primnormaly = torch.where(
                    (torch.abs(yy) >= torch.abs(xx)) & (torch.abs(yy) >= torch.abs(zz)),
                    torch.sign(yy) * torch.ones_like(xx),
                    torch.zeros_like(xx))
            primnormalz = torch.where(
                    (torch.abs(zz) >= torch.abs(xx)) & (torch.abs(zz) >= torch.abs(yy)),
                    torch.sign(zz) * torch.ones_like(xx),
                    torch.zeros_like(xx))
            primnormal = torch.stack([primnormalx, primnormaly, primnormalz], dim=-1)
            primnormal = F.normalize(primnormal, dim=-1)

            torch.manual_seed(123456)

            gridz, gridy, gridx = torch.meshgrid(
                    torch.linspace(-1., 1., self.primsize[2], device=encoding.device),
                    torch.linspace(-1., 1., self.primsize[1], device=encoding.device),
                    torch.linspace(-1., 1., self.primsize[0], device=encoding.device))
            grid = torch.stack([gridx, gridy, gridz], dim=-1)

            if chlast is not None and chlast:
                template[:] = torch.rand(1, template.size(1), 1, 1, 1, template.size(-1), device=template.device) * 255.
                template[:, :, :, :, :, 3] = 1000.
            else:
                template[:] = torch.rand(1, template.size(1), template.size(2), 1, 1, 1, device=template.device) * 255.
                template[:, :, 3, :, :, :] = 1000.

            if chlast is not None and chlast:
                lightdir0 = torch.sum(primrot[:, :, :, :] * lightdir[None, None, :, None], dim=-2)
                template[:, :, :, :, :, :3] *= 1.2 * torch.sum(
                        lightdir0[:, :, None, None, None, :] * primnormal, dim=-1)[:, :, :, :, :, None].clamp(min=0.05)
            else:
                lightdir0 = torch.sum(primrot[:, :, :, :] * lightdir[None, None, :, None], dim=-2)
                template[:, :, :3, :, :, :] *= 1.2 * torch.sum(
                        lightdir0[:, :, None, None, None, :] * primnormal, dim=-1)[:, :, None, :, :, :].clamp(min=0.05)

        # view slab as a 2d grid
        if viewslab is not None and bool(viewslab):
            assert evaliter is not None

            yy, xx = torch.meshgrid(
                    torch.linspace(0., 1., int(math.sqrt(self.nprims)), device=template.device),
                    torch.linspace(0., 1., int(math.sqrt(self.nprims)), device=template.device))
            primpos0 = torch.stack([xx*1.5, 0.75-yy*1.5, xx*0.+0.5], dim=-1)[None, :, :, :].repeat(template.size(0), 1, 1, 1).view(-1, self.nprims, 3)
            primrot0 = torch.eye(3, device=template.device)[None, None, :, :].repeat(template.size(0), self.nprims, 1, 1)
            primrot0.data[:, :, 1, 1] *= -1.
            primscale0 = torch.ones((template.size(0), self.nprims, 3), device=template.device) * math.sqrt(self.nprims) * 1.25 #* 0.5

            blend = ((evaliter - 256.) / 64.).clamp(min=0., max=1.)[:, None, None]
            blend = 3 * blend ** 2 - 2 * blend ** 3

            primpos = (1. - blend) * primpos0 + blend * primpos
            primrot = models.utils.rotation_interp(primrot0, primrot, blend)
            primscale = torch.exp((1. - blend) * torch.log(primscale0) + blend * torch.log(primscale))

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
        if primtransf is not None:
            result["primtransf"] = primtransf
        if warp is not None:
            result["warp"] = warp
        if geo is not None:
            result["verts"] = geo
        return result, losses
