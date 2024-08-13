# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, List

import numpy as np

import torch
import torch.nn as nn

from models.utils import LinearELR, Conv2dELR

class Encoder(torch.nn.Module):
    def __init__(self, latentdim=256, hiq=True, texin=True,
            conv=Conv2dELR, lin=LinearELR,
            demod=True, texsize=1024, vertsize=21918):
        super(Encoder, self).__init__()

        self.latentdim = latentdim

        self.vertbranch = lin(vertsize, 256, norm="demod", act=nn.LeakyReLU(0.2))
        if texin:
            cm = 2 if hiq else 1

            layers = []
            chout = 128*cm
            chin = 128*cm
            nlayers = int(np.log2(texsize)) - 2
            for i in range(nlayers):
                if i == nlayers - 1:
                    chin = 3
                layers.append(
                    conv(chin, chout, 4, 2, 1, norm="demod" if demod else None, act=nn.LeakyReLU(0.2)))
                if chin == chout:
                    chin = chout // 2
                else:
                    chout = chin

            self.texbranch1 = nn.Sequential(*(layers[::-1]))

            self.texbranch2 = lin(cm*128*4*4, 256, norm="demod", act=nn.LeakyReLU(0.2))
            self.mu = lin(512, self.latentdim)
            self.logstd = lin(512, self.latentdim)
        else:
            self.mu = lin(256, self.latentdim)
            self.logstd = lin(256, self.latentdim)

    def forward(self, verts, texture : Optional[torch.Tensor]=None, losslist : Optional[List[str]]=None):
        assert losslist is not None

        x = self.vertbranch(verts.view(verts.size(0), -1))
        if texture is not None:
            texture = self.texbranch1(texture).reshape(verts.size(0), -1)
            texture = self.texbranch2(texture)
            x = torch.cat([x, texture], dim=1)

        mu, logstd = self.mu(x) * 0.1, self.logstd(x) * 0.01
        if self.training:
            z = mu + torch.exp(logstd) * torch.randn_like(logstd)
        else:
            z = mu

        losses = {}
        if "kldiv" in losslist:
            losses["kldiv"] = torch.mean(-0.5 - logstd + 0.5 * mu ** 2 + 0.5 * torch.exp(2 * logstd), dim=-1)

        return {"encoding": z}, losses
