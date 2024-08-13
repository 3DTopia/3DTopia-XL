# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import BufferDict, Conv2dELR

class BGModel(nn.Module):
    def __init__(self, width, height, allcameras, bgdict=True, demod=True, trainstart=0):
        super(BGModel, self).__init__()

        self.allcameras = allcameras
        self.trainstart = trainstart

        if bgdict:
            self.bg = BufferDict({k: torch.ones(3, height, width) for k in allcameras})
        else:
            self.bg = None

        if trainstart > -1:
            self.mlp1 = nn.Sequential(
                    Conv2dELR(60+24, 256, 1, 1, 0, demod="demod" if demod else None), nn.LeakyReLU(0.2),
                    Conv2dELR(  256, 256, 1, 1, 0, demod="demod" if demod else None), nn.LeakyReLU(0.2),
                    Conv2dELR(  256, 256, 1, 1, 0, demod="demod" if demod else None), nn.LeakyReLU(0.2),
                    Conv2dELR(  256, 256, 1, 1, 0, demod="demod" if demod else None), nn.LeakyReLU(0.2),
                    Conv2dELR(  256, 256, 1, 1, 0, demod="demod" if demod else None))

            self.mlp2 = nn.Sequential(
                    Conv2dELR(60+24+256, 256, 1, 1, 0, demod="demod" if demod else None), nn.LeakyReLU(0.2),
                    Conv2dELR(      256, 256, 1, 1, 0, demod="demod" if demod else None), nn.LeakyReLU(0.2),
                    Conv2dELR(      256, 256, 1, 1, 0, demod="demod" if demod else None), nn.LeakyReLU(0.2),
                    Conv2dELR(      256,   3, 1, 1, 0, demod=False))

    def forward(self, bg=None, camindex=None, raypos=None, rayposend=None,
            raydir=None, samplecoords=None, trainiter=-1, **kwargs):
        if self.trainstart > -1 and trainiter >= self.trainstart:# and camindex is not None:
            # generate position encoding
            posenc = torch.cat([
                torch.sin(2 ** i * np.pi * rayposend[:, :, :, :])
                for i in range(10)] + [
                torch.cos(2 ** i * np.pi * rayposend[:, :, :, :])
                for i in range(10)], dim=-1).permute(0, 3, 1, 2)

            direnc = torch.cat([
                torch.sin(2 ** i * np.pi * raydir[:, :, :, :])
                for i in range(4)] + [
                torch.cos(2 ** i * np.pi * raydir[:, :, :, :])
                for i in range(4)], dim=-1).permute(0, 3, 1, 2)

            decout = torch.cat([posenc, direnc], dim=1)
            decout = self.mlp1(decout)

            decout = torch.cat([posenc, direnc, decout], dim=1)
            decout = self.mlp2(decout)
        else:
            decout = None

        if bg is None and self.bg is not None and camindex is not None:
            bg = torch.stack([self.bg[self.allcameras[camindex[i].item()]] for i in range(camindex.size(0))], dim=0)
        else:
            bg = None

        if bg is not None and samplecoords is not None:
            if samplecoords.size()[1:3] != bg.size()[2:4]:
                bg = F.grid_sample(bg, samplecoords, align_corners=False)

        if decout is not None:
            if bg is not None:
                return F.softplus(bg + decout)
            else:
                return F.softplus(decout)
        else:
            if bg is not None:
                return F.softplus(bg)
            else:
                return None
