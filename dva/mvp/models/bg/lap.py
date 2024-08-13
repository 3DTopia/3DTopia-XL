# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import models.utils

class ImageMod(nn.Module):
    def __init__(self, width, height, depth, buf=False):
        super(ImageMod, self).__init__()

        if buf:
            self.register_buffer("image", torch.randn(1, 3, depth, height, width) * 0.001, persistent=False)
        else:
            self.image = nn.Parameter(torch.randn(1, 3, depth, height, width) * 0.001)

    def forward(self, samplecoords):
        image = self.image.expand(samplecoords.size(0), -1, -1, -1, -1)
        return F.grid_sample(image, samplecoords, align_corners=True)

class LapImage(nn.Module):
    def __init__(self, width, height, depth, levels, startlevel=0, buftop=False, align_corners=True):
        super(LapImage, self).__init__()

        self.width : int = int(width)
        self.height : int = int(height)
        self.levels = levels
        self.startlevel = startlevel
        self.align_corners = align_corners

        self.pyr = nn.ModuleList(
                [ImageMod(self.width // 2 ** i, self.height // 2 ** i, depth)
                    for i in list(range(startlevel, levels - 1))[::-1]] +
                ([ImageMod(self.width, self.height, depth, buf=True)] if buftop else []))
        self.pyr0 = ImageMod(self.width // 2 ** (levels - 1), self.height // 2 ** (levels - 1), depth)

    def forward(self, samplecoords):
        image = self.pyr0(samplecoords)

        for i, layer in enumerate(self.pyr):
            image = image + layer(samplecoords)

        return image

class BGModel(nn.Module):
    def __init__(self, width, height, allcameras, bgdict=True, trainstart=0,
            levels=5, startlevel=0, buftop=False, align_corners=True):
        super(BGModel, self).__init__()

        self.allcameras = allcameras
        self.trainstart = trainstart

        if trainstart > -1:
            self.lap = LapImage(width, height, len(allcameras), levels=levels,
                    startlevel=startlevel, buftop=buftop,
                    align_corners=align_corners)

    def forward(
            self,
            bg : Optional[torch.Tensor]=None,
            camindex : Optional[torch.Tensor]=None,
            raypos : Optional[torch.Tensor]=None,
            rayposend : Optional[torch.Tensor]=None,
            raydir : Optional[torch.Tensor]=None,
            samplecoords : Optional[torch.Tensor]=None,
            trainiter : float=-1):
        if self.trainstart > -1 and trainiter >= self.trainstart and camindex is not None:
            assert samplecoords is not None
            assert camindex is not None

            samplecoordscam = torch.cat([
                samplecoords[:, None, :, :, :], # [B, 1, H, W, 2]
                ((camindex[:, None, None, None, None] * 2.) / (len(self.allcameras) - 1.) - 1.)
                    .expand(-1, -1, samplecoords.size(1), samplecoords.size(2), -1)],
                dim=-1) # [B, 1, H, W, 3]
            lap = self.lap(samplecoordscam)[:, :, 0, :, :]
        else:
            lap = None

        if lap is None:
            return None
        else:
            return F.softplus(lap)
