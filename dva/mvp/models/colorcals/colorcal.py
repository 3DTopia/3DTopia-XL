# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F

class Colorcal(nn.Module):
    def __init__(self, allcameras):
        super(Colorcal, self).__init__()

        self.allcameras = allcameras

        self.weight = nn.Parameter(
                torch.ones(len(self.allcameras), 3))
        self.bias = nn.Parameter(
                torch.zeros(len(self.allcameras), 3))

    def forward(self, image, camindex):
        # collect weights
        weight = self.weight[camindex]
        bias = self.bias[camindex]

        # reshape
        b = image.size(0)
        groups = b * 3
        image = image.view(1, -1, image.size(2), image.size(3))
        weight = weight.view(-1, 1, 1, 1)
        bias = bias.view(-1)

        # conv
        result = F.conv2d(image, weight, bias, groups=groups)

        # unshape
        result = result.view(b, 3, image.size(2), image.size(3))
        return result

    def parameters(self):
        for p in super(Colorcal, self).parameters():
            if p.requires_grad:
                yield p
