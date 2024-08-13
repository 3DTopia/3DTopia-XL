# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
import torch.nn as nn
from torchvision.models import vgg19
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg19_network = vgg19(pretrained=True)
        # vgg19_network.load_state_dict(state_dict)
        vgg_pretrained_features = vgg19_network.features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLossMasked(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.vgg = Vgg19()
        if weights is None:
            # self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
            self.weights = [20.0, 5.0, 0.9, 0.5, 0.5]
        else:
            self.weights = weights

    def normalize(self, batch):
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return ((batch / 255.0).clamp(0.0, 1.0) - mean) / std

    def forward(self, x_rgb, y_rgb, mask):

        x_norm = self.normalize(x_rgb)
        y_norm = self.normalize(y_rgb)

        x_vgg = self.vgg(x_norm)
        y_vgg = self.vgg(y_norm)
        loss = 0
        for i in range(len(x_vgg)):
            if isinstance(mask, th.Tensor):
                m = F.interpolate(
                    mask, size=(x_vgg[i].shape[-2], x_vgg[i].shape[-1]), mode="bilinear"
                ).detach()
            else:
                m = mask

            vx = x_vgg[i] * m
            vy = y_vgg[i] * m

            loss += self.weights[i] * (vx - vy).abs().mean()

            # logger.info(
            #     f"loss for {i}, {loss.item()} vx={vx.shape} vy={vy.shape} {vx.max()} {vy.max()}"
            # )
        return loss
