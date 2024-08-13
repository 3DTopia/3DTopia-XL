# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, List

import torch
import torch.nn as nn

from models.utils import LinearELR, Conv2dELR, Downsample2d

class Encoder(torch.nn.Module):
    def __init__(self, ninputs, size, nlayers=7, conv=Conv2dELR, lin=LinearELR):
        super(Encoder, self).__init__()

        self.ninputs = ninputs
        height, width = size
        self.nlayers = nlayers

        ypad = ((height + 2 ** nlayers - 1) // 2 ** nlayers) * 2 ** nlayers - height
        xpad = ((width + 2 ** nlayers - 1) // 2 ** nlayers) * 2 ** nlayers - width
        self.pad = nn.ZeroPad2d((xpad // 2, xpad - xpad // 2, ypad // 2, ypad - ypad // 2))

        self.downwidth = ((width + 2 ** nlayers - 1) // 2 ** nlayers)
        self.downheight = ((height + 2 ** nlayers - 1) // 2 ** nlayers)

        # compile layers
        layers = []
        inch, outch = 3, 64
        for i in range(nlayers):
            layers.append(conv(inch, outch, 4, 2, 1, norm="demod", act=nn.LeakyReLU(0.2)))

            if inch == outch:
                outch = inch * 2
            else:
                inch = outch
            if outch > 256:
                outch = 256

        self.down1 = nn.ModuleList([nn.Sequential(*layers)
                for i in range(self.ninputs)])
        self.down2 = lin(256 * self.ninputs * self.downwidth * self.downheight, 512, norm="demod", act=nn.LeakyReLU(0.2))
        self.mu = lin(512, 256)
        self.logstd = lin(512, 256)

    def forward(self, x, losslist : Optional[List[str]]=None):
        assert losslist is not None

        x = self.pad(x)
        x = [self.down1[i](x[:, i*3:(i+1)*3, :, :]).view(x.size(0), 256 * self.downwidth * self.downheight)
                for i in range(self.ninputs)]
        x = torch.cat(x, dim=1)
        x = self.down2(x)

        mu, logstd = self.mu(x) * 0.1, self.logstd(x) * 0.01
        if self.training:
            z = mu + torch.exp(logstd) * torch.randn(*logstd.size(), device=logstd.device)
        else:
            z = mu

        losses = {}
        if "kldiv" in losslist:
            losses["kldiv"] = torch.mean(-0.5 - logstd + 0.5 * mu ** 2 + 0.5 * torch.exp(2 * logstd), dim=-1)

        return {"encoding": z}, losses
