# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
""" Raymarcher for a mixture of volumetric primitives """
import os
import itertools
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from extensions.mvpraymarch.mvpraymarch import mvpraymarch

class Raymarcher(nn.Module):
    def __init__(self, volradius):
        super(Raymarcher, self).__init__()

        self.volradius = volradius

    def forward(self, raypos, raydir, tminmax, decout,
            encoding=None, renderoptions={}, trainiter=-1, evaliter=-1,
            rayterm=None,
            **kwargs):

        # rescale world
        dt = renderoptions["dt"] / self.volradius

        rayrgba = mvpraymarch(raypos, raydir, dt, tminmax,
                (decout["primpos"], decout["primrot"], decout["primscale"]),
                template=decout["template"],
                warp=decout["warp"] if "warp" in decout else None,
                rayterm=rayterm,
                **{k:v for k, v in renderoptions.items() if k in mvpraymarch.__code__.co_varnames})

        return rayrgba.permute(0, 3, 1, 2), {}
