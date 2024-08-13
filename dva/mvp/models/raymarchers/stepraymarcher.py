# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
""" Raymarching in pure pytorch """
import torch
import torch.nn as nn
import torch.nn.functional as F

class Raymarcher(nn.Module):
    def __init__(self, volradius):
        super(Raymarcher, self).__init__()

        self.volradius = volradius

    def forward(self, raypos, raydir, tminmax, decout,
            encoding=None, renderoptions={}, **kwargs):

        dt = renderoptions["dt"] / self.volradius

        tminmax = torch.floor(tminmax / dt) * dt

        t = tminmax[..., 0] + 0.
        raypos = raypos + raydir * t[..., None]

        rayrgb = torch.zeros_like(raypos.permute(0, 3, 1, 2)) # NCHW
        if "multaccum" in renderoptions and renderoptions["multaccum"]:
            lograyalpha = torch.zeros_like(rayrgb[:, 0:1, :, :]) # NCHW
        else:
            rayalpha = torch.zeros_like(rayrgb[:, 0:1, :, :]) # NCHW

        # raymarch
        done = torch.zeros_like(t).bool()
        while not done.all():
            valid = torch.prod((raypos > -1.) * (raypos < 1.), dim=-1).float()
            samplepos = F.grid_sample(decout["warp"][:, 0], raypos[:, None, :, :, :], align_corners=True).permute(0, 2, 3, 4, 1)
            val = F.grid_sample(decout["template"][:, 0], samplepos, align_corners=True)[:, :, 0, :, :]
            val = val * valid[:, None, :, :]
            sample_rgb, sample_alpha = val[:, :3, :, :], val[:, 3:, :, :]

            done = done | ((t + dt) >= tminmax[..., 1])

            if "multaccum" in renderoptions and renderoptions["multaccum"]:
                contrib = torch.exp(-lograyalpha) * (1. - torch.exp(-sample_alpha * dt))

                rayrgb = rayrgb + sample_rgb * contrib
                lograyalpha = lograyalpha + sample_alpha * dt
            else:
                contrib = ((rayalpha + sample_alpha * dt).clamp(max=1.) - rayalpha)

                rayrgb = rayrgb + sample_rgb * contrib
                rayalpha = rayalpha + contrib

            raypos = raypos + raydir * dt
            t = t + dt

        if "multaccum" in renderoptions and renderoptions["multaccum"]:
            rayalpha = 1. - torch.exp(-lograyalpha)

        rayrgba = torch.cat([rayrgb, rayalpha], dim=1)
        return rayrgba, {}
