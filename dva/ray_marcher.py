# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Dict, Tuple
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import random

from dva.mvp.extensions.mvpraymarch.mvpraymarch import mvpraymarch
from dva.mvp.extensions.utils.utils import compute_raydirs

import logging

logger = logging.getLogger(__name__)


def convert_camera_parameters(Rt, K):
    R = Rt[:, :3, :3]
    t = -R.permute(0, 2, 1).bmm(Rt[:, :3, 3].unsqueeze(2)).squeeze(2)
    return dict(
        campos=t,
        camrot=R,
        focal=K[:, :2, :2],
        princpt=K[:, :2, 2],
    )

def subsample_pixel_coords(
    pixel_coords: th.Tensor, batch_size: int, ray_subsample_factor: int = 4
):

    H, W = pixel_coords.shape[:2]
    SW = W // ray_subsample_factor
    SH = H // ray_subsample_factor

    all_coords = []
    for _ in range(batch_size):
        # TODO: this is ugly, switch to pytorch?
        x0 = th.randint(0, ray_subsample_factor - 1, size=())
        y0 = th.randint(0, ray_subsample_factor - 1, size=())
        dx = ray_subsample_factor
        dy = ray_subsample_factor
        x1 = x0 + dx * SW
        y1 = y0 + dy * SH
        all_coords.append(pixel_coords[y0:y1:dy, x0:x1:dx, :])
    all_coords = th.stack(all_coords, dim=0)
    return all_coords


def resize_pixel_coords(
    pixel_coords: th.Tensor, batch_size: int, ray_subsample_factor: int = 4
):

    H, W = pixel_coords.shape[:2]
    SW = W // ray_subsample_factor
    SH = H // ray_subsample_factor

    all_coords = []
    for _ in range(batch_size):
        # TODO: this is ugly, switch to pytorch?
        x0, y0 = ray_subsample_factor // 2, ray_subsample_factor // 2
        dx = ray_subsample_factor
        dy = ray_subsample_factor
        x1 = x0 + dx * SW
        y1 = y0 + dy * SH
        all_coords.append(pixel_coords[y0:y1:dy, x0:x1:dx, :])
    all_coords = th.stack(all_coords, dim=0)
    return all_coords


class RayMarcher(nn.Module):
    def __init__(
        self,
        image_height,
        image_width,
        volradius,
        fadescale=8.0,
        fadeexp=8.0,
        dt=1.0,
        ray_subsample_factor=1,
        accum=2,
        termthresh=0.99,
        blocksize=None,
        with_t_img=True,
        chlast=False,
        assets=None,
    ):
        super().__init__()

        # TODO: add config?
        self.image_height = image_height
        self.image_width = image_width
        self.volradius = volradius
        self.dt = dt

        self.fadescale = fadescale
        self.fadeexp = fadeexp

        # NOTE: this seems to not work for other configs?
        if blocksize is None:
            blocksize = (8, 16)

        self.blocksize = blocksize
        self.with_t_img = with_t_img
        self.chlast = chlast

        self.accum = accum
        self.termthresh = termthresh

        base_pixel_coords = th.stack(
            th.meshgrid(
                th.arange(self.image_height, dtype=th.float32),
                th.arange(self.image_width, dtype=th.float32),
            )[::-1],
            dim=-1,
        )
        self.register_buffer("base_pixel_coords", base_pixel_coords, persistent=False)
        self.fixed_bvh_cache = {-1: (th.empty(0), th.empty(0), th.empty(0))}
        self.ray_subsample_factor = ray_subsample_factor

    def _set_pix_coords(self):
        dev = self.base_pixel_coords.device
        self.base_pixel_coords = th.stack(
            th.meshgrid(
                th.arange(self.image_height, dtype=th.float32, device=dev),
                th.arange(self.image_width, dtype=th.float32, device=dev),
            )[::-1],
            dim=-1,
        )

    def resize(self, h: int, w: int):
        self.image_height = h
        self.image_width = w

        self._set_pix_coords()

    def forward(
        self,
        prim_rgba: th.Tensor,
        prim_pos: th.Tensor,
        prim_rot: th.Tensor,
        prim_scale: th.Tensor,
        K: th.Tensor,
        RT: th.Tensor,
        ray_subsample_factor: Optional[int] = None,
    ):
        """
        Args:
            prim_rgba: primitive payload [B, K, 4, S, S, S],
                K - # of primitives, S - primitive size
            prim_pos: locations [B, K, 3]
            prim_rot: rotations [B, K, 3, 3]
            prim_scale: scales [B, K, 3]
            K: intrinsics [B, 3, 3]
            RT: extrinsics [B, 3, 4]
        Returns:
            a dict of tensors
        """
        # TODO: maybe we can re-use mvpraymarcher?
        B = prim_rgba.shape[0]
        device = prim_rgba.device

        # TODO: this should return focal 2x2?
        camera = convert_camera_parameters(RT, K)
        camera = {k: v.contiguous() for k, v in camera.items()}

        dt = self.dt / self.volradius

        if ray_subsample_factor is None:
            ray_subsample_factor = self.ray_subsample_factor

        if ray_subsample_factor > 1 and self.training:
            pixel_coords = subsample_pixel_coords(
                self.base_pixel_coords, int(B), ray_subsample_factor
            )
        elif ray_subsample_factor > 1:
            pixel_coords = resize_pixel_coords(
                self.base_pixel_coords,
                int(B),
                ray_subsample_factor,
            )
        else:
            pixel_coords = (
                self.base_pixel_coords[np.newaxis].expand(B, -1, -1, -1).contiguous()
            )

        prim_pos = prim_pos / self.volradius

        focal = th.diagonal(camera["focal"], dim1=1, dim2=2).contiguous()

        # TODO: port this?
        raypos, raydir, tminmax = compute_raydirs(
            viewpos=camera["campos"],
            viewrot=camera["camrot"],
            focal=focal,
            princpt=camera["princpt"],
            pixelcoords=pixel_coords,
            volradius=self.volradius,
        )

        rgba = mvpraymarch(
            raypos,
            raydir,
            stepsize=dt,
            tminmax=tminmax,
            algo=0,
            template=prim_rgba.permute(0, 1, 3, 4, 5, 2).contiguous(),
            warp=None,
            termthresh=self.termthresh,
            primtransf=(prim_pos, prim_rot, prim_scale),
            fadescale=self.fadescale,
            fadeexp=self.fadeexp,
            usebvh="fixedorder",
            chlast=True,
        )

        rgba = rgba.permute(0, 3, 1, 2)

        preds = {
            "rgba_image": rgba,
            "pixel_coords": pixel_coords,
        }

        return preds


def generate_colored_boxes(template, prim_rot, alpha=10000.0, seed=123456):
    B = template.shape[0]
    output = template.clone()
    device = template.device

    lightdir = -3 * th.ones([B, 3], dtype=th.float32, device=device)
    lightdir = lightdir / th.norm(lightdir, p=2, dim=1, keepdim=True)

    zz, yy, xx = th.meshgrid(
        th.linspace(-1.0, 1.0, template.size(-1), device=device),
        th.linspace(-1.0, 1.0, template.size(-1), device=device),
        th.linspace(-1.0, 1.0, template.size(-1), device=device),
    )
    primnormalx = th.where(
        (th.abs(xx) >= th.abs(yy)) & (th.abs(xx) >= th.abs(zz)),
        th.sign(xx) * th.ones_like(xx),
        th.zeros_like(xx),
    )
    primnormaly = th.where(
        (th.abs(yy) >= th.abs(xx)) & (th.abs(yy) >= th.abs(zz)),
        th.sign(yy) * th.ones_like(xx),
        th.zeros_like(xx),
    )
    primnormalz = th.where(
        (th.abs(zz) >= th.abs(xx)) & (th.abs(zz) >= th.abs(yy)),
        th.sign(zz) * th.ones_like(xx),
        th.zeros_like(xx),
    )
    primnormal = th.stack([primnormalx, -primnormaly, -primnormalz], dim=-1)
    primnormal = primnormal / th.sqrt(th.sum(primnormal**2, dim=-1, keepdim=True))

    output[:, :, 3, :, :, :] = alpha

    np.random.seed(seed)

    for i in range(template.size(1)):
        # generating a random color
        output[:, i, 0, :, :, :] = np.random.rand() * 255.0
        output[:, i, 1, :, :, :] = np.random.rand() * 255.0
        output[:, i, 2, :, :, :] = np.random.rand() * 255.0

        # get light direction in local coordinate system?
        lightdir0 = lightdir
        mult = th.sum(
            lightdir0[:, None, None, None, :] * primnormal[np.newaxis], dim=-1
        )[:, np.newaxis, :, :, :].clamp(min=0.2)
        output[:, i, :3, :, :, :] *= 1.4 * mult
    return output
