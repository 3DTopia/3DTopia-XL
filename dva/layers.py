# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch as th
import torch.nn as nn

import numpy as np

from dva.mvp.models.utils import Conv2dWN, Conv2dWNUB, ConvTranspose2dWNUB, initmod


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        size,
        lrelu_slope=0.2,
        kernel_size=3,
        padding=1,
        wnorm_dim=0,
    ):
        super().__init__()

        self.conv_resize = Conv2dWN(in_channels, out_channels, kernel_size=1)
        self.conv1 = Conv2dWNUB(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            height=size,
            width=size,
        )

        self.lrelu1 = nn.LeakyReLU(lrelu_slope)
        self.conv2 = Conv2dWNUB(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            height=size,
            width=size,
        )
        self.lrelu2 = nn.LeakyReLU(lrelu_slope)

    def forward(self, x):
        x_skip = self.conv_resize(x)
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)
        return x + x_skip


def tile2d(x, size: int):
    """Tile a given set of features into a convolutional map.

    Args:
        x: float tensor of shape [N, F]
        size: int or a tuple

    Returns:
        a feature map [N, F, size[0], size[1]]
    """
    # size = size if isinstance(size, tuple) else (size, size)
    # NOTE: expecting only int here (!!!)
    return x[:, :, np.newaxis, np.newaxis].expand(-1, -1, size, size)


def weights_initializer(m, alpha: float = 1.0):
    return initmod(m, nn.init.calculate_gain("leaky_relu", alpha))


class UNetWB(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        size,
        n_init_ftrs=8,
        out_scale=0.1,
    ):
        # super().__init__(*args, **kwargs)
        super().__init__()

        self.out_scale = 0.1

        F = n_init_ftrs

        # TODO: allow changing the size?
        self.size = size

        self.down1 = nn.Sequential(
            Conv2dWNUB(in_channels, F, self.size // 2, self.size // 2, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.down2 = nn.Sequential(
            Conv2dWNUB(F, 2 * F, self.size // 4, self.size // 4, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.down3 = nn.Sequential(
            Conv2dWNUB(2 * F, 4 * F, self.size // 8, self.size // 8, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.down4 = nn.Sequential(
            Conv2dWNUB(4 * F, 8 * F, self.size // 16, self.size // 16, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.down5 = nn.Sequential(
            Conv2dWNUB(8 * F, 16 * F, self.size // 32, self.size // 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.up1 = nn.Sequential(
            ConvTranspose2dWNUB(
                16 * F, 8 * F, self.size // 16, self.size // 16, 4, 2, 1
            ),
            nn.LeakyReLU(0.2),
        )
        self.up2 = nn.Sequential(
            ConvTranspose2dWNUB(8 * F, 4 * F, self.size // 8, self.size // 8, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.up3 = nn.Sequential(
            ConvTranspose2dWNUB(4 * F, 2 * F, self.size // 4, self.size // 4, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.up4 = nn.Sequential(
            ConvTranspose2dWNUB(2 * F, F, self.size // 2, self.size // 2, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.up5 = nn.Sequential(
            ConvTranspose2dWNUB(F, F, self.size, self.size, 4, 2, 1), nn.LeakyReLU(0.2)
        )
        self.out = Conv2dWNUB(
            F + in_channels, out_channels, self.size, self.size, kernel_size=1
        )
        self.apply(lambda x: initmod(x, 0.2))
        initmod(self.out, 1.0)

    def forward(self, x):
        x1 = x
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        # TODO: switch to concat?
        x = self.up1(x6) + x5
        x = self.up2(x) + x4
        x = self.up3(x) + x3
        x = self.up4(x) + x2
        x = self.up5(x)
        x = th.cat([x, x1], dim=1)
        return self.out(x) * self.out_scale
