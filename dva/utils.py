# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import torch as th
import torch.nn as nn
import torch.nn.functional as F

import cv2


def label_image(
    image,
    label,
    font_scale=1.0,
    font_thickness=1,
    label_origin=(10, 64),
    font_color=(255, 255, 255),
    font=cv2.FONT_HERSHEY_SIMPLEX,
):
    text_size, baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    image[
        label_origin[1] - text_size[1] : label_origin[1] + baseline,
        label_origin[0] : label_origin[0] + text_size[0],
    ] = (255 - font_color[0], 255 - font_color[1], 255 - font_color[2])
    cv2.putText(
        image, label, label_origin, font, font_scale, font_color, font_thickness
    )
    return image


def to_device(values, device=None, non_blocking=True):
    """Transfer a set of values to the device.
    Args:
        values: a nested dict/list/tuple of tensors
        device: argument to `to()` for the underlying vector
    NOTE:
        if the device is not specified, using `th.cuda()`
    """
    if device is None:
        device = th.device("cuda")

    if isinstance(values, dict):
        return {k: to_device(v, device=device) for k, v in values.items()}
    elif isinstance(values, tuple):
        return tuple(to_device(v, device=device) for v in values)
    elif isinstance(values, list):
        return [to_device(v, device=device) for v in values]
    elif isinstance(values, th.Tensor):
        return values.to(device, non_blocking=non_blocking)
    elif isinstance(values, nn.Module):
        return values.to(device)
    elif isinstance(values, np.ndarray):
        return th.from_numpy(values).to(device)
    else:
        return values
