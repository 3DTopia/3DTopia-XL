# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import cv2
import numpy as np
import copy
import importlib
from typing import Any, Dict

def load_module(module_name, class_name=None, silent: bool = False):
    module = importlib.import_module(module_name)
    return getattr(module, class_name) if class_name else module


def load_class(class_name):
    return load_module(*class_name.rsplit(".", 1))


def load_from_config(config, **kwargs):
    """Instantiate an object given a config and arguments."""
    assert "class_name" in config and "module_name" not in config
    config = copy.deepcopy(config)
    class_name = config.pop("class_name")
    object_class = load_class(class_name)
    return object_class(**config, **kwargs)


def load_opencv_calib(extrin_path, intrin_path):
    cameras = {}

    fse = cv2.FileStorage()
    fse.open(extrin_path, cv2.FileStorage_READ)

    fsi = cv2.FileStorage()
    fsi.open(intrin_path, cv2.FileStorage_READ)

    names = [
        fse.getNode("names").at(c).string() for c in range(fse.getNode("names").size())
    ]

    for camera in names:
        rot = fse.getNode(f"R_{camera}").mat()
        R = fse.getNode(f"Rot_{camera}").mat()
        T = fse.getNode(f"T_{camera}").mat()
        R_pred = cv2.Rodrigues(rot)[0]
        assert np.all(np.isclose(R_pred, R))
        K = fsi.getNode(f"K_{camera}").mat()
        cameras[camera] = {
            "Rt": np.concatenate([R, T], axis=1).astype(np.float32),
            "K": K.astype(np.float32),
        }
    return cameras