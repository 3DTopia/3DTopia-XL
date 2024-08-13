# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup

from torch.utils.cpp_extension import CUDAExtension, BuildExtension

if __name__ == "__main__":
    import torch
    setup(
        name="utils",
        ext_modules=[
            CUDAExtension(
                "utilslib",
                sources=["utils.cpp", "utils_kernel.cu"],
                extra_compile_args={
                    "nvcc": [
                        "-arch=sm_70",
                        "-std=c++14",
                        "-lineinfo",
                    ]
                }
            )
        ],
        cmdclass={"build_ext": BuildExtension}
    )
