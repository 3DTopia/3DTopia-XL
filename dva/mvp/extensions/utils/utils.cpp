// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <vector>

void compute_raydirs_forward_cuda(
        int N, int H, int W,
        float * viewposim,
        float * viewrotim,
        float * focalim,
        float * princptim,
        float * pixelcoordsim,
        float volradius,
        float * raypos,
        float * raydir,
        float * tminmax,
        cudaStream_t stream);

void compute_raydirs_backward_cuda(
        int N, int H, int W,
        float * viewposim,
        float * viewrotim,
        float * focalim,
        float * princptim,
        float * pixelcoordsim,
        float volradius,
        float * raypos,
        float * raydir,
        float * tminmax,
        float * grad_viewposim,
        float * grad_viewrotim,
        float * grad_focalim,
        float * grad_princptim,
        cudaStream_t stream);

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA((x)); CHECK_CONTIGUOUS((x))

std::vector<torch::Tensor> compute_raydirs_forward(
        torch::Tensor viewposim,
        torch::Tensor viewrotim,
        torch::Tensor focalim,
        torch::Tensor princptim,
        torch::optional<torch::Tensor> pixelcoordsim,
        int W, int H,
        float volradius,
        torch::Tensor rayposim,
        torch::Tensor raydirim,
        torch::Tensor tminmaxim) {
    CHECK_INPUT(viewposim);
    CHECK_INPUT(viewrotim);
    CHECK_INPUT(focalim);
    CHECK_INPUT(princptim);
    if (pixelcoordsim) { CHECK_INPUT(*pixelcoordsim); }
    CHECK_INPUT(rayposim);
    CHECK_INPUT(raydirim);
    CHECK_INPUT(tminmaxim);

    int N = viewposim.size(0);
    assert(!pixelcoordsim || (pixelcoordsim.size(1) == H && pixelcoordsim.size(2) == W));

    compute_raydirs_forward_cuda(N, H, W,
            reinterpret_cast<float *>(viewposim.data_ptr()),
            reinterpret_cast<float *>(viewrotim.data_ptr()),
            reinterpret_cast<float *>(focalim.data_ptr()),
            reinterpret_cast<float *>(princptim.data_ptr()),
            pixelcoordsim ? reinterpret_cast<float *>(pixelcoordsim->data_ptr()) : nullptr,
            volradius,
            reinterpret_cast<float *>(rayposim.data_ptr()),
            reinterpret_cast<float *>(raydirim.data_ptr()),
            reinterpret_cast<float *>(tminmaxim.data_ptr()),
            0);

    return {};
}

std::vector<torch::Tensor> compute_raydirs_backward(
        torch::Tensor viewposim,
        torch::Tensor viewrotim,
        torch::Tensor focalim,
        torch::Tensor princptim,
        torch::optional<torch::Tensor> pixelcoordsim,
        int W, int H,
        float volradius,
        torch::Tensor rayposim,
        torch::Tensor raydirim,
        torch::Tensor tminmaxim,
        torch::Tensor grad_viewpos,
        torch::Tensor grad_viewrot,
        torch::Tensor grad_focal,
        torch::Tensor grad_princpt) {
    CHECK_INPUT(viewposim);
    CHECK_INPUT(viewrotim);
    CHECK_INPUT(focalim);
    CHECK_INPUT(princptim);
    if (pixelcoordsim) { CHECK_INPUT(*pixelcoordsim); }
    CHECK_INPUT(rayposim);
    CHECK_INPUT(raydirim);
    CHECK_INPUT(tminmaxim);
    CHECK_INPUT(grad_viewpos);
    CHECK_INPUT(grad_viewrot);
    CHECK_INPUT(grad_focal);
    CHECK_INPUT(grad_princpt);

    int N = viewposim.size(0);
    assert(!pixelcoordsim || (pixelcoordsim.size(1) == H && pixelcoordsim.size(2) == W));

    compute_raydirs_backward_cuda(N, H, W,
            reinterpret_cast<float *>(viewposim.data_ptr()),
            reinterpret_cast<float *>(viewrotim.data_ptr()),
            reinterpret_cast<float *>(focalim.data_ptr()),
            reinterpret_cast<float *>(princptim.data_ptr()),
            pixelcoordsim ? reinterpret_cast<float *>(pixelcoordsim->data_ptr()) : nullptr,
            volradius,
            reinterpret_cast<float *>(rayposim.data_ptr()),
            reinterpret_cast<float *>(raydirim.data_ptr()),
            reinterpret_cast<float *>(tminmaxim.data_ptr()),
            reinterpret_cast<float *>(grad_viewpos.data_ptr()),
            reinterpret_cast<float *>(grad_viewrot.data_ptr()),
            reinterpret_cast<float *>(grad_focal.data_ptr()),
            reinterpret_cast<float *>(grad_princpt.data_ptr()),
            0);

    return {};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_raydirs_forward",  &compute_raydirs_forward,  "raydirs forward (CUDA)");
    m.def("compute_raydirs_backward", &compute_raydirs_backward, "raydirs backward (CUDA)");
}
