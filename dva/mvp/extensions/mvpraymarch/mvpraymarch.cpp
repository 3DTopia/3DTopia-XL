// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <vector>

void compute_morton_cuda(
        int N, int K,
        float * primpos,
        int * code,
        int algorithm,
        cudaStream_t stream);

void build_tree_cuda(
        int N, int K,
        int * sortedcode,
        int * nodechildren,
        int * nodeparent,
        cudaStream_t stream);

void compute_aabb_cuda(
        int N, int K,
        float * primpos,
        float * primrot,
        float * primscale,
        int * sortedobjid,
        int * nodechildren,
        int * nodeparent,
        float * nodeaabb,
        int algorithm,
        cudaStream_t stream);

void raymarch_forward_cuda(
        int N, int H, int W, int K,
        float * rayposim,
        float * raydirim,
        float stepsize,
        float * tminmaxim,

        int * sortedobjid,
        int * nodechildren,
        float * nodeaabb,

        float * primpos,
        float * primrot,
        float * primscale,

        int TD, int TH, int TW,
        float * tplate,
        int WD, int WH, int WW,
        float * warp,

        float * rayrgbaim,
        float * raysatim,
        int * raytermim,

        int algorithm, bool sortboxes, int maxhitboxes, bool synchitboxes,
        bool chlast, float fadescale, float fadeexp, int accum, float termthresh,
        int griddim, int blocksizex, int blocksizey,
        cudaStream_t stream);

void raymarch_backward_cuda(
        int N, int H, int W, int K,
        float * rayposim,
        float * raydirim,
        float stepsize,
        float * tminmaxim,

        int * sortedobjid,
        int * nodechildren,
        float * nodeaabb,

        float * primpos,
        float * grad_primpos,
        float * primrot,
        float * grad_primrot,
        float * primscale,
        float * grad_primscale,

        int TD, int TH, int TW,
        float * tplate,
        float * grad_tplate,
        int WD, int WH, int WW,
        float * warp,
        float * grad_warp,

        float * rayrgbaim,
        float * grad_rayrgba,
        float * raysatim,
        int * raytermim,

        int algorithm, bool sortboxes, int maxhitboxes, bool synchitboxes,
        bool chlast, float fadescale, float fadeexp, int accum, float termthresh,
        int griddim, int blocksizex, int blocksizey,
        cudaStream_t stream);

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA((x)); CHECK_CONTIGUOUS((x))

std::vector<torch::Tensor> compute_morton(
        torch::Tensor primpos,
        torch::Tensor code,
        int algorithm) {
    CHECK_INPUT(primpos);
    CHECK_INPUT(code);

    int N = primpos.size(0);
    int K = primpos.size(1);

    compute_morton_cuda(
            N, K,
            reinterpret_cast<float *>(primpos.data_ptr()),
            reinterpret_cast<int *>(code.data_ptr()),
            algorithm,
            0);

    return {};
}

std::vector<torch::Tensor> build_tree(
        torch::Tensor sortedcode,
        torch::Tensor nodechildren,
        torch::Tensor nodeparent) {
    CHECK_INPUT(sortedcode);
    CHECK_INPUT(nodechildren);
    CHECK_INPUT(nodeparent);

    int N = sortedcode.size(0);
    int K = sortedcode.size(1);

    build_tree_cuda(N, K,
            reinterpret_cast<int *>(sortedcode.data_ptr()),
            reinterpret_cast<int *>(nodechildren.data_ptr()),
            reinterpret_cast<int *>(nodeparent.data_ptr()),
            0);

    return {};
}

std::vector<torch::Tensor> compute_aabb(
        torch::Tensor primpos,
        torch::optional<torch::Tensor> primrot,
        torch::optional<torch::Tensor> primscale,
        torch::Tensor sortedobjid,
        torch::Tensor nodechildren,
        torch::Tensor nodeparent,
        torch::Tensor nodeaabb,
        int algorithm) {
    CHECK_INPUT(sortedobjid);
    CHECK_INPUT(primpos);
    if (primrot) { CHECK_INPUT(*primrot); }
    if (primscale) { CHECK_INPUT(*primscale); }
    CHECK_INPUT(nodechildren);
    CHECK_INPUT(nodeparent);
    CHECK_INPUT(nodeaabb);

    int N = primpos.size(0);
    int K = primpos.size(1);

    compute_aabb_cuda(N, K,
            reinterpret_cast<float *>(primpos.data_ptr()),
            primrot ? reinterpret_cast<float *>(primrot->data_ptr()) : nullptr,
            primscale ? reinterpret_cast<float *>(primscale->data_ptr()) : nullptr,
            reinterpret_cast<int *>(sortedobjid.data_ptr()),
            reinterpret_cast<int *>(nodechildren.data_ptr()),
            reinterpret_cast<int *>(nodeparent.data_ptr()),
            reinterpret_cast<float *>(nodeaabb.data_ptr()),
            algorithm,
            0);

    return {};
}

std::vector<torch::Tensor> raymarch_forward(
        torch::Tensor rayposim,
        torch::Tensor raydirim,
        float stepsize,
        torch::Tensor tminmaxim,

        torch::optional<torch::Tensor> sortedobjid,
        torch::optional<torch::Tensor> nodechildren,
        torch::optional<torch::Tensor> nodeaabb,

        torch::Tensor primpos,
        torch::optional<torch::Tensor> primrot,
        torch::optional<torch::Tensor> primscale,

        torch::Tensor tplate,
        torch::optional<torch::Tensor> warp,

        torch::Tensor rayrgbaim,
        torch::optional<torch::Tensor> raysatim,
        torch::optional<torch::Tensor> raytermim,

        int algorithm=0,
        bool sortboxes=true,
        int maxhitboxes=512,
        bool synchitboxes=false,
        bool chlast=false,
        float fadescale=8.f,
        float fadeexp=8.f,
        int accum=0,
        float termthresh=0.f,
        int griddim=3,
        int blocksizex=8,
        int blocksizey=16) {
    CHECK_INPUT(rayposim);
    CHECK_INPUT(raydirim);
    CHECK_INPUT(tminmaxim);
    if (sortedobjid) { CHECK_INPUT(*sortedobjid); }
    if (nodechildren) { CHECK_INPUT(*nodechildren); }
    if (nodeaabb) { CHECK_INPUT(*nodeaabb); }
    CHECK_INPUT(tplate);
    if (warp) { CHECK_INPUT(*warp); }
    CHECK_INPUT(primpos);
    if (primrot) { CHECK_INPUT(*primrot); }
    if (primscale) { CHECK_INPUT(*primscale); }
    CHECK_INPUT(rayrgbaim);
    if (raysatim) { CHECK_INPUT(*raysatim); }
    if (raytermim) { CHECK_INPUT(*raytermim); }

    int N = rayposim.size(0);
    int H = rayposim.size(1);
    int W = rayposim.size(2);
    int K = primpos.size(1);

    int TD, TH, TW;
    if (chlast) {
        TD = tplate.size(2); TH = tplate.size(3); TW = tplate.size(4);
    } else {
        TD = tplate.size(3); TH = tplate.size(4); TW = tplate.size(5);
    }

    int WD = 0, WH = 0, WW = 0;
    if (warp) {
        if (chlast) {
            WD = warp->size(2); WH = warp->size(3); WW = warp->size(4);
        } else {
            WD = warp->size(3); WH = warp->size(4); WW = warp->size(5);
        }
    }

    raymarch_forward_cuda(N, H, W, K,
            reinterpret_cast<float *>(rayposim.data_ptr()),
            reinterpret_cast<float *>(raydirim.data_ptr()),
            stepsize,
            reinterpret_cast<float *>(tminmaxim.data_ptr()),
            sortedobjid ? reinterpret_cast<int *>(sortedobjid->data_ptr()) : nullptr,
            nodechildren ? reinterpret_cast<int *>(nodechildren->data_ptr()) : nullptr,
            nodeaabb ? reinterpret_cast<float *>(nodeaabb->data_ptr()) : nullptr,

            // prim transforms
            reinterpret_cast<float *>(primpos.data_ptr()),
            primrot ? reinterpret_cast<float *>(primrot->data_ptr()) : nullptr,
            primscale ? reinterpret_cast<float *>(primscale->data_ptr()) : nullptr,

            // prim sampler
            TD, TH, TW,
            reinterpret_cast<float *>(tplate.data_ptr()),
            WD, WH, WW,
            warp ? reinterpret_cast<float *>(warp->data_ptr()) : nullptr,

            // prim accumulator
            reinterpret_cast<float *>(rayrgbaim.data_ptr()),
            raysatim ? reinterpret_cast<float *>(raysatim->data_ptr()) : nullptr,
            raytermim ? reinterpret_cast<int *>(raytermim->data_ptr()) : nullptr,

            // options
            algorithm, sortboxes, maxhitboxes, synchitboxes, chlast, fadescale, fadeexp, accum, termthresh,
            griddim, blocksizex, blocksizey,
            0);

    return {};
}

std::vector<torch::Tensor> raymarch_backward(
        torch::Tensor rayposim,
        torch::Tensor raydirim,
        float stepsize,
        torch::Tensor tminmaxim,

        torch::optional<torch::Tensor> sortedobjid,
        torch::optional<torch::Tensor> nodechildren,
        torch::optional<torch::Tensor> nodeaabb,

        torch::Tensor primpos,
        torch::Tensor grad_primpos,
        torch::optional<torch::Tensor> primrot,
        torch::optional<torch::Tensor> grad_primrot,
        torch::optional<torch::Tensor> primscale,
        torch::optional<torch::Tensor> grad_primscale,

        torch::Tensor tplate,
        torch::Tensor grad_tplate,
        torch::optional<torch::Tensor> warp,
        torch::optional<torch::Tensor> grad_warp,

        torch::Tensor rayrgbaim,
        torch::Tensor grad_rayrgba,
        torch::optional<torch::Tensor> raysatim,
        torch::optional<torch::Tensor> raytermim,

        int algorithm=0,
        bool sortboxes=true,
        int maxhitboxes=512,
        bool synchitboxes=false,
        bool chlast=false,
        float fadescale=8.f,
        float fadeexp=8.f,
        int accum=0,
        float termthresh=0.f,
        int griddim=3,
        int blocksizex=8,
        int blocksizey=16) {
    CHECK_INPUT(rayposim);
    CHECK_INPUT(raydirim);
    CHECK_INPUT(tminmaxim);
    if (sortedobjid) { CHECK_INPUT(*sortedobjid); }
    if (nodechildren) { CHECK_INPUT(*nodechildren); }
    if (nodeaabb) { CHECK_INPUT(*nodeaabb); }
    CHECK_INPUT(tplate);
    if (warp) { CHECK_INPUT(*warp); }
    CHECK_INPUT(primpos);
    if (primrot) { CHECK_INPUT(*primrot); }
    if (primscale) { CHECK_INPUT(*primscale); }
    CHECK_INPUT(rayrgbaim);
    if (raysatim) { CHECK_INPUT(*raysatim); }
    if (raytermim) { CHECK_INPUT(*raytermim); }
    CHECK_INPUT(grad_rayrgba);
    CHECK_INPUT(grad_tplate);
    if (grad_warp) { CHECK_INPUT(*grad_warp); }
    CHECK_INPUT(grad_primpos);
    if (grad_primrot) { CHECK_INPUT(*grad_primrot); }
    if (grad_primscale) { CHECK_INPUT(*grad_primscale); }

    int N = rayposim.size(0);
    int H = rayposim.size(1);
    int W = rayposim.size(2);
    int K = primpos.size(1);

    int TD, TH, TW;
    if (chlast) {
        TD = tplate.size(2); TH = tplate.size(3); TW = tplate.size(4);
    } else {
        TD = tplate.size(3); TH = tplate.size(4); TW = tplate.size(5);
    }

    int WD = 0, WH = 0, WW = 0;
    if (warp) {
        if (chlast) {
            WD = warp->size(2); WH = warp->size(3); WW = warp->size(4);
        } else {
            WD = warp->size(3); WH = warp->size(4); WW = warp->size(5);
        }
    }

    raymarch_backward_cuda(N, H, W, K,
            reinterpret_cast<float *>(rayposim.data_ptr()),
            reinterpret_cast<float *>(raydirim.data_ptr()),
            stepsize,
            reinterpret_cast<float *>(tminmaxim.data_ptr()),
            sortedobjid ? reinterpret_cast<int *>(sortedobjid->data_ptr()) : nullptr,
            nodechildren ? reinterpret_cast<int *>(nodechildren->data_ptr()) : nullptr,
            nodeaabb ? reinterpret_cast<float *>(nodeaabb->data_ptr()) : nullptr,

            reinterpret_cast<float *>(primpos.data_ptr()),
            reinterpret_cast<float *>(grad_primpos.data_ptr()),
            primrot ? reinterpret_cast<float *>(primrot->data_ptr()) : nullptr,
            grad_primrot ? reinterpret_cast<float *>(grad_primrot->data_ptr()) : nullptr,
            primscale ? reinterpret_cast<float *>(primscale->data_ptr()) : nullptr,
            grad_primscale ? reinterpret_cast<float *>(grad_primscale->data_ptr()) : nullptr,

            TD, TH, TW,
            reinterpret_cast<float *>(tplate.data_ptr()),
            reinterpret_cast<float *>(grad_tplate.data_ptr()),
            WD, WH, WW,
            warp ? reinterpret_cast<float *>(warp->data_ptr()) : nullptr,
            grad_warp ? reinterpret_cast<float *>(grad_warp->data_ptr()) : nullptr,

            reinterpret_cast<float *>(rayrgbaim.data_ptr()),
            reinterpret_cast<float *>(grad_rayrgba.data_ptr()),
            raysatim ? reinterpret_cast<float *>(raysatim->data_ptr()) : nullptr,
            raytermim ? reinterpret_cast<int *>(raytermim->data_ptr()) : nullptr,

            algorithm, sortboxes, maxhitboxes, synchitboxes, chlast, fadescale, fadeexp, accum, termthresh,
            griddim, blocksizex, blocksizey,
            0);

    return {};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_morton", &compute_morton, "compute morton codes (CUDA)");
    m.def("build_tree", &build_tree, "build BVH tree (CUDA)");
    m.def("compute_aabb", &compute_aabb, "compute AABB sizes (CUDA)");

    m.def("raymarch_forward",  &raymarch_forward,  "raymarch forward (CUDA)");
    m.def("raymarch_backward", &raymarch_backward, "raymarch backward (CUDA)");
}
