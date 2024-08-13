// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <chrono>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

#include "helper_math.h"

#include "cudadispatch.h"

#include "utils.h"

#include "primtransf.h"
#include "primsampler.h"
#include "primaccum.h"

#include "mvpraymarch_subset_kernel.h"

typedef std::shared_ptr<PrimTransfDataBase> PrimTransfDataBase_ptr;
typedef std::shared_ptr<PrimSamplerDataBase> PrimSamplerDataBase_ptr;
typedef std::shared_ptr<PrimAccumDataBase> PrimAccumDataBase_ptr;
typedef std::function<void(dim3, dim3, cudaStream_t, int, int, int, int,
        float3*, float3*, float, float2*, int*, int2*, float3*,
        PrimTransfDataBase_ptr, PrimSamplerDataBase_ptr,
        PrimAccumDataBase_ptr)> mapfn_t;
typedef RaySubsetFixedBVH<false, 512, true, PrimTransfSRT> raysubset_t;

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

        int algorithm,
        bool sortboxes,
        int maxhitboxes,
        bool synchitboxes,
        bool chlast,
        float fadescale, 
        float fadeexp,
        int accum,
        float termthresh,
        int griddim, int blocksizex, int blocksizey,
        cudaStream_t stream) {
    dim3 blocksize(blocksizex, blocksizey);
    dim3 gridsize;
    gridsize = dim3(
            (W + blocksize.x - 1) / blocksize.x,
            (H + blocksize.y - 1) / blocksize.y,
            N);

    std::shared_ptr<PrimTransfDataBase> primtransf_data;
    primtransf_data = std::make_shared<PrimTransfSRT::Data>(PrimTransfSRT::Data{
            PrimTransfDataBase{},
            K, (float3*)primpos, nullptr,
            K * 3, (float3*)primrot, nullptr,
            K, (float3*)primscale, nullptr});
    std::shared_ptr<PrimSamplerDataBase> primsampler_data;
    if (algorithm == 1) {
        primsampler_data = std::make_shared<PrimSamplerTW<true>::Data>(PrimSamplerTW<true>::Data{
            PrimSamplerDataBase{},
            fadescale, fadeexp,
            K * TD * TH * TW * 4, TD, TH, TW, tplate, nullptr,
            K * WD * WH * WW * 3, WD, WH, WW, warp, nullptr});
    } else {
        primsampler_data = std::make_shared<PrimSamplerTW<false>::Data>(PrimSamplerTW<false>::Data{
            PrimSamplerDataBase{},
            fadescale, fadeexp,
            K * TD * TH * TW * 4, TD, TH, TW, tplate, nullptr,
            0, 0, 0, 0, nullptr, nullptr});
    }
    std::shared_ptr<PrimAccumDataBase> primaccum_data = std::make_shared<PrimAccumAdditive::Data>(PrimAccumAdditive::Data{
            PrimAccumDataBase{},
            termthresh, H * W, W, 1, (float4*)rayrgbaim, nullptr, (float3*)raysatim});

    std::map<int, mapfn_t> dispatcher = {
        {0, make_cudacall(raymarch_subset_forward_kernel<512, 4, raysubset_t, PrimTransfSRT, PrimSamplerTW<false>, PrimAccumAdditive>)},
        {1, make_cudacall(raymarch_subset_forward_kernel<512, 4, raysubset_t, PrimTransfSRT, PrimSamplerTW<true>, PrimAccumAdditive>)}};

    auto iter = dispatcher.find(algorithm);
    if (iter != dispatcher.end()) {
        (iter->second)(
            gridsize, blocksize, stream,
            N, H, W, K,
            reinterpret_cast<float3 *>(rayposim),
            reinterpret_cast<float3 *>(raydirim),
            stepsize,
            reinterpret_cast<float2 *>(tminmaxim),
            reinterpret_cast<int    *>(sortedobjid),
            reinterpret_cast<int2   *>(nodechildren),
            reinterpret_cast<float3 *>(nodeaabb),
            primtransf_data,
            primsampler_data,
            primaccum_data);
    }
}

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

        cudaStream_t stream) {
    dim3 blocksize(blocksizex, blocksizey);
    dim3 gridsize;
    gridsize = dim3(
            (W + blocksize.x - 1) / blocksize.x,
            (H + blocksize.y - 1) / blocksize.y,
            N);

    std::shared_ptr<PrimTransfDataBase> primtransf_data;
    primtransf_data = std::make_shared<PrimTransfSRT::Data>(PrimTransfSRT::Data{
        PrimTransfDataBase{},
        K, (float3*)primpos, (float3*)grad_primpos,
        K * 3, (float3*)primrot, (float3*)grad_primrot,
        K, (float3*)primscale, (float3*)grad_primscale});
    std::shared_ptr<PrimSamplerDataBase> primsampler_data;
    if (algorithm == 1) {
        primsampler_data = std::make_shared<PrimSamplerTW<true>::Data>(PrimSamplerTW<true>::Data{
            PrimSamplerDataBase{},
            fadescale, fadeexp,
            K * TD * TH * TW * 4, TD, TH, TW, tplate, grad_tplate,
            K * WD * WH * WW * 3, WD, WH, WW, warp, grad_warp});
    } else {
        primsampler_data = std::make_shared<PrimSamplerTW<false>::Data>(PrimSamplerTW<false>::Data{
            PrimSamplerDataBase{},
            fadescale, fadeexp,
            K * TD * TH * TW * 4, TD, TH, TW, tplate, grad_tplate,
            0, 0, 0, 0, nullptr, nullptr});
    }
    std::shared_ptr<PrimAccumDataBase> primaccum_data = std::make_shared<PrimAccumAdditive::Data>(PrimAccumAdditive::Data{
            PrimAccumDataBase{},
            termthresh, H * W, W, 1, (float4*)rayrgbaim, (float4*)grad_rayrgba, (float3*)raysatim});

    std::map<int, mapfn_t> dispatcher = {
        {0, make_cudacall(raymarch_subset_backward_kernel<true, 512, 4, raysubset_t, PrimTransfSRT, PrimSamplerTW<false>, PrimAccumAdditive>)},
        {1, make_cudacall(raymarch_subset_backward_kernel<true, 512, 4, raysubset_t, PrimTransfSRT, PrimSamplerTW<true>, PrimAccumAdditive>)}};

    auto iter = dispatcher.find(algorithm);
    if (iter != dispatcher.end()) {
        (iter->second)(
            gridsize, blocksize, stream,
            N, H, W, K,
            reinterpret_cast<float3 *>(rayposim),
            reinterpret_cast<float3 *>(raydirim),
            stepsize,
            reinterpret_cast<float2 *>(tminmaxim),
            reinterpret_cast<int    *>(sortedobjid),
            reinterpret_cast<int2   *>(nodechildren),
            reinterpret_cast<float3 *>(nodeaabb),
            primtransf_data,
            primsampler_data,
            primaccum_data);
    }
}
