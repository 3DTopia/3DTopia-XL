// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <functional>
#include <map>
#include <tuple>
#include <vector>

#include "helper_math.h"

__global__ void compute_raydirs_forward_kernel(
        int N, int H, int W,
        float3 * viewposim,
        float3 * viewrotim,
        float2 * focalim,
        float2 * princptim,
        float2 * pixelcoordsim,
        float volradius,
        float3 * rayposim,
        float3 * raydirim,
        float2 * tminmaxim
        ) {
    bool validthread = false;
    int w, h, n;
    w = blockIdx.x * blockDim.x + threadIdx.x;
    h = (blockIdx.y * blockDim.y + threadIdx.y)%H;
    n = (blockIdx.y * blockDim.y + threadIdx.y)/H;
    validthread = (w < W) && (h < H) && (n<N);

    if (validthread) {
        float3 raypos = viewposim[n] / volradius;
        float3 viewrot0 = viewrotim[n * 3 + 0];
        float3 viewrot1 = viewrotim[n * 3 + 1];
        float3 viewrot2 = viewrotim[n * 3 + 2];
        float2 pixelcoord = pixelcoordsim ? pixelcoordsim[n * H * W + h * W + w] : make_float2(w, h);
        pixelcoord = (pixelcoord - princptim[n]) / focalim[n];
        float3 raydir = make_float3(pixelcoord, 1.f);
        raydir = viewrot0 * raydir.x + viewrot1 * raydir.y + viewrot2 * raydir.z;
        raydir = normalize(raydir);

        float3 t1 = (-1.f - raypos) / raydir;
        float3 t2 = ( 1.f - raypos) / raydir;
        float tmin = fmaxf(fminf(t1.x, t2.x), fmaxf(fminf(t1.y, t2.y), fminf(t1.z, t2.z)));
        float tmax = fminf(fmaxf(t1.x, t2.x), fminf(fmaxf(t1.y, t2.y), fmaxf(t1.z, t2.z)));
        float2 tminmax = make_float2(fmaxf(tmin, 0.f), tmax);

        rayposim[n * H * W + h * W + w] = raypos;
        raydirim[n * H * W + h * W + w] = raydir;
        tminmaxim[n * H * W + h * W + w] = tminmax;
    }
}

__global__ void compute_raydirs_backward_kernel(
        int N, int H, int W,
        float3 * viewposim,
        float3 * viewrotim,
        float2 * focalim,
        float2 * princptim,
        float2 * pixelcoordsim,
        float volradius,
        float3 * rayposim,
        float3 * raydirim,
        float2 * tminmaxim,
        float3 * grad_viewposim,
        float3 * grad_viewrotim,
        float2 * grad_focalim,
        float2 * grad_princptim
        ) {
    bool validthread = false;
    int w, h, n;
    w = blockIdx.x * blockDim.x + threadIdx.x;
    h = (blockIdx.y * blockDim.y + threadIdx.y)%H;
    n = (blockIdx.y * blockDim.y + threadIdx.y)/H;
    validthread = (w < W) && (h < H) && (n<N);

    if (validthread) {
        float3 raypos = viewposim[n] / volradius;
        float3 viewrot0 = viewrotim[n * 3 + 0];
        float3 viewrot1 = viewrotim[n * 3 + 1];
        float3 viewrot2 = viewrotim[n * 3 + 2];
        float2 pixelcoord = pixelcoordsim ? pixelcoordsim[n * H * W + h * W + w] : make_float2(w, h);
        pixelcoord = (pixelcoord - princptim[n]) / focalim[n];
        float3 raydir = make_float3(pixelcoord, 1.f);
        raydir = viewrot0 * raydir.x + viewrot1 * raydir.y + viewrot2 * raydir.z;
        raydir = normalize(raydir);

        float3 t1 = (-1.f - raypos) / raydir;
        float3 t2 = ( 1.f - raypos) / raydir;
        float tmin = fmaxf(fminf(t1.x, t2.x), fmaxf(fminf(t1.y, t2.y), fminf(t1.z, t2.z)));
        float tmax = fminf(fmaxf(t1.x, t2.x), fminf(fmaxf(t1.y, t2.y), fmaxf(t1.z, t2.z)));
        float2 tminmax = make_float2(fmaxf(tmin, 0.f), tmax);

    }
}

void compute_raydirs_forward_cuda(
        int N, int H, int W,
        float * viewposim,
        float * viewrotim,
        float * focalim,
        float * princptim,
        float * pixelcoordsim,
        float volradius,
        float * rayposim,
        float * raydirim,
        float * tminmaxim,
        cudaStream_t stream) {
    int blocksizex = 16;
    int blocksizey = 16;
    dim3 blocksize(blocksizex, blocksizey);
    dim3 gridsize;
    gridsize = dim3(
            (W + blocksize.x - 1) / blocksize.x,
            (N*H + blocksize.y - 1) / blocksize.y);

    auto fn = compute_raydirs_forward_kernel;
    fn<<<gridsize, blocksize, 0, stream>>>(
        N, H, W,
        reinterpret_cast<float3 *>(viewposim),
        reinterpret_cast<float3 *>(viewrotim),
        reinterpret_cast<float2 *>(focalim),
        reinterpret_cast<float2 *>(princptim),
        reinterpret_cast<float2 *>(pixelcoordsim),
        volradius,
        reinterpret_cast<float3 *>(rayposim),
        reinterpret_cast<float3 *>(raydirim),
        reinterpret_cast<float2 *>(tminmaxim));
}

void compute_raydirs_backward_cuda(
        int N, int H, int W,
        float * viewposim,
        float * viewrotim,
        float * focalim,
        float * princptim,
        float * pixelcoordsim,
        float volradius,
        float * rayposim,
        float * raydirim,
        float * tminmaxim,
        float * grad_viewposim,
        float * grad_viewrotim,
        float * grad_focalim,
        float * grad_princptim,
        cudaStream_t stream) {
    int blocksizex = 16;
    int blocksizey = 16;
    dim3 blocksize(blocksizex, blocksizey);
    dim3 gridsize;
    gridsize = dim3(
            (W + blocksize.x - 1) / blocksize.x,
            (N*H + blocksize.y - 1) / blocksize.y);

    auto fn = compute_raydirs_backward_kernel;
    fn<<<gridsize, blocksize, 0, stream>>>(
            N, H, W,
            reinterpret_cast<float3 *>(viewposim),
            reinterpret_cast<float3 *>(viewrotim),
            reinterpret_cast<float2 *>(focalim),
            reinterpret_cast<float2 *>(princptim),
            reinterpret_cast<float2 *>(pixelcoordsim),
            volradius,
            reinterpret_cast<float3 *>(rayposim),
            reinterpret_cast<float3 *>(raydirim),
            reinterpret_cast<float2 *>(tminmaxim),
            reinterpret_cast<float3 *>(grad_viewposim),
            reinterpret_cast<float3 *>(grad_viewrotim),
            reinterpret_cast<float2 *>(grad_focalim),
            reinterpret_cast<float2 *>(grad_princptim));
}
