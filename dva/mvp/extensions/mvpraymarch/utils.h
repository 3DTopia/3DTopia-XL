// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef MVPRAYMARCHER_UTILS_H_
#define MVPRAYMARCHER_UTILS_H_

#include <cassert>
#include <cmath>

#include <limits>

#include "helper_math.h"

static __forceinline__ __device__ float clock_diff(long long int end, long long int start) {
    long long int max_clock = std::numeric_limits<long long int>::max();
    return (end<start? (end + float(max_clock-start)) : float(end-start));
}

static __forceinline__ __device__
bool allgt(float3 a, float3 b) {
    return a.x >= b.x && a.y >= b.y && a.z >= b.z;
}

static __forceinline__ __device__
bool alllt(float3 a, float3 b) {
    return a.x <= b.x && a.y <= b.y && a.z <= b.z;
}

static __forceinline__ __device__
float4 softplus(float4 x) {
    return make_float4(
            x.x > 20.f ? x.x : logf(1.f + expf(x.x)),
            x.y > 20.f ? x.y : logf(1.f + expf(x.y)),
            x.z > 20.f ? x.z : logf(1.f + expf(x.z)),
            x.w > 20.f ? x.w : logf(1.f + expf(x.w)));
}

static __forceinline__ __device__
float softplus(float x) {
    // that's a neat trick
    return __logf(1.f + __expf(-abs(x))) + max(x, 0.f);
}
static __forceinline__ __device__
float softplus_grad(float x) {
    // that's a neat trick
    float expnabsx = __expf(-abs(x));
    return (0.5f - expnabsx / (1.f + expnabsx)) * copysign(1.f, x) + 0.5f;
}


static __forceinline__ __device__
float4 sigmoid(float4 x) {
    return make_float4(
            1.f / (1.f + expf(-x.x)),
            1.f / (1.f + expf(-x.y)),
            1.f / (1.f + expf(-x.z)),
            1.f / (1.f + expf(-x.w)));
}

// perform reduction on warp, then call atomicAdd for only one lane
static __forceinline__ __device__ void fastAtomicAdd(float * ptr, float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    const int laneid = (threadIdx.y * blockDim.x + threadIdx.x) % 32;
    if (laneid == 0) {
        atomicAdd(ptr, val);
    }
}


static __forceinline__ __device__
bool within_bounds_3d(int d, int h, int w, int D, int H, int W) {
    return d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W;
}

static __forceinline__ __device__
void safe_add_3d(float *data, int d, int h, int w,
               int sD, int sH, int sW, int D, int H, int W,
               float delta) {
    if (within_bounds_3d(d, h, w, D, H, W)) {
        atomicAdd(data + d * sD + h * sH + w * sW, delta);
    }
}

static __forceinline__ __device__
void safe_add_3d(float3 *data, int d, int h, int w,
               int sD, int sH, int sW, int D, int H, int W,
               float3 delta) {
    if (within_bounds_3d(d, h, w, D, H, W)) {
        atomicAdd((float*)data + (d * sD + h * sH + w * sW) * 3 + 0, delta.x);
        atomicAdd((float*)data + (d * sD + h * sH + w * sW) * 3 + 1, delta.y);
        atomicAdd((float*)data + (d * sD + h * sH + w * sW) * 3 + 2, delta.z);
    }
}

static __forceinline__ __device__
void safe_add_3d(float4 *data, int d, int h, int w,
               int sD, int sH, int sW, int D, int H, int W,
               float4 delta) {
    if (within_bounds_3d(d, h, w, D, H, W)) {
        atomicAdd((float*)data + (d * sD + h * sH + w * sW) * 4 + 0, delta.x);
        atomicAdd((float*)data + (d * sD + h * sH + w * sW) * 4 + 1, delta.y);
        atomicAdd((float*)data + (d * sD + h * sH + w * sW) * 4 + 2, delta.z);
        atomicAdd((float*)data + (d * sD + h * sH + w * sW) * 4 + 3, delta.w);
    }
}

static __forceinline__ __device__
float clip_coordinates(float in, int clip_limit) {
    return ::min(static_cast<float>(clip_limit - 1), ::max(in, 0.f));
}

template <typename scalar_t>
static __forceinline__ __device__
float clip_coordinates_set_grad(float in, int clip_limit, scalar_t *grad_in) {
    if (in < 0.f) {
        *grad_in = static_cast<scalar_t>(0);
        return 0.f;
    } else {
        float max = static_cast<float>(clip_limit - 1);
        if (in > max) {
            *grad_in = static_cast<scalar_t>(0);
            return max;
        } else {
            *grad_in = static_cast<scalar_t>(1);
            return in;
        }
    }
}

template<typename out_t>
static __device__ out_t grid_sample_forward(int C, int inp_D, int inp_H,
        int inp_W, float* vals, float3 pos, bool border) {
    int inp_sW = 1, inp_sH = inp_W, inp_sD = inp_W * inp_H, inp_sC = inp_W * inp_H * inp_D;
    int out_sC = 1;

    // normalize ix, iy, iz from [-1, 1] to [0, inp_W-1] & [0, inp_H-1] & [0, inp_D-1]
    float ix = max(-10.f, min(10.f, ((pos.x + 1.f) * 0.5f))) * (inp_W - 1);
    float iy = max(-10.f, min(10.f, ((pos.y + 1.f) * 0.5f))) * (inp_H - 1);
    float iz = max(-10.f, min(10.f, ((pos.z + 1.f) * 0.5f))) * (inp_D - 1);

    if (border) {
        // clip coordinates to image borders
        ix = clip_coordinates(ix, inp_W);
        iy = clip_coordinates(iy, inp_H);
        iz = clip_coordinates(iz, inp_D);
    }

    // get corner pixel values from (x, y, z)
    // for 4d, we used north-east-south-west
    // for 5d, we add top-bottom
    int ix_tnw = static_cast<int>(::floor(ix));
    int iy_tnw = static_cast<int>(::floor(iy));
    int iz_tnw = static_cast<int>(::floor(iz));

    int ix_tne = ix_tnw + 1;
    int iy_tne = iy_tnw;
    int iz_tne = iz_tnw;

    int ix_tsw = ix_tnw;
    int iy_tsw = iy_tnw + 1;
    int iz_tsw = iz_tnw;

    int ix_tse = ix_tnw + 1;
    int iy_tse = iy_tnw + 1;
    int iz_tse = iz_tnw;

    int ix_bnw = ix_tnw;
    int iy_bnw = iy_tnw;
    int iz_bnw = iz_tnw + 1;

    int ix_bne = ix_tnw + 1;
    int iy_bne = iy_tnw;
    int iz_bne = iz_tnw + 1;

    int ix_bsw = ix_tnw;
    int iy_bsw = iy_tnw + 1;
    int iz_bsw = iz_tnw + 1;

    int ix_bse = ix_tnw + 1;
    int iy_bse = iy_tnw + 1;
    int iz_bse = iz_tnw + 1;

    // get surfaces to each neighbor:
    float tnw = (ix_bse - ix)    * (iy_bse - iy)    * (iz_bse - iz);
    float tne = (ix    - ix_bsw) * (iy_bsw - iy)    * (iz_bsw - iz);
    float tsw = (ix_bne - ix)    * (iy    - iy_bne) * (iz_bne - iz);
    float tse = (ix    - ix_bnw) * (iy    - iy_bnw) * (iz_bnw - iz);
    float bnw = (ix_tse - ix)    * (iy_tse - iy)    * (iz - iz_tse);
    float bne = (ix    - ix_tsw) * (iy_tsw - iy)    * (iz - iz_tsw);
    float bsw = (ix_tne - ix)    * (iy    - iy_tne) * (iz - iz_tne);
    float bse = (ix    - ix_tnw) * (iy    - iy_tnw) * (iz - iz_tnw);

    out_t result;
    //auto inp_ptr_NC = input.data + n * inp_sN;
    //auto out_ptr_NCDHW = output.data + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
    float * inp_ptr_NC = vals;
    float * out_ptr_NCDHW = &result.x;
    for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
      //   (c, iz_tnw, iy_tnw, ix_tnw) * tnw + (c, iz_tne, iy_tne, ix_tne) * tne
      // + (c, iz_tsw, iy_tsw, ix_tsw) * tsw + (c, iz_tse, iy_tse, ix_tse) * tse
      // + (c, iz_bnw, iy_bnw, ix_bnw) * bnw + (c, iz_bne, iy_bne, ix_bne) * bne
      // + (c, iz_bsw, iy_bsw, ix_bsw) * bsw + (c, iz_bse, iy_bse, ix_bse) * bse
      *out_ptr_NCDHW = static_cast<float>(0);
      if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
        *out_ptr_NCDHW += inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW] * tnw;
      }
      if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
        *out_ptr_NCDHW += inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW] * tne;
      }
      if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
        *out_ptr_NCDHW += inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW] * tsw;
      }
      if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
        *out_ptr_NCDHW += inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW] * tse;
      }
      if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
        *out_ptr_NCDHW += inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW] * bnw;
      }
      if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
        *out_ptr_NCDHW += inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW] * bne;
      }
      if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
        *out_ptr_NCDHW += inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW] * bsw;
      }
      if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
        *out_ptr_NCDHW += inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW] * bse;
      }
    }
    return result;
}

template<typename out_t>
static __device__ float3 grid_sample_backward(int C, int inp_D, int inp_H,
        int inp_W, float* vals, float* grad_vals, float3 pos, out_t grad_out,
        bool border) {
    int inp_sW = 1, inp_sH = inp_W, inp_sD = inp_W * inp_H, inp_sC = inp_W * inp_H * inp_D;
    int gInp_sW = 1, gInp_sH = inp_W, gInp_sD = inp_W * inp_H, gInp_sC = inp_W * inp_H * inp_D;
    int gOut_sC = 1;

    // normalize ix, iy, iz from [-1, 1] to [0, inp_W-1] & [0, inp_H-1] & [0, inp_D-1]
    float ix = max(-10.f, min(10.f, ((pos.x + 1.f) * 0.5f))) * (inp_W - 1);
    float iy = max(-10.f, min(10.f, ((pos.y + 1.f) * 0.5f))) * (inp_H - 1);
    float iz = max(-10.f, min(10.f, ((pos.z + 1.f) * 0.5f))) * (inp_D - 1);

    float gix_mult = (inp_W - 1.f) / 2;
    float giy_mult = (inp_H - 1.f) / 2;
    float giz_mult = (inp_D - 1.f) / 2;

    if (border) {
        // clip coordinates to image borders
        ix = clip_coordinates_set_grad(ix, inp_W, &gix_mult);
        iy = clip_coordinates_set_grad(iy, inp_H, &giy_mult);
        iz = clip_coordinates_set_grad(iz, inp_D, &giz_mult);
    }

    // get corner pixel values from (x, y, z)
    // for 4d, we used north-east-south-west
    // for 5d, we add top-bottom
    int ix_tnw = static_cast<int>(::floor(ix));
    int iy_tnw = static_cast<int>(::floor(iy));
    int iz_tnw = static_cast<int>(::floor(iz));

    int ix_tne = ix_tnw + 1;
    int iy_tne = iy_tnw;
    int iz_tne = iz_tnw;

    int ix_tsw = ix_tnw;
    int iy_tsw = iy_tnw + 1;
    int iz_tsw = iz_tnw;

    int ix_tse = ix_tnw + 1;
    int iy_tse = iy_tnw + 1;
    int iz_tse = iz_tnw;

    int ix_bnw = ix_tnw;
    int iy_bnw = iy_tnw;
    int iz_bnw = iz_tnw + 1;

    int ix_bne = ix_tnw + 1;
    int iy_bne = iy_tnw;
    int iz_bne = iz_tnw + 1;

    int ix_bsw = ix_tnw;
    int iy_bsw = iy_tnw + 1;
    int iz_bsw = iz_tnw + 1;

    int ix_bse = ix_tnw + 1;
    int iy_bse = iy_tnw + 1;
    int iz_bse = iz_tnw + 1;

    // get surfaces to each neighbor:
    float tnw = (ix_bse - ix)    * (iy_bse - iy)    * (iz_bse - iz);
    float tne = (ix    - ix_bsw) * (iy_bsw - iy)    * (iz_bsw - iz);
    float tsw = (ix_bne - ix)    * (iy    - iy_bne) * (iz_bne - iz);
    float tse = (ix    - ix_bnw) * (iy    - iy_bnw) * (iz_bnw - iz);
    float bnw = (ix_tse - ix)    * (iy_tse - iy)    * (iz - iz_tse);
    float bne = (ix    - ix_tsw) * (iy_tsw - iy)    * (iz - iz_tsw);
    float bsw = (ix_tne - ix)    * (iy    - iy_tne) * (iz - iz_tne);
    float bse = (ix    - ix_tnw) * (iy    - iy_tnw) * (iz - iz_tnw);

    float gix = static_cast<float>(0), giy = static_cast<float>(0), giz = static_cast<float>(0);
    //float *gOut_ptr_NCDHW = grad_output.data + n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
    //float *gInp_ptr_NC = grad_input.data + n * gInp_sN;
    //float *inp_ptr_NC = input.data + n * inp_sN;
    float *gOut_ptr_NCDHW = &grad_out.x;
    float *gInp_ptr_NC = grad_vals;
    float *inp_ptr_NC = vals;
    // calculate bilinear weighted pixel value and set output pixel
    for (int c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, gInp_ptr_NC += gInp_sC, inp_ptr_NC += inp_sC) {
      float gOut = *gOut_ptr_NCDHW;

      // calculate and set grad_input
      safe_add_3d(gInp_ptr_NC, iz_tnw, iy_tnw, ix_tnw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tnw * gOut);
      safe_add_3d(gInp_ptr_NC, iz_tne, iy_tne, ix_tne, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tne * gOut);
      safe_add_3d(gInp_ptr_NC, iz_tsw, iy_tsw, ix_tsw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tsw * gOut);
      safe_add_3d(gInp_ptr_NC, iz_tse, iy_tse, ix_tse, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tse * gOut);
      safe_add_3d(gInp_ptr_NC, iz_bnw, iy_bnw, ix_bnw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bnw * gOut);
      safe_add_3d(gInp_ptr_NC, iz_bne, iy_bne, ix_bne, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bne * gOut);
      safe_add_3d(gInp_ptr_NC, iz_bsw, iy_bsw, ix_bsw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bsw * gOut);
      safe_add_3d(gInp_ptr_NC, iz_bse, iy_bse, ix_bse, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bse * gOut);

      // calculate grad_grid
      if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
        float tnw_val = inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW];
        gix -= tnw_val * (iy_bse - iy)    * (iz_bse - iz)    * gOut;
        giy -= tnw_val * (ix_bse - ix)    * (iz_bse - iz)    * gOut;
        giz -= tnw_val * (ix_bse - ix)    * (iy_bse - iy)    * gOut;
      }
      if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
        float tne_val = inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW];
        gix += tne_val * (iy_bsw - iy)    * (iz_bsw - iz)    * gOut;
        giy -= tne_val * (ix    - ix_bsw) * (iz_bsw - iz)    * gOut;
        giz -= tne_val * (ix    - ix_bsw) * (iy_bsw - iy)    * gOut;
      }
      if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
        float tsw_val = inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW];
        gix -= tsw_val * (iy - iy_bne)    * (iz_bne - iz)    * gOut;
        giy += tsw_val * (ix_bne - ix)    * (iz_bne - iz)    * gOut;
        giz -= tsw_val * (ix_bne - ix)    * (iy    - iy_bne) * gOut;
      }
      if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
        float tse_val = inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW];
        gix += tse_val * (iy - iy_bnw)    * (iz_bnw - iz)    * gOut;
        giy += tse_val * (ix    - ix_bnw) * (iz_bnw - iz)    * gOut;
        giz -= tse_val * (ix    - ix_bnw) * (iy    - iy_bnw) * gOut;
      }
      if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
        float bnw_val = inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW];
        gix -= bnw_val * (iy_tse - iy)    * (iz - iz_tse)    * gOut;
        giy -= bnw_val * (ix_tse - ix)    * (iz - iz_tse)    * gOut;
        giz += bnw_val * (ix_tse - ix)    * (iy_tse - iy)    * gOut;
      }
      if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
        float bne_val = inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW];
        gix += bne_val * (iy_tsw - iy)    * (iz - iz_tsw)    * gOut;
        giy -= bne_val * (ix    - ix_tsw) * (iz - iz_tsw)    * gOut;
        giz += bne_val * (ix    - ix_tsw) * (iy_tsw - iy)    * gOut;
      }
      if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
        float bsw_val = inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW];
        gix -= bsw_val * (iy - iy_tne)    * (iz - iz_tne)    * gOut;
        giy += bsw_val * (ix_tne - ix)    * (iz - iz_tne)    * gOut;
        giz += bsw_val * (ix_tne - ix)    * (iy    - iy_tne) * gOut;
      }
      if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
        float bse_val = inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW];
        gix += bse_val * (iy - iy_tnw)    * (iz - iz_tnw)    * gOut;
        giy += bse_val * (ix    - ix_tnw) * (iz - iz_tnw)    * gOut;
        giz += bse_val * (ix    - ix_tnw) * (iy    - iy_tnw) * gOut;
      }
    }

    return make_float3(gix_mult * gix, giy_mult * giy, giz_mult * giz);
}

// this dummy struct necessary because c++ is dumb
template<typename out_t>
struct GridSampler {
    static __forceinline__ __device__ out_t forward(int C, int inp_D, int inp_H, int inp_W,
            float* vals, float3 pos, bool border) {
        return grid_sample_forward<out_t>(C, inp_D, inp_H, inp_W, vals, pos, border);        
    }

    static __forceinline__ __device__ float3 backward(int C, int inp_D, int inp_H, int inp_W,
            float* vals, float* grad_vals, float3 pos, out_t grad_out, bool border) {
        return grid_sample_backward<out_t>(C, inp_D, inp_H, inp_W, vals, grad_vals, pos, grad_out, border);
    }
};

//template <typename T>
//__device__ void cswap ( T& a, T& b ) {
//    T c(a); a=b; b=c;
//}

static __forceinline__ __device__
int within_bounds_3d_ind(int d, int h, int w, int D, int H, int W) {
    return d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W ? ((d * H) + h) * W + w : -1;
}

template<class out_t>
static __device__ out_t grid_sample_chlast_forward(int, int inp_D, int inp_H,
        int inp_W, float * vals, float3 pos, bool border) {
    int inp_sW = 1, inp_sH = inp_W, inp_sD = inp_W * inp_H;

    // normalize ix, iy, iz from [-1, 1] to [0, inp_W-1] & [0, inp_H-1] & [0, inp_D-1]
    float ix = max(-100.f, min(100.f, ((pos.x + 1.f) / 2))) * (inp_W - 1);
    float iy = max(-100.f, min(100.f, ((pos.y + 1.f) / 2))) * (inp_H - 1);
    float iz = max(-100.f, min(100.f, ((pos.z + 1.f) / 2))) * (inp_D - 1);

    if (border) {
        // clip coordinates to image borders
        ix = clip_coordinates(ix, inp_W);
        iy = clip_coordinates(iy, inp_H);
        iz = clip_coordinates(iz, inp_D);
    }

    // get corner pixel values from (x, y, z)
    // for 4d, we used north-east-south-west
    // for 5d, we add top-bottom
    int ix_tnw = static_cast<int>(::floor(ix));
    int iy_tnw = static_cast<int>(::floor(iy));
    int iz_tnw = static_cast<int>(::floor(iz));

    int ix_tne = ix_tnw + 1;
    int iy_tne = iy_tnw;
    int iz_tne = iz_tnw;

    int ix_tsw = ix_tnw;
    int iy_tsw = iy_tnw + 1;
    int iz_tsw = iz_tnw;

    int ix_tse = ix_tnw + 1;
    int iy_tse = iy_tnw + 1;
    int iz_tse = iz_tnw;

    int ix_bnw = ix_tnw;
    int iy_bnw = iy_tnw;
    int iz_bnw = iz_tnw + 1;

    int ix_bne = ix_tnw + 1;
    int iy_bne = iy_tnw;
    int iz_bne = iz_tnw + 1;

    int ix_bsw = ix_tnw;
    int iy_bsw = iy_tnw + 1;
    int iz_bsw = iz_tnw + 1;

    int ix_bse = ix_tnw + 1;
    int iy_bse = iy_tnw + 1;
    int iz_bse = iz_tnw + 1;

    // get surfaces to each neighbor:
    float tnw = (ix_bse - ix)    * (iy_bse - iy)    * (iz_bse - iz);
    float tne = (ix    - ix_bsw) * (iy_bsw - iy)    * (iz_bsw - iz);
    float tsw = (ix_bne - ix)    * (iy    - iy_bne) * (iz_bne - iz);
    float tse = (ix    - ix_bnw) * (iy    - iy_bnw) * (iz_bnw - iz);
    float bnw = (ix_tse - ix)    * (iy_tse - iy)    * (iz - iz_tse);
    float bne = (ix    - ix_tsw) * (iy_tsw - iy)    * (iz - iz_tsw);
    float bsw = (ix_tne - ix)    * (iy    - iy_tne) * (iz - iz_tne);
    float bse = (ix    - ix_tnw) * (iy    - iy_tnw) * (iz - iz_tnw);

    out_t result;
    memset(&result, 0, sizeof(out_t));
    out_t * inp_ptr_NC = (out_t*)vals;
    out_t * out_ptr_NCDHW = &result;
    {
        if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
            *out_ptr_NCDHW += inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW] * tnw;
        }
        if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
            *out_ptr_NCDHW += inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW] * tne;
        }
        if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
            *out_ptr_NCDHW += inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW] * tsw;
        }
        if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
            *out_ptr_NCDHW += inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW] * tse;
        }
        if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
            *out_ptr_NCDHW += inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW] * bnw;
        }
        if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
            *out_ptr_NCDHW += inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW] * bne;
        }
        if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
            *out_ptr_NCDHW += inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW] * bsw;
        }
        if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
            *out_ptr_NCDHW += inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW] * bse;
        }
    }

    return result;
}

template<typename out_t>
static __device__ float3 grid_sample_chlast_backward(int, int inp_D, int inp_H,
        int inp_W, float* vals, float* grad_vals, float3 pos, out_t grad_out,
        bool border) {
    int inp_sW = 1, inp_sH = inp_W, inp_sD = inp_W * inp_H;
    int gInp_sW = 1, gInp_sH = inp_W, gInp_sD = inp_W * inp_H;

    // normalize ix, iy, iz from [-1, 1] to [0, inp_W-1] & [0, inp_H-1] & [0, inp_D-1]
    float ix = max(-100.f, min(100.f, ((pos.x + 1.f) / 2))) * (inp_W - 1);
    float iy = max(-100.f, min(100.f, ((pos.y + 1.f) / 2))) * (inp_H - 1);
    float iz = max(-100.f, min(100.f, ((pos.z + 1.f) / 2))) * (inp_D - 1);

    float gix_mult = (inp_W - 1.f) / 2;
    float giy_mult = (inp_H - 1.f) / 2;
    float giz_mult = (inp_D - 1.f) / 2;

    if (border) {
        // clip coordinates to image borders
        ix = clip_coordinates_set_grad(ix, inp_W, &gix_mult);
        iy = clip_coordinates_set_grad(iy, inp_H, &giy_mult);
        iz = clip_coordinates_set_grad(iz, inp_D, &giz_mult);
    }

    // get corner pixel values from (x, y, z)
    // for 4d, we used north-east-south-west
    // for 5d, we add top-bottom
    int ix_tnw = static_cast<int>(::floor(ix));
    int iy_tnw = static_cast<int>(::floor(iy));
    int iz_tnw = static_cast<int>(::floor(iz));

    int ix_tne = ix_tnw + 1;
    int iy_tne = iy_tnw;
    int iz_tne = iz_tnw;

    int ix_tsw = ix_tnw;
    int iy_tsw = iy_tnw + 1;
    int iz_tsw = iz_tnw;

    int ix_tse = ix_tnw + 1;
    int iy_tse = iy_tnw + 1;
    int iz_tse = iz_tnw;

    int ix_bnw = ix_tnw;
    int iy_bnw = iy_tnw;
    int iz_bnw = iz_tnw + 1;

    int ix_bne = ix_tnw + 1;
    int iy_bne = iy_tnw;
    int iz_bne = iz_tnw + 1;

    int ix_bsw = ix_tnw;
    int iy_bsw = iy_tnw + 1;
    int iz_bsw = iz_tnw + 1;

    int ix_bse = ix_tnw + 1;
    int iy_bse = iy_tnw + 1;
    int iz_bse = iz_tnw + 1;

    // get surfaces to each neighbor:
    float tnw = (ix_bse - ix)    * (iy_bse - iy)    * (iz_bse - iz);
    float tne = (ix    - ix_bsw) * (iy_bsw - iy)    * (iz_bsw - iz);
    float tsw = (ix_bne - ix)    * (iy    - iy_bne) * (iz_bne - iz);
    float tse = (ix    - ix_bnw) * (iy    - iy_bnw) * (iz_bnw - iz);
    float bnw = (ix_tse - ix)    * (iy_tse - iy)    * (iz - iz_tse);
    float bne = (ix    - ix_tsw) * (iy_tsw - iy)    * (iz - iz_tsw);
    float bsw = (ix_tne - ix)    * (iy    - iy_tne) * (iz - iz_tne);
    float bse = (ix    - ix_tnw) * (iy    - iy_tnw) * (iz - iz_tnw);

    float gix = static_cast<float>(0), giy = static_cast<float>(0), giz = static_cast<float>(0);
    out_t *gOut_ptr_NCDHW = &grad_out;
    out_t *gInp_ptr_NC = (out_t*)grad_vals;
    out_t *inp_ptr_NC = (out_t*)vals;

    // calculate bilinear weighted pixel value and set output pixel
    {
      out_t gOut = *gOut_ptr_NCDHW;

      // calculate and set grad_input
      safe_add_3d(gInp_ptr_NC, iz_tnw, iy_tnw, ix_tnw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tnw * gOut);
      safe_add_3d(gInp_ptr_NC, iz_tne, iy_tne, ix_tne, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tne * gOut);
      safe_add_3d(gInp_ptr_NC, iz_tsw, iy_tsw, ix_tsw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tsw * gOut);
      safe_add_3d(gInp_ptr_NC, iz_tse, iy_tse, ix_tse, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tse * gOut);
      safe_add_3d(gInp_ptr_NC, iz_bnw, iy_bnw, ix_bnw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bnw * gOut);
      safe_add_3d(gInp_ptr_NC, iz_bne, iy_bne, ix_bne, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bne * gOut);
      safe_add_3d(gInp_ptr_NC, iz_bsw, iy_bsw, ix_bsw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bsw * gOut);
      safe_add_3d(gInp_ptr_NC, iz_bse, iy_bse, ix_bse, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bse * gOut);

      // calculate grad_grid
      if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
        out_t tnw_val = inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW];
        gix -= (iy_bse - iy)    * (iz_bse - iz)    * dot(tnw_val, gOut);
        giy -= (ix_bse - ix)    * (iz_bse - iz)    * dot(tnw_val, gOut);
        giz -= (ix_bse - ix)    * (iy_bse - iy)    * dot(tnw_val, gOut);
      }
      if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
        out_t tne_val = inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW];
        gix += (iy_bsw - iy)    * (iz_bsw - iz)    * dot(tne_val, gOut);
        giy -= (ix    - ix_bsw) * (iz_bsw - iz)    * dot(tne_val, gOut);
        giz -= (ix    - ix_bsw) * (iy_bsw - iy)    * dot(tne_val, gOut);
      }
      if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
        out_t tsw_val = inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW];
        gix -= (iy - iy_bne)    * (iz_bne - iz)    * dot(tsw_val, gOut);
        giy += (ix_bne - ix)    * (iz_bne - iz)    * dot(tsw_val, gOut);
        giz -= (ix_bne - ix)    * (iy    - iy_bne) * dot(tsw_val, gOut);
      }
      if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
        out_t tse_val = inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW];
        gix += (iy - iy_bnw)    * (iz_bnw - iz)    * dot(tse_val, gOut);
        giy += (ix    - ix_bnw) * (iz_bnw - iz)    * dot(tse_val, gOut);
        giz -= (ix    - ix_bnw) * (iy    - iy_bnw) * dot(tse_val, gOut);
      }
      if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
        out_t bnw_val = inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW];
        gix -= (iy_tse - iy)    * (iz - iz_tse)    * dot(bnw_val, gOut);
        giy -= (ix_tse - ix)    * (iz - iz_tse)    * dot(bnw_val, gOut);
        giz += (ix_tse - ix)    * (iy_tse - iy)    * dot(bnw_val, gOut);
      }
      if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
        out_t bne_val = inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW];
        gix += (iy_tsw - iy)    * (iz - iz_tsw)    * dot(bne_val, gOut);
        giy -= (ix    - ix_tsw) * (iz - iz_tsw)    * dot(bne_val, gOut);
        giz += (ix    - ix_tsw) * (iy_tsw - iy)    * dot(bne_val, gOut);
      }
      if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
        out_t bsw_val = inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW];
        gix -= (iy - iy_tne)    * (iz - iz_tne)    * dot(bsw_val, gOut);
        giy += (ix_tne - ix)    * (iz - iz_tne)    * dot(bsw_val, gOut);
        giz += (ix_tne - ix)    * (iy    - iy_tne) * dot(bsw_val, gOut);
      }
      if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
        out_t bse_val = inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW];
        gix += (iy - iy_tnw)    * (iz - iz_tnw)    * dot(bse_val, gOut);
        giy += (ix    - ix_tnw) * (iz - iz_tnw)    * dot(bse_val, gOut);
        giz += (ix    - ix_tnw) * (iy    - iy_tnw) * dot(bse_val, gOut);
      }
    }

    return make_float3(gix_mult * gix, giy_mult * giy, giz_mult * giz);
}

template<typename out_t>
struct GridSamplerChlast {
    static __forceinline__ __device__ out_t forward(int C, int inp_D, int inp_H, int inp_W,
            float* vals, float3 pos, bool border) {
        return grid_sample_chlast_forward<out_t>(C, inp_D, inp_H, inp_W, vals, pos, border);        
    }

    static __forceinline__ __device__ float3 backward(int C, int inp_D, int inp_H, int inp_W,
            float* vals, float* grad_vals, float3 pos, out_t grad_out, bool border) {
        return grid_sample_chlast_backward<out_t>(C, inp_D, inp_H, inp_W, vals, grad_vals, pos, grad_out, border);
    }
};


inline __host__ __device__ float min_component(float3 a) {
    return fminf(fminf(a.x,a.y),a.z);
}

inline __host__ __device__ float max_component(float3 a) {
    return fmaxf(fmaxf(a.x,a.y),a.z);
}

 inline __host__ __device__ float3 abs(float3 a) {
    return make_float3(abs(a.x), abs(a.y), abs(a.z));
}

__forceinline__ __device__ bool ray_aabb_hit(float3 p0, float3 p1, float3 raypos, float3 raydir) {
    float3 t0 = (p0 - raypos) / raydir;
    float3 t1 = (p1 - raypos) / raydir;
    float3 tmin = fminf(t0,t1), tmax = fmaxf(t0,t1);
  
    return max_component(tmin) <= min_component(tmax);
}

__forceinline__ __device__ bool ray_aabb_hit_ird(float3 p0, float3 p1, float3 raypos, float3 ird) {
    float3 t0 = (p0 - raypos) * ird;
    float3 t1 = (p1 - raypos) * ird;
    float3 tmin = fminf(t0,t1), tmax = fmaxf(t0,t1);
  
    return max_component(tmin) <= min_component(tmax);

}
__forceinline__ __device__ void ray_aabb_hit_ird_tminmax(float3 p0, float3 p1,
        float3 raypos, float3 ird, float &otmin, float &otmax) {
    float3 t0 = (p0 - raypos) * ird;
    float3 t1 = (p1 - raypos) * ird;
    float3 tmin = fminf(t0,t1), tmax = fmaxf(t0,t1);
    tmin = fminf(t0,t1);
    tmax = fmaxf(t0,t1);
    otmin = max_component(tmin);
    otmax = min_component(tmax);
}

inline  __device__ bool aabb_intersect(float3 p0, float3 p1, float3 r0, float3 rd, float &tmin, float &tmax) {
    float tymin, tymax, tzmin, tzmax;
    const float3 bounds[2] = {p0, p1};
    float3 ird = 1.0f/rd;
    int sx = (ird.x<0) ? 1 : 0;
    int sy = (ird.y<0) ? 1 : 0;
    int sz = (ird.z<0) ? 1 : 0;
    tmin = (bounds[sx].x - r0.x) * ird.x;
    tmax = (bounds[1-sx].x - r0.x) * ird.x;
    tymin = (bounds[sy].y - r0.y) * ird.y;
    tymax = (bounds[1-sy].y - r0.y) * ird.y;

    if ((tmin > tymax) || (tymin > tmax))
        return false;
    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;

    tzmin = (bounds[sz].z - r0.z) * ird.z;
    tzmax = (bounds[1-sz].z - r0.z) * ird.z;

    if ((tmin > tzmax) || (tzmin > tmax))
        return false;
    if (tzmin > tmin)
        tmin = tzmin;
    if (tzmax < tmax)
        tmax = tzmax;

    return true;
}

template<bool sortboxes, int maxhitboxes, bool sync, class PrimTransfT>
static __forceinline__ __device__ void ray_subset_fixedbvh(
        unsigned warpmask,
        int K,
        float3 raypos,
        float3 raydir,
        float2 tminmax,
        float2 &rtminmax,
        int * sortedobjid,
        int2 * nodechildren,
        float3 * nodeaabb,
        const typename PrimTransfT::Data & primtransf_data,
        int *hitboxes,
        int & num) {
    float3 iraydir = 1.0f/raydir;
    int stack[64];
    int* stack_ptr = stack;
    *stack_ptr++ = -1;
    int node = 0;
    do {
        // check if we're in a leaf
        if (node >= (K - 1)) {
            {
                int k = node - (K - 1);

                float3 r0, rd;
                PrimTransfT::forward2(primtransf_data, k, raypos, raydir, r0, rd);

                float3 ird = 1.0f/rd;
                float3 t0 = (-1.f - r0) * ird;
                float3 t1 = (1.f - r0) * ird;
                float3 tmin = fminf(t0,t1), tmax = fmaxf(t0,t1);

                float trmin = max_component(tmin);
                float trmax = min_component(tmax);

                bool intersection = trmin <= trmax;

                if (intersection) {
                    // hit
                    rtminmax.x = fminf(rtminmax.x, trmin);
                    rtminmax.y = fmaxf(rtminmax.y, trmax);
                }

                if (sync) {
                    intersection = __any_sync(warpmask, intersection);
                }

                if (intersection) {
                    if (sortboxes) {
                        if (num < maxhitboxes) {
                            int j = num - 1;
                            while (j >= 0 && hitboxes[j] > k) {
                                hitboxes[j + 1] = hitboxes[j];
                                j = j - 1;
                            }
                            hitboxes[j + 1] = k;
                            num++;
                        }
                    } else {
                        if (num < maxhitboxes) {
                            hitboxes[num++] = k;
                        }
                    }
                }
            }

            node = *--stack_ptr;
        } else {
            int2 children = make_int2(node * 2 + 1, node * 2 + 2);

            // check if we're in each child's bbox
            float3 * nodeaabb_ptr = nodeaabb + children.x * 2;
            bool traverse_l = ray_aabb_hit_ird(nodeaabb_ptr[0], nodeaabb_ptr[1], raypos, iraydir);
            bool traverse_r = ray_aabb_hit_ird(nodeaabb_ptr[2], nodeaabb_ptr[3], raypos, iraydir);

            if (sync) {
                traverse_l = __any_sync(warpmask, traverse_l);
                traverse_r = __any_sync(warpmask, traverse_r);
            }

            // update stack
            if (!traverse_l && !traverse_r) {
                node = *--stack_ptr;
            } else {
                node = traverse_l ? children.x : children.y;
                if (traverse_l && traverse_r) {
                    *stack_ptr++ = children.y;
                }
            }

            if (sync) {
                __syncwarp(warpmask);
            }
        }
    } while (node != -1);
}

template<bool sortboxes, int maxhitboxes, bool sync, class PrimTransfT>
struct RaySubsetFixedBVH {
    static __forceinline__ __device__ void forward(
        unsigned warpmask,
        int K,
        float3 raypos,
        float3 raydir,
        float2 tminmax,
        float2 &rtminmax,
        int * sortedobjid,
        int2 * nodechildren,
        float3 * nodeaabb,
        const typename PrimTransfT::Data & primtransf_data,
        int *hitboxes,
        int & num) {
        ray_subset_fixedbvh<sortboxes, maxhitboxes, sync, PrimTransfT>(
                warpmask, K, raypos, raydir, tminmax, rtminmax,
                sortedobjid, nodechildren, nodeaabb, primtransf_data, hitboxes, num);
    }
};

#endif
