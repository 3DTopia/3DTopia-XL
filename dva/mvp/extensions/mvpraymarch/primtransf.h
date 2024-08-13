// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef MVPRAYMARCHER_PRIMTRANSF_H_
#define MVPRAYMARCHER_PRIMTRANSF_H_

#include "utils.h"

__forceinline__ __device__ void compute_aabb_srt(
        float3 pt, float3 pr0, float3 pr1, float3 pr2, float3 ps,
        float3 & pmin, float3 & pmax) {
    float3 p;
    p = make_float3(-1.f, -1.f, -1.f) / ps;
    p = make_float3(dot(p, pr0), dot(p, pr1), dot(p, pr2)) + pt;

    pmin = p;
    pmax = p;

    p = make_float3(1.f, -1.f, -1.f) / ps;
    p = make_float3(dot(p, pr0), dot(p, pr1), dot(p, pr2)) + pt;

    pmin = fminf(pmin, p);
    pmax = fmaxf(pmax, p);

    p = make_float3(-1.f, 1.f, -1.f) / ps;
    p = make_float3(dot(p, pr0), dot(p, pr1), dot(p, pr2)) + pt;

    pmin = fminf(pmin, p);
    pmax = fmaxf(pmax, p);

    p = make_float3(1.f, 1.f, -1.f) / ps;
    p = make_float3(dot(p, pr0), dot(p, pr1), dot(p, pr2)) + pt;

    pmin = fminf(pmin, p);
    pmax = fmaxf(pmax, p);

    p = make_float3(-1.f, -1.f, 1.f) / ps;
    p = make_float3(dot(p, pr0), dot(p, pr1), dot(p, pr2)) + pt;

    pmin = fminf(pmin, p);
    pmax = fmaxf(pmax, p);

    p = make_float3(1.f, -1.f, 1.f) / ps;
    p = make_float3(dot(p, pr0), dot(p, pr1), dot(p, pr2)) + pt;

    pmin = fminf(pmin, p);
    pmax = fmaxf(pmax, p);

    p = make_float3(-1.f, 1.f, 1.f) / ps;
    p = make_float3(dot(p, pr0), dot(p, pr1), dot(p, pr2)) + pt;

    pmin = fminf(pmin, p);
    pmax = fmaxf(pmax, p);

    p = make_float3(1.f, 1.f, 1.f) / ps;
    p = make_float3(dot(p, pr0), dot(p, pr1), dot(p, pr2)) + pt;

    pmin = fminf(pmin, p);
    pmax = fmaxf(pmax, p);
}

struct PrimTransfDataBase {
    typedef PrimTransfDataBase base;
};

struct PrimTransfSRT {
    struct Data : public PrimTransfDataBase {
        int primpos_nstride;
        float3 * primpos;
        float3 * grad_primpos;
        int primrot_nstride;
        float3 * primrot;
        float3 * grad_primrot;
        int primscale_nstride;
        float3 * primscale;
        float3 * grad_primscale;

        __forceinline__ __device__ void n_stride(int n) {
            primpos += n * primpos_nstride;
            grad_primpos += n * primpos_nstride;
            primrot += n * primrot_nstride;
            grad_primrot += n * primrot_nstride;
            primscale += n * primscale_nstride;
            grad_primscale += n * primscale_nstride;
        }

        __forceinline__ __device__ float3 get_center(int n, int k) {
            return primpos[n * primpos_nstride + k];
        }

        __forceinline__ __device__ void compute_aabb(int n, int k, float3 & pmin, float3 & pmax) {
            float3 pt = primpos[n * primpos_nstride + k];
            float3 pr0 = primrot[n * primrot_nstride + k * 3 + 0];
            float3 pr1 = primrot[n * primrot_nstride + k * 3 + 1];
            float3 pr2 = primrot[n * primrot_nstride + k * 3 + 2];
            float3 ps = primscale[n * primscale_nstride + k];

            compute_aabb_srt(pt, pr0, pr1, pr2, ps, pmin, pmax);
        }
    };

    float3 xmt;
    float3 pr0;
    float3 pr1;
    float3 pr2;
    float3 rxmt;
    float3 ps;

    static __forceinline__ __device__ bool valid(float3 pos) {
        return (
            pos.x > -1.f && pos.x < 1.f &&
            pos.y > -1.f && pos.y < 1.f &&
            pos.z > -1.f && pos.z < 1.f);
    }

    __forceinline__ __device__ float3 forward(
            const Data & data,
            int k,
            float3 x) {
        float3 pt = data.primpos[k];
        pr0 = data.primrot[(k) * 3 + 0];
        pr1 = data.primrot[(k) * 3 + 1];
        pr2 = data.primrot[(k) * 3 + 2];
        ps = data.primscale[k];
        xmt = x - pt;
        rxmt = pr0 * xmt.x + pr1 * xmt.y + pr2 * xmt.z;
        float3 y0 = rxmt * ps;
        return y0;
    }

    static __forceinline__ __device__ void forward2(
            const Data & data,
            int k,
            float3 r, float3 d, float3 & rout, float3 & dout) {
        float3 pt = data.primpos[k];
        float3 pr0 = data.primrot[k * 3 + 0];
        float3 pr1 = data.primrot[k * 3 + 1];
        float3 pr2 = data.primrot[k * 3 + 2];
        float3 ps = data.primscale[k];
        float3 xmt = r - pt;
        float3 dmt = d;
        float3 rxmt = pr0 * xmt.x;
        float3 rdmt = pr0 * dmt.x;
        rxmt += pr1 * xmt.y;
        rdmt += pr1 * dmt.y;
        rxmt += pr2 * xmt.z;
        rdmt += pr2 * dmt.z;
        rout = rxmt * ps;
        dout = rdmt * ps;
    }

    __forceinline__ __device__ void backward(const Data & data, int k, float3 x, float3 dL_y0, bool validthread) {
        fastAtomicAdd((float*)data.grad_primscale + k * 3 + 0, validthread ? rxmt.x * dL_y0.x : 0.f);
        fastAtomicAdd((float*)data.grad_primscale + k * 3 + 1, validthread ? rxmt.y * dL_y0.y : 0.f);
        fastAtomicAdd((float*)data.grad_primscale + k * 3 + 2, validthread ? rxmt.z * dL_y0.z : 0.f);

        dL_y0 *= ps;
        float3 gpr0 = xmt.x * dL_y0;
        fastAtomicAdd((float*)data.grad_primrot + (k * 3 + 0) * 3 + 0, validthread ? gpr0.x : 0.f);
        fastAtomicAdd((float*)data.grad_primrot + (k * 3 + 0) * 3 + 1, validthread ? gpr0.y : 0.f);
        fastAtomicAdd((float*)data.grad_primrot + (k * 3 + 0) * 3 + 2, validthread ? gpr0.z : 0.f);

        float3 gpr1 = xmt.y * dL_y0;
        fastAtomicAdd((float*)data.grad_primrot + (k * 3 + 1) * 3 + 0, validthread ? gpr1.x : 0.f);
        fastAtomicAdd((float*)data.grad_primrot + (k * 3 + 1) * 3 + 1, validthread ? gpr1.y : 0.f);
        fastAtomicAdd((float*)data.grad_primrot + (k * 3 + 1) * 3 + 2, validthread ? gpr1.z : 0.f);

        float3 gpr2 = xmt.z * dL_y0;
        fastAtomicAdd((float*)data.grad_primrot + (k * 3 + 2) * 3 + 0, validthread ? gpr2.x : 0.f);
        fastAtomicAdd((float*)data.grad_primrot + (k * 3 + 2) * 3 + 1, validthread ? gpr2.y : 0.f);
        fastAtomicAdd((float*)data.grad_primrot + (k * 3 + 2) * 3 + 2, validthread ? gpr2.z : 0.f);

        fastAtomicAdd((float*)data.grad_primpos + k * 3 + 0, validthread ? -dot(pr0, dL_y0) : 0.f);
        fastAtomicAdd((float*)data.grad_primpos + k * 3 + 1, validthread ? -dot(pr1, dL_y0) : 0.f);
        fastAtomicAdd((float*)data.grad_primpos + k * 3 + 2, validthread ? -dot(pr2, dL_y0) : 0.f);
    }
};

#endif
