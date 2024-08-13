// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef MVPRAYMARCHER_PRIMACCUM_H_
#define MVPRAYMARCHER_PRIMACCUM_H_

struct PrimAccumDataBase {
    typedef PrimAccumDataBase base;
};

struct PrimAccumAdditive {
    struct Data : public PrimAccumDataBase {
        float termthresh;

        int nstride, hstride, wstride;
        float4 * rayrgbaim;
        float4 * grad_rayrgbaim;
        float3 * raysatim;

        __forceinline__ __device__ void n_stride(int n, int h, int w) {
            rayrgbaim += n * nstride + h * hstride + w * wstride;
            grad_rayrgbaim += n * nstride + h * hstride + w * wstride;
            if (raysatim) {
                raysatim += n * nstride + h * hstride + w * wstride;
            }
        }
    };

    float4 rayrgba;
    float3 raysat;
    bool sat;
    float4 dL_rayrgba;

    __forceinline__ __device__ PrimAccumAdditive() :
        rayrgba(make_float4(0.f)),
        raysat(make_float3(-1.f)),
        sat(false) {
    }

    __forceinline__ __device__ bool is_done() const {
        return sat;
    }

    __forceinline__ __device__ int get_nsteps() const {
        return 0;
    }

    __forceinline__ __device__ void write(const Data & data) {
        *data.rayrgbaim = rayrgba;
        if (data.raysatim) {
            *data.raysatim = raysat;
        }
    }

    __forceinline__ __device__ void read(const Data & data) {
        dL_rayrgba = *data.grad_rayrgbaim;
        raysat = *data.raysatim;
    }

    __forceinline__ __device__ void forward_prim(const Data & data, float4 sample, float stepsize) {
        // accumulate
        float3 rgb = make_float3(sample);
        float alpha = sample.w;
        float newalpha = rayrgba.w + alpha * stepsize;
        float contrib = fminf(newalpha, 1.f) - rayrgba.w;

        rayrgba += make_float4(rgb, 1.f) * contrib;

        if (newalpha >= 1.f) {
            // save saturation point
            if (!sat) {
                raysat = rgb;
            }
            sat = true;
        }
    }

    __forceinline__ __device__ float4 forwardbackward_prim(const Data & data, float4 sample, float stepsize) {
        float3 rgb = make_float3(sample);
        float4 rgb1 = make_float4(rgb, 1.f);
        sample.w *= stepsize;

        bool thissat = rayrgba.w + sample.w >= 1.f;
        sat = sat || thissat;

        float weight = sat ? (1.f - rayrgba.w) : sample.w;

        float3 dL_rgb = weight * make_float3(dL_rayrgba);
        float dL_alpha = sat ? 0.f : 
            stepsize * dot(rgb1 - (raysat.x > -1.f ? make_float4(raysat, 1.f) : make_float4(0.f)), dL_rayrgba);

        rayrgba += make_float4(rgb, 1.f) * weight;

        return make_float4(dL_rgb, dL_alpha);
    }
};

#endif
