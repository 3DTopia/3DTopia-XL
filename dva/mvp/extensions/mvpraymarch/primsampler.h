// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef MVPRAYMARCHER_PRIMSAMPLER_H_
#define MVPRAYMARCHER_PRIMSAMPLER_H_

struct PrimSamplerDataBase {
    typedef PrimSamplerDataBase base;
};

template<
    bool dowarp,
    template<typename> class GridSamplerT=GridSamplerChlast>
struct PrimSamplerTW {
    struct Data : public PrimSamplerDataBase {
        float fadescale, fadeexp;

        int tplate_nstride;
        int TD, TH, TW;
        float * tplate;
        float * grad_tplate;

        int warp_nstride;
        int WD, WH, WW;
        float * warp;
        float * grad_warp;

        __forceinline__ __device__ void n_stride(int n) {
            tplate += n * tplate_nstride;
            grad_tplate += n * tplate_nstride;
            warp += n * warp_nstride;
            grad_warp += n * warp_nstride;
        }
    };

    float fade;
    float * tplate_ptr;
    float * warp_ptr;
    float3 yy1;

    __forceinline__ __device__ float4 forward(
            const Data & data,
            int k,
            float3 y0) {
        fade = __expf(-data.fadescale * (
                    __powf(abs(y0.x), data.fadeexp) +
                    __powf(abs(y0.y), data.fadeexp) +
                    __powf(abs(y0.z), data.fadeexp)));

        if (dowarp) {
            warp_ptr = data.warp + (k * 3 * data.WD * data.WH * data.WW);
            yy1 = GridSamplerT<float3>::forward(3, data.WD, data.WH, data.WW, warp_ptr, y0, false);
        } else {
            yy1 = y0;
        }

        tplate_ptr = data.tplate + (k * 4 * data.TD * data.TH * data.TW);
        float4 sample = GridSamplerT<float4>::forward(4, data.TD, data.TH, data.TW, tplate_ptr, yy1, false);

        sample.w *= fade;

        return sample;
    }

    __forceinline__ __device__ float3 backward(const Data & data, int k, float3 y0,
            float4 sample, float4 dL_sample, bool validthread) {
        float3 dfade_y0 = -(data.fadescale * data.fadeexp) * make_float3(
                    __powf(abs(y0.x), data.fadeexp - 1.f) * (y0.x > 0.f ? 1.f : -1.f),
                    __powf(abs(y0.y), data.fadeexp - 1.f) * (y0.y > 0.f ? 1.f : -1.f),
                    __powf(abs(y0.z), data.fadeexp - 1.f) * (y0.z > 0.f ? 1.f : -1.f));
        float3 dL_y0 = dfade_y0 * sample.w * dL_sample.w;

        dL_sample.w *= fade;

        float * grad_tplate_ptr = data.grad_tplate + (k * 4 * data.TD * data.TH * data.TW);
        float3 dL_y1 = GridSamplerT<float4>::backward(4, data.TD, data.TH, data.TW,
                tplate_ptr, grad_tplate_ptr, yy1, validthread ? dL_sample : make_float4(0.f), false);

        if (dowarp) {
            float * grad_warp_ptr = data.grad_warp + (k * 3 * data.WD * data.WH * data.WW);
            dL_y0 += GridSamplerT<float3>::backward(3, data.WD, data.WH, data.WW,
                    warp_ptr, grad_warp_ptr, y0, validthread ? dL_y1 : make_float3(0.f), false);
        } else {
            dL_y0 += dL_y1;
        }

        return dL_y0;
    }
};

#endif
