// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

template<
    int maxhitboxes,
    int nwarps,
    class RaySubsetT=RaySubsetFixedBVH<false, 512, true, PrimTransfSRT>,
    class PrimTransfT=PrimTransfSRT,
    class PrimSamplerT=PrimSamplerTW<false>,
    class PrimAccumT=PrimAccumAdditive>
__global__ void raymarch_subset_forward_kernel(
        int N, int H, int W, int K,
        float3 * rayposim,
        float3 * raydirim,
        float stepsize,
        float2 * tminmaxim,
        int * sortedobjid,
        int2 * nodechildren,
        float3 * nodeaabb,
        typename PrimTransfT::Data primtransf_data,
        typename PrimSamplerT::Data primsampler_data,
        typename PrimAccumT::Data primaccum_data
        ) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z;
    bool validthread = (w < W) && (h < H) && (n<N);

    assert(nwarps == 0 || blockDim.x * blockDim.y / 32 <= nwarps);
    const int warpid = __shfl_sync(0xffffffff, (threadIdx.y * blockDim.x + threadIdx.x) / 32, 0);
    assert(__match_any_sync(0xffffffff, (threadIdx.y * blockDim.x + threadIdx.x) / 32) == 0xffffffff);

    // warpmask contains the valid threads in the warp
    unsigned warpmask = 0xffffffff;
    n = min(N - 1, n);
    h = min(H - 1, h);
    w = min(W - 1, w);

    sortedobjid += n * K;
    nodechildren += n * (K + K - 1);
    nodeaabb += n * (K + K - 1) * 2;

    primtransf_data.n_stride(n);
    primsampler_data.n_stride(n);
    primaccum_data.n_stride(n, h, w);

    float3 raypos = rayposim[n * H * W + h * W + w];
    float3 raydir = raydirim[n * H * W + h * W + w];
    float2 tminmax = tminmaxim[n * H * W + h * W + w];

    int hitboxes[nwarps > 0 ? 1 : maxhitboxes];
    __shared__ int hitboxes_sh[nwarps > 0 ? maxhitboxes * nwarps : 1];
    int * hitboxes_ptr = nwarps > 0 ? hitboxes_sh + maxhitboxes * warpid : hitboxes;
    int nhitboxes = 0;

    // find raytminmax
    float2 rtminmax = make_float2(std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
    RaySubsetT::forward(warpmask, K, raypos, raydir, tminmax, rtminmax,
            sortedobjid, nodechildren, nodeaabb,
            primtransf_data, hitboxes_ptr, nhitboxes);
    rtminmax.x = max(rtminmax.x, tminmax.x);
    rtminmax.y = min(rtminmax.y, tminmax.y);
    __syncwarp(warpmask);

    float t = tminmax.x;
    raypos = raypos + raydir * tminmax.x;

    int incs = floor((rtminmax.x - t) / stepsize);
    t += incs * stepsize;
    raypos += raydir * incs * stepsize;

    PrimAccumT pa;

    while (!__all_sync(warpmask, t > rtminmax.y + 1e-5f || pa.is_done())) {
        for (int ks = 0; ks < nhitboxes; ++ks) {
            int k = hitboxes_ptr[ks];

            // compute primitive-relative coordinate
            PrimTransfT pt;
            float3 samplepos = pt.forward(primtransf_data, k, raypos);

            if (pt.valid(samplepos) && !pa.is_done() && t < rtminmax.y + 1e-5f) {
                // sample
                PrimSamplerT ps;
                float4 sample = ps.forward(primsampler_data, k, samplepos);

                // accumulate
                pa.forward_prim(primaccum_data, sample, stepsize);
            }
        }

        // update position
        t += stepsize;
        raypos += raydir * stepsize;
    }

    pa.write(primaccum_data);
}

template <
    bool forwarddir,
    int maxhitboxes,
    int nwarps,
    class RaySubsetT=RaySubsetFixedBVH<false, 512, true, PrimTransfSRT>,
    class PrimTransfT=PrimTransfSRT,
    class PrimSamplerT=PrimSamplerTW<false>,
    class PrimAccumT=PrimAccumAdditive>
__global__ void raymarch_subset_backward_kernel(
        int N, int H, int W, int K,
        float3 * rayposim,
        float3 * raydirim,
        float stepsize,
        float2 * tminmaxim,
        int * sortedobjid,
        int2 * nodechildren,
        float3 * nodeaabb,
        typename PrimTransfT::Data primtransf_data,
        typename PrimSamplerT::Data primsampler_data,
        typename PrimAccumT::Data primaccum_data
        ) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z;
    bool validthread = (w < W) && (h < H) && (n<N);

    assert(nwarps == 0 || blockDim.x * blockDim.y / 32 <= nwarps);
    const int warpid = __shfl_sync(0xffffffff, (threadIdx.y * blockDim.x + threadIdx.x) / 32, 0);
    assert(__match_any_sync(0xffffffff, (threadIdx.y * blockDim.x + threadIdx.x) / 32) == 0xffffffff);

    // warpmask contains the valid threads in the warp
    unsigned warpmask = 0xffffffff;
    n = min(N - 1, n);
    h = min(H - 1, h);
    w = min(W - 1, w);

    sortedobjid += n * K;
    nodechildren += n * (K + K - 1);
    nodeaabb += n * (K + K - 1) * 2;

    primtransf_data.n_stride(n);
    primsampler_data.n_stride(n);
    primaccum_data.n_stride(n, h, w);

    float3 raypos = rayposim[n * H * W + h * W + w];
    float3 raydir = raydirim[n * H * W + h * W + w];
    float2 tminmax = tminmaxim[n * H * W + h * W + w];

    PrimAccumT pa;
    pa.read(primaccum_data);

    int hitboxes[nwarps > 0 ? 1 : maxhitboxes];
    __shared__ int hitboxes_sh[nwarps > 0 ? maxhitboxes * nwarps : 1];
    int * hitboxes_ptr = nwarps > 0 ? hitboxes_sh + maxhitboxes * warpid : hitboxes;
    int nhitboxes = 0;

    // find raytminmax
    float2 rtminmax = make_float2(std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
    RaySubsetT::forward(warpmask, K, raypos, raydir, tminmax, rtminmax,
            sortedobjid, nodechildren, nodeaabb,
            primtransf_data, hitboxes_ptr, nhitboxes);
    rtminmax.x = max(rtminmax.x, tminmax.x);
    rtminmax.y = min(rtminmax.y, tminmax.y);
    __syncwarp(warpmask);

    // set up raymarching position
    float t = tminmax.x;
    raypos = raypos + raydir * tminmax.x;

    int incs = floor((rtminmax.x - t) / stepsize);
    t += incs * stepsize;
    raypos += raydir * incs * stepsize;

    if (!forwarddir) {
        int nsteps = pa.get_nsteps();
        t += nsteps * stepsize;
        raypos += raydir * nsteps * stepsize;
    }

    while (__any_sync(warpmask, (
                    (forwarddir && t < rtminmax.y + 1e-5f ||
                     !forwarddir && t > rtminmax.x - 1e-5f) &&
                    !pa.is_done()))) {
        for (int ks = 0; ks < nhitboxes; ++ks) {
            int k = hitboxes_ptr[forwarddir ? ks : nhitboxes - ks - 1];

            PrimTransfT pt;
            float3 samplepos = pt.forward(primtransf_data, k, raypos);

            bool evalprim = pt.valid(samplepos) && !pa.is_done() && t < rtminmax.y + 1e-5f;

            float3 dL_samplepos = make_float3(0.f);
            if (evalprim) {
                PrimSamplerT ps;
                float4 sample = ps.forward(primsampler_data, k, samplepos);

                float4 dL_sample = pa.forwardbackward_prim(primaccum_data, sample, stepsize);

                dL_samplepos = ps.backward(primsampler_data, k, samplepos, sample, dL_sample, validthread);
            }

            if (__any_sync(warpmask, evalprim)) {
                pt.backward(primtransf_data, k, samplepos, dL_samplepos, validthread && evalprim);
            }
        }

        if (forwarddir) {
            t += stepsize;
            raypos += raydir * stepsize;
        } else {
            t -= stepsize;
            raypos -= raydir * stepsize;
        }
    }
}

