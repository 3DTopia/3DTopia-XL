// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <cmath>
#include <cstdio>
#include <functional>
#include <map>

#include "helper_math.h"

#include "cudadispatch.h"

#include "primtransf.h"

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__device__ unsigned int expand_bits(unsigned int v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__device__ unsigned int morton3D(float x, float y, float z) {
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expand_bits((unsigned int)x);
    unsigned int yy = expand_bits((unsigned int)y);
    unsigned int zz = expand_bits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}

template<typename PrimTransfT>
__global__ void compute_morton_kernel(
        int N, int K,
        typename PrimTransfT::Data data,
        int * code
        ) {
    const int count = N * K;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x) {
        const int k = index % K;
        const int n = index / K;

        //float4 c = center[n * K + k];
        float3 c = data.get_center(n, k);
        code[n * K + k] = morton3D(c.x, c.y, c.z);
    }
}

__forceinline__ __device__ int delta(int* sortedcodes, int x, int y, int K) {
	if (x >= 0 && x <= K - 1 && y >= 0 && y <= K - 1) {
        return sortedcodes[x] == sortedcodes[y] ?
            32 + __clz(x ^ y) :
            __clz(sortedcodes[x] ^ sortedcodes[y]);
    }
	return -1;
}

__forceinline__ __device__ int sign(int x) {
	return (int)(x > 0) - (int)(x < 0);
}

__device__ int find_split(
       int* sortedcodes,
       int first,
       int last,
       int K) {
    float commonPrefix = delta(sortedcodes, first, last, K);
    int split = first;
    int step = last - first;

    do {
        step = (step + 1) >> 1; // exponential decrease
        int newSplit = split + step; // proposed new position

        if (newSplit < last) {
            int splitPrefix = delta(sortedcodes, first, newSplit, K);
            if (splitPrefix > commonPrefix) {
                split = newSplit; // accept proposal
            }
        }
    } while (step > 1);

    return split;
}

__device__ int2 determine_range(int* sortedcodes, int K, int idx) {
    int d = sign(delta(sortedcodes, idx, idx + 1, K) - delta(sortedcodes, idx, idx - 1, K));
    int dmin = delta(sortedcodes, idx, idx - d, K);
    int lmax = 2;
    while (delta(sortedcodes, idx, idx + lmax * d, K) > dmin) {
        lmax = lmax * 2;
    }

    int l = 0;
    for (int t = lmax / 2; t >= 1; t /= 2) {
        if (delta(sortedcodes, idx, idx + (l + t)*d, K) > dmin) {
            l += t;
        }
    }

    int j = idx + l*d;
    int2 range;
    range.x = min(idx, j);
    range.y = max(idx, j);

    return range;
}

__global__ void build_tree_kernel(
        int N, int K,
        int * sortedcodes,
        int2 * nodechildren,
        int * nodeparent) {
    const int count = N * (K + K - 1);
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x) {
        const int k = index % (K + K - 1);
        const int n = index / (K + K - 1);

        if (k >= K - 1) {
            // leaf
            nodechildren[n * (K + K - 1) + k] = make_int2(-(k - (K - 1)) - 1, -(k - (K - 1)) - 2);
        } else {
            // internal node

            // find out which range of objects the node corresponds to
            int2 range = determine_range(sortedcodes + n * K, K, k);
            int first = range.x;
            int last = range.y;

            // determine where to split the range
            int split = find_split(sortedcodes + n * K, first, last, K);

            // select childA
            int childa = split == first ? (K - 1) + split : split;

            // select childB
            int childb = split + 1 == last ? (K - 1) + split + 1 : split + 1;

            // record parent-child relationships
            nodechildren[n * (K + K - 1) + k] = make_int2(childa, childb);
            nodeparent[n * (K + K - 1) + childa] = k;
            nodeparent[n * (K + K - 1) + childb] = k;
        }
    }
}

template<typename PrimTransfT>
__global__ void compute_aabb_kernel(
        int N, int K,
        typename PrimTransfT::Data data,
        int * sortedobjid,
        int2 * nodechildren,
        int * nodeparent,
        float3 * nodeaabb,
        int * atom) {
    const int count = N * K;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x) {
        const int k = index % K;
        const int n = index / K;

        // compute BBOX for leaf
        int kk = sortedobjid[n * K + k];

        float3 pmin;
        float3 pmax;
        data.compute_aabb(n, kk, pmin, pmax);

        nodeaabb[n * (K + K - 1) * 2 + ((K - 1) + k) * 2 + 0] = pmin;
        nodeaabb[n * (K + K - 1) * 2 + ((K - 1) + k) * 2 + 1] = pmax;

        int node = nodeparent[n * (K + K - 1) + ((K - 1) + k)];

        while (node != -1 && atomicCAS(&atom[n * (K - 1) + node], 0, 1) == 1) {
            int2 children = nodechildren[n * (K + K - 1) + node];
            float3 laabbmin = nodeaabb[n * (K + K - 1) * 2 + children.x * 2 + 0];
            float3 laabbmax = nodeaabb[n * (K + K - 1) * 2 + children.x * 2 + 1];
            float3 raabbmin = nodeaabb[n * (K + K - 1) * 2 + children.y * 2 + 0];
            float3 raabbmax = nodeaabb[n * (K + K - 1) * 2 + children.y * 2 + 1];

            float3 aabbmin = fminf(laabbmin, raabbmin);
            float3 aabbmax = fmaxf(laabbmax, raabbmax);

            nodeaabb[n * (K + K - 1) * 2 + node * 2 + 0] = aabbmin;
            nodeaabb[n * (K + K - 1) * 2 + node * 2 + 1] = aabbmax;

            node = nodeparent[n * (K + K - 1) + node];

            __threadfence();
        }
    }
}

void compute_morton_cuda(
        int N, int K,
        float * primpos,
        int * code,
        int algorithm,
        cudaStream_t stream) {
    int count = N * K;
    int blocksize = 512;
    int gridsize = (count + blocksize - 1) / blocksize;

    std::shared_ptr<PrimTransfDataBase> primtransf_data;
    primtransf_data = std::make_shared<PrimTransfSRT::Data>(PrimTransfSRT::Data{
            PrimTransfDataBase{},
            K, (float3*)primpos, nullptr,
            K * 3, nullptr, nullptr,
            K, nullptr, nullptr});

    std::map<int, std::function<void(dim3, dim3, cudaStream_t, int, int, std::shared_ptr<PrimTransfDataBase>, int*)>> dispatcher = {
      { 0, make_cudacall(compute_morton_kernel<PrimTransfSRT>) }
    };

    auto iter = dispatcher.find(min(0, algorithm));
    if (iter != dispatcher.end()) {
        (iter->second)(
            dim3(gridsize), dim3(blocksize), stream,
            N, K,
            primtransf_data,
            code);
    }
}

void build_tree_cuda(
        int N, int K,
        int * sortedcode,
        int * nodechildren,
        int * nodeparent,
        cudaStream_t stream) {
    int count = N * (K + K - 1);
    int nthreads = 512;
    int nblocks = (count + nthreads - 1) / nthreads;
    build_tree_kernel<<<nblocks, nthreads, 0, stream>>>(
            N, K,
            sortedcode,
            reinterpret_cast<int2 *>(nodechildren),
            nodeparent);
}

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
        cudaStream_t stream) {
    int * atom;
    cudaMalloc(&atom, N * (K - 1) * 4);
    cudaMemset(atom, 0, N * (K - 1) * 4);

    int count = N * K;
    int blocksize = 512;
    int gridsize = (count + blocksize - 1) / blocksize;

    std::shared_ptr<PrimTransfDataBase> primtransf_data;
    primtransf_data = std::make_shared<PrimTransfSRT::Data>(PrimTransfSRT::Data{
            PrimTransfDataBase{},
            K, (float3*)primpos, nullptr,
            K * 3, (float3*)primrot, nullptr,
            K, (float3*)primscale, nullptr});

    std::map<int, std::function<void(dim3, dim3, cudaStream_t, int, int, std::shared_ptr<PrimTransfDataBase>, int*, int2*, int*, float3*, int*)>> dispatcher = {
      { 0, make_cudacall(compute_aabb_kernel<PrimTransfSRT>) }
    };
    
    auto iter = dispatcher.find(min(0, algorithm));
    if (iter != dispatcher.end()) {
        (iter->second)(
            dim3(gridsize), dim3(blocksize), stream,
            N, K,
            primtransf_data,
            sortedobjid,
            reinterpret_cast<int2 *>(nodechildren),
            nodeparent,
            reinterpret_cast<float3 *>(nodeaabb),
            atom);
    }

    cudaFree(atom);
}
