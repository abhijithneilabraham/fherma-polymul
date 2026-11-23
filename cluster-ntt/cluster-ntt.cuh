// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cooperative_groups.h>
#include "odd-fft.cuh"
#include "cluster-layout.hpp"
#include "warp-fft.cuh"

namespace bytecrt::warp_ntt {

template<unsigned N, unsigned P, unsigned Kpad>
__global__ __launch_bounds__(cluster_threads) void cluster_product(
        const uint32_t* input, uint32_t* output, const uint32_t* forward_phase,
        const uint32_t* inverse_phase, const Twiddle* roots, const Mod* mods) {
#if __CUDA_ARCH__ >= 900
    constexpr unsigned K = N / 256, tile_words = K * ids_per_block;
    extern __shared__ uint32_t tile[];
    auto cluster = cooperative_groups::this_cluster();
    unsigned rank = cluster.block_rank(), prime = blockIdx.x / cluster_blocks;
    unsigned warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    unsigned id = ids_per_block * rank + warp;
    const Mod m = mods[prime];
    size_t prime_base = size_t(prime) * N;
    const Twiddle* table = roots + size_t(prime) * (2*K+256);

    // Each warp owns one input id. The shared XOR permutation avoids bank conflicts
    // when its lanes write successive rows of one column.
    for (unsigned operand = 0; operand < 2; ++operand) {
        uint32_t v[K/32];
        #pragma unroll
        for (unsigned r = 0; r < K/32; ++r)
            v[r] = input[size_t(operand)*P*N + prime_base + id*K + lane+32*r];
        odd_fft<K,false>(v, table, m, lane);
        #pragma unroll
        for (unsigned r = 0; r < K/32; ++r) {
            unsigned row = lane+32*r;
            tile[operand*tile_words + cluster_slot(row,warp)] =
                multiply(v[r], forward_phase[prime_base + id*K + row], m);
        }
    }
    cluster.sync();

    // A single warp owns each entire row, including the later remote writes.
    // All its remote addresses are contiguous across lanes.
    if (warp < K/cluster_blocks) {
        unsigned row = rank*(K/cluster_blocks) + warp;
        unsigned column = cluster_slot(row,lane);
        uint32_t a[8], b[8];
        #pragma unroll
        for (unsigned peer = 0; peer < cluster_blocks; ++peer) {
            uint32_t* remote = cluster.map_shared_rank(tile, peer);
            a[peer] = remote[column];
            b[peer] = remote[tile_words+column];
        }
        fft<256,false>(a, table+2*K, m, lane);
        fft<256,false>(b, table+2*K, m, lane);
        #pragma unroll
        for (unsigned j = 0; j < 8; ++j) a[j] = multiply(a[j], b[j], m);
        fft<256,true>(a, table+2*K, m, lane);
        #pragma unroll
        for (unsigned peer = 0; peer < cluster_blocks; ++peer) {
            uint32_t* remote = cluster.map_shared_rank(tile, peer);
            remote[column] = multiply(a[peer], inverse_phase[prime_base + row*256 + 32*peer + lane], m);
        }
    }
    cluster.sync();
    uint32_t v[K/32];
    #pragma unroll
    for (unsigned r = 0; r < K/32; ++r)
        v[r] = tile[cluster_slot(lane+32*r,warp)];
    odd_fft<K,true>(v, table+K, m, lane);
    #pragma unroll
    for (unsigned r = 0; r < K/32; ++r)
        output[(size_t(id)*(Kpad/4)+prime)*K + lane+32*r] = v[r];
#else
    asm volatile("trap;");
#endif
}

template<unsigned N,unsigned P>
cudaLaunchConfig_t cluster_config(cudaStream_t stream, cudaLaunchAttribute& attribute) {
    attribute = {};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim.x = cluster_blocks;
    attribute.val.clusterDim.y = 1;
    attribute.val.clusterDim.z = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(P * cluster_blocks);
    config.blockDim = dim3(cluster_threads);
    config.dynamicSmemBytes = 2 * (N/256) * ids_per_block * sizeof(uint32_t);
    config.stream = stream;
    config.attrs = &attribute;
    config.numAttrs = 1;
    return config;
}
}
