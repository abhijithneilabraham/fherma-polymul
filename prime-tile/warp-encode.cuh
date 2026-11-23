// SPDX-License-Identifier: Apache-2.0
// Derived from ByteCRT's exact unsigned-byte MT32 encoder.
// Accumulator values move directly between paired lanes instead of shared C.
#pragma once
#include "core/rns-encode-mma.cuh"
#include "warp-output.cuh"
#include "prime-tile.cuh"

namespace bytecrt {
struct EncodeMma {
    __device__ __forceinline__ void operator()(int32_t* c,uint32_t a0,uint32_t a1,
            uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1) const {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
            : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
};

template <unsigned L, unsigned P, unsigned Kpad, bool Transposed = false>
__global__ void encode_mma32_warp(unsigned N,
                            const uint32_t* __restrict__ inputs,
                            const uint8_t* __restrict__ b,
                            const uint32_t* __restrict__ primes,
                            const uint64_t* __restrict__ mu,
                            uint32_t* __restrict__ ab,
                            unsigned block_begin = 0) {
    static_assert((L == 55 && P == 114 && Kpad == 224) ||
                  (L == 28 && P == 57 && Kpad == 128), "tensor RNS geometry");
    static_assert(4 * L * 255 * 255 < (uint32_t{1} << 31), "exact s32 byte dots");
    constexpr unsigned MT = 32, threads = 224;
    constexpr unsigned shared_stride = Kpad + 16;
    constexpr unsigned word_stride = shared_stride / 4;
    // +16 bytes makes each MMA half-row gather bank-distinct at both widths.
    __shared__ __align__(16) uint8_t shared_a[MT][shared_stride];

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31, warp = tid >> 5;
    // Transposed launch: blockIdx.x = tile*256 + id, row r selects
    // coefficient id + (tile*32+r)*256; each row retains contiguous limbs.
    constexpr unsigned transpose_m = 256;
    const unsigned block = block_begin + blockIdx.x;
    const unsigned id = Transposed ? block % transpose_m : 0;
    const unsigned first_row = (Transposed ? block / transpose_m : block) * MT;
    const unsigned operand = blockIdx.y;
    const std::size_t input_base = std::size_t{operand} * N * L;
    auto* shared_words = reinterpret_cast<uint32_t*>(shared_a);
    for (unsigned word = tid; word < MT * word_stride; word += threads) {
        const unsigned row = first_row + word / word_stride;
        const unsigned coefficient = Transposed ? id + row * transpose_m : row;
        const unsigned limb = word % word_stride;
        shared_words[word] = coefficient < N && limb < L
            ? inputs[input_base + std::size_t{coefficient} * L + limb]
            : 0;
    }
    __syncthreads();

    constexpr unsigned pairs = (P+1)/2;
    for (unsigned first = prime_tile::pairs_per_warp*warp; first < pairs;
         first += prime_tile::pairs_per_warp*(threads/32)) {
        int32_t sums[prime_tile::pairs_per_warp][8] = {};
        prime_tile::accumulate<P,Kpad>(shared_words,word_stride,b,first,lane,sums,EncodeMma{});
        #pragma unroll
        for (unsigned j=0; j<prime_tile::pairs_per_warp; ++j) {
            if (first+j >= pairs) continue;
            #pragma unroll
            for (unsigned q=0; q<4; ++q)
                warp_encoding::store_pair<Transposed>(sums[j][2*q],sums[j][2*q+1],
                    lane,first+j,0,first_row,id,q,N,P,operand,primes,mu,ab);
        }
    }
}

inline void launch_encode_mma32_warp_transposed(unsigned N, unsigned L, unsigned P,
        const uint32_t* inputs, const uint8_t* b, const uint32_t* primes,
        const uint64_t* mu, uint32_t* ab, cudaStream_t stream, unsigned operands = 2) {
    const dim3 grid(N / 32, operands);
    if (N == 65536 && L == 55 && P == 114)
        encode_mma32_warp<55, 114, 224, true><<<grid, 224, 0, stream>>>(N, inputs, b, primes, mu, ab);
    else if (N == 32768 && L == 28 && P == 57)
        encode_mma32_warp<28, 57, 128, true><<<grid, 224, 0, stream>>>(N, inputs, b, primes, mu, ab);
    else
        throw std::runtime_error("Warp encoder requires full staged geometries");
}
} // namespace bytecrt
