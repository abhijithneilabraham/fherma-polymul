// SPDX-License-Identifier: Apache-2.0
// Derived from ByteCRT's exact unsigned-byte MT32 encoder.
// Accumulator values move directly between paired lanes instead of shared C.
#pragma once
#include "core/rns-encode-mma.cuh"
#include "warp-output.cuh"

namespace bytecrt {
template <unsigned L, unsigned P, unsigned Kpad, bool Transposed = false>
__global__ void encode_mma32_warp(unsigned N,
                            const uint32_t* __restrict__ left,
                            const uint32_t* __restrict__ right,
                            const uint8_t* __restrict__ b,
                            const uint32_t* __restrict__ primes,
                            const uint64_t* __restrict__ mu,
                            uint32_t* __restrict__ ab,
                            unsigned block_begin = 0) {
    static_assert((L == 55 && P == 114 && Kpad == 224) ||
                  (L == 28 && P == 57 && Kpad == 128), "tensor RNS geometry");
    static_assert(4 * L * 255 * 255 < (uint32_t{1} << 31), "exact s32 byte dots");
    constexpr unsigned MT = 32, NT = 56, threads = 224;
    constexpr unsigned groups = (P + 13) / 14;
    constexpr unsigned shared_stride = Kpad + 16;
    constexpr unsigned word_stride = shared_stride / 4;
    // +16 bytes makes each MMA half-row gather bank-distinct at both widths.
    __shared__ __align__(16) uint8_t shared_a[2 * MT][shared_stride];

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31, warp = tid >> 5;
    const unsigned g = lane >> 2, t = lane & 3;
    const unsigned k_lane = 4 * t;
    // Transposed launch: blockIdx.x = tile*256 + id, row r selects
    // coefficient id + (tile*32+r)*256; each row retains contiguous limbs.
    constexpr unsigned transpose_m = 256;
    const unsigned block = block_begin + blockIdx.x;
    const unsigned id = Transposed ? block % transpose_m : 0;
    const unsigned first_row = (Transposed ? block / transpose_m : block) * MT;
    auto* shared_words = reinterpret_cast<uint32_t*>(shared_a);
    for (unsigned word = tid; word < MT * word_stride; word += threads) {
        const unsigned row = first_row + word / word_stride;
        const unsigned coefficient = Transposed ? id + row * transpose_m : row;
        const unsigned limb = word % word_stride;
        const bool valid = coefficient < N && limb < L;
        const size_t source = size_t(coefficient) * L + limb;
        shared_words[word] = valid ? left[source] : 0;
        shared_words[MT * word_stride + word] = valid ? right[source] : 0;
    }
    __syncthreads();

    for (unsigned group = 0; group < groups; ++group) {
        // Warp-uniform: there are no block barriers after the immutable A load.
        if (group * 14 + 2 * warp >= P) continue;
        const unsigned b_col = group * NT + warp * 8 + g;
        int32_t accum[4][4] = {};
        #pragma unroll
        for (unsigned k = 0; k < Kpad; k += 32) {
            uint32_t b0 = 0, b1 = 0;
            if (b_col < 4 * P) {
                const uint8_t* column = b + std::size_t{b_col} * Kpad + k + k_lane;
                b0 = *reinterpret_cast<const uint32_t*>(column);
                b1 = *reinterpret_cast<const uint32_t*>(column + 16);
            }
            #pragma unroll
            for (unsigned tile = 0; tile < 4; ++tile) {
                const unsigned row = 16 * tile + g;
                const auto* top = reinterpret_cast<const uint32_t*>(shared_a[row] + k + k_lane);
                const auto* bottom = reinterpret_cast<const uint32_t*>(shared_a[row+8] + k + k_lane);
                auto& c = accum[tile];
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                    : "r"(top[0]), "r"(bottom[0]), "r"(top[4]), "r"(bottom[4]), "r"(b0), "r"(b1));
            }
        }
        #pragma unroll
        for (unsigned tile = 0; tile < 4; ++tile) {
            const unsigned operand = tile / 2;
            const unsigned quarter = 2 * (tile % 2);
            const auto& c = accum[tile];
            warp_encoding::store_pair<Transposed>(c[0],c[1],lane,warp,group,first_row,id,
                quarter,N,P,operand,primes,mu,ab);
            warp_encoding::store_pair<Transposed>(c[2],c[3],lane,warp,group,first_row,id,
                quarter+1,N,P,operand,primes,mu,ab);
        }
    }
}

inline void launch_encode_mma32_warp_transposed(unsigned N, unsigned L, unsigned P,
        const uint32_t* left, const uint32_t* right, const uint8_t* b, const uint32_t* primes,
        const uint64_t* mu, uint32_t* ab, cudaStream_t stream) {
    const dim3 grid(N / 32);
    if (N == 65536 && L == 55 && P == 114)
        encode_mma32_warp<55, 114, 224, true><<<grid, 224, 0, stream>>>(N, left, right, b, primes, mu, ab);
    else if (N == 32768 && L == 28 && P == 57)
        encode_mma32_warp<28, 57, 128, true><<<grid, 224, 0, stream>>>(N, left, right, b, primes, mu, ab);
    else
        throw std::runtime_error("Warp encoder requires full staged geometries");
}
} // namespace bytecrt
