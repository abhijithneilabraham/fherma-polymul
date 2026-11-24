// SPDX-License-Identifier: Apache-2.0
// Derived from ByteCRT's exact unsigned-byte MT32 encoder.
// Accumulator values move directly between paired lanes instead of shared C.
#pragma once
#include "core/rns-encode-mma.cuh"
#include "warp-output.cuh"

namespace bytecrt {
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
    constexpr unsigned MT = 32, NT = 56, threads = 224;
    constexpr unsigned groups = (P + 13) / 14;
    constexpr unsigned shared_stride = Kpad + 16;
    constexpr unsigned word_stride = shared_stride / 4;
    // +16 bytes makes each MMA half-row gather bank-distinct at both widths.
    __shared__ __align__(16) uint8_t shared_a[MT][shared_stride];

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

    for (unsigned group = 0; group < groups; ++group) {
        // Warp-uniform: there are no block barriers after the immutable A load.
        if (group * 14 + 2 * warp >= P) continue;
        const unsigned b_col = group * NT + warp * 8 + g;
        int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        int32_t c4 = 0, c5 = 0, c6 = 0, c7 = 0;
        #pragma unroll
        for (unsigned k = 0; k < Kpad; k += 32) {
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(shared_a[g] + k + k_lane);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(shared_a[g + 8] + k + k_lane);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(shared_a[g] + k + k_lane + 16);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(shared_a[g + 8] + k + k_lane + 16);
            uint32_t b0 = 0, b1 = 0;
            if (b_col < 4 * P) {
                const uint8_t* column = b + std::size_t{b_col} * Kpad + k + k_lane;
                b0 = *reinterpret_cast<const uint32_t*>(column);
                b1 = *reinterpret_cast<const uint32_t*>(column + 16);
            }
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            const uint32_t a4 = *reinterpret_cast<const uint32_t*>(shared_a[g + 16] + k + k_lane);
            const uint32_t a5 = *reinterpret_cast<const uint32_t*>(shared_a[g + 24] + k + k_lane);
            const uint32_t a6 = *reinterpret_cast<const uint32_t*>(shared_a[g + 16] + k + k_lane + 16);
            const uint32_t a7 = *reinterpret_cast<const uint32_t*>(shared_a[g + 24] + k + k_lane + 16);
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(c4), "+r"(c5), "+r"(c6), "+r"(c7)
                : "r"(a4), "r"(a5), "r"(a6), "r"(a7), "r"(b0), "r"(b1));
        }
        warp_encoding::store_pair<Transposed>(c0, c1, lane, warp, group, first_row, id,
            0, N, P, operand, primes, mu, ab);
        warp_encoding::store_pair<Transposed>(c2, c3, lane, warp, group, first_row, id,
            1, N, P, operand, primes, mu, ab);
        warp_encoding::store_pair<Transposed>(c4, c5, lane, warp, group, first_row, id,
            2, N, P, operand, primes, mu, ab);
        warp_encoding::store_pair<Transposed>(c6, c7, lane, warp, group, first_row, id,
            3, N, P, operand, primes, mu, ab);
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
