// SPDX-License-Identifier: MIT
#pragma once
#include "word-carry.hpp"
#include "folded-math.hpp"

namespace folded_carry {
__device__ __forceinline__ void normalize_group(const uint64_t sums[8], uint32_t words[8], unsigned lane) {
    constexpr unsigned mask = 0xffffffffu;
    uint32_t preceding = __shfl_up_sync(mask, uint32_t(sums[7] >> 32), 1, 8);
    Carry value = normalize_tile(sums, lane ? preceding : 0, words);
    #pragma unroll
    for (unsigned distance = 1; distance < 8; distance *= 2) {
        bool gen = __shfl_up_sync(mask, unsigned(value.generate), distance, 8);
        bool prop = __shfl_up_sync(mask, unsigned(value.propagate), distance, 8);
        if (lane >= distance) value = compose({gen, prop}, value);
    }
    bool incoming = __shfl_up_sync(mask, unsigned(value.generate), 1, 8);
    apply_carry(words, lane && incoming);
}

// All 256 threads call this function. scratch is a distinct 64-word shared
// row for each subgroup. A block barrier separates each read and write phase.
__device__ __forceinline__ void assemble_group(const uint32_t digits[8][16],
        const folded::Plan& plan, unsigned lane, uint32_t scratch[64], uint32_t* output) {
    constexpr uint32_t mask = (1u << 19) - 1;
    uint64_t sums[8];
    uint32_t words[8];
    #pragma unroll
    for (unsigned k = 0; k < 8; ++k) {
        unsigned column = 8*lane+k;
        uint64_t sum = column < folded::L ? plan.bias_correction[column] : 0;
        #pragma unroll
        for (unsigned d = 0; d < 8; ++d) {
            unsigned position = folded::position(d), shift = position % 32;
            int i = int(column) - int(position / 32);
            if (i >= 0 && i < 16) sum += uint32_t(digits[d][i] << shift);
            if (shift && i > 0 && i <= 16) sum += digits[d][i-1] >> (32-shift);
        }
        sums[k] = sum;
    }
    normalize_group(sums, words, lane);
    #pragma unroll
    for (unsigned pass = 0; pass < 2; ++pass) {
        #pragma unroll
        for (unsigned k = 0; k < 8; ++k) scratch[8*lane+k] = words[k];
        __syncthreads();
        #pragma unroll
        for (unsigned k = 0; k < 8; ++k) {
            unsigned j = 8*lane+k;
            uint32_t high = j < 10 ? scratch[54+j] >> 19 : 0;
            if (j < 9) high |= scratch[55+j] << 13;
            uint32_t low = j < 54 ? words[k] : j == 54 ? words[k] & mask : 0;
            sums[k] = low + uint64_t(high) * folded::gap;
        }
        __syncthreads();
        normalize_group(sums, words, lane);
    }
    // x >= q iff x + gap reaches bit W. All eight lanes use that one decision.
    #pragma unroll
    for (unsigned k = 0; k < 8; ++k)
        sums[k] = uint64_t(words[k]) + ((lane == 0 && k == 0) ? folded::gap : 0);
    uint32_t trial[8];
    normalize_group(sums, trial, lane);
    bool use_trial = __shfl_sync(0xffffffffu, unsigned(trial[6] >> 19), 6, 8);
    #pragma unroll
    for (unsigned k = 0; k < 8; ++k) {
        unsigned j = 8*lane+k;
        if (j < folded::L) {
            uint32_t selected = use_trial ? trial[k] : words[k];
            output[j] = j == 54 ? selected & mask : selected;
        }
    }
}
}
