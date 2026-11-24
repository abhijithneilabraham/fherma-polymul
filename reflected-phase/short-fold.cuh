// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace bytecrt::short_fold {
// Input is below 2^(W+38) and q = 2^W - gap, with gap a nonzero u32.
// Store the canonical W-bit remainder back in little-endian limbs.
template<unsigned W, unsigned L>
__device__ __forceinline__ void reduce(uint32_t* digits, uint32_t gap) {
    constexpr unsigned top_bits = W - 32*(L-1);
    static_assert(W > 70 && top_bits > 0 && top_bits < 32);
    constexpr uint32_t top_mask = (uint32_t{1} << top_bits) - 1;
    const uint64_t high = (uint64_t{digits[L-1]} >> top_bits) |
                          (uint64_t{digits[L]} << (32-top_bits)) |
                          (uint64_t{digits[L+1]} << (64-top_bits));
    digits[L-1] &= top_mask;
    digits[L] = digits[L+1] = 0;
    if (high) {
        const uint64_t low_product = uint64_t{uint32_t(high)} * gap;
        const uint64_t upper_product = (high >> 32)*gap + (low_product >> 32);
        uint64_t sum = uint64_t{digits[0]} + uint32_t(low_product);
        digits[0] = uint32_t(sum);
        sum = uint64_t{digits[1]} + uint32_t(upper_product) + (sum >> 32);
        digits[1] = uint32_t(sum);
        sum = uint64_t{digits[2]} + (upper_product >> 32) + (sum >> 32);
        digits[2] = uint32_t(sum);
        for (unsigned i=3; (sum >> 32) && i<L; ++i) {
            sum = uint64_t{digits[i]} + 1;
            digits[i] = uint32_t(sum);
        }
        // The first fold is below 2^W + 2^70. If it crosses 2^W,
        // the low part plus gap fits in three limbs.
        if (digits[L-1] > top_mask) {
            digits[L-1] &= top_mask;
            sum = uint64_t{digits[0]} + gap;
            digits[0] = uint32_t(sum);
            sum = uint64_t{digits[1]} + (sum >> 32);
            digits[1] = uint32_t(sum);
            digits[2] += uint32_t(sum >> 32);
            return;
        }
    }
    // Only gap values in the entire W-bit range still need subtraction.
    const uint32_t low_q = uint32_t(0) - gap;
    if (digits[L-1] != top_mask || digits[0] < low_q) return;
    for (unsigned i=1; i<L-1; ++i)
        if (digits[i] != UINT32_MAX) return;
    digits[0] -= low_q;
    for (unsigned i=1; i<L; ++i) digits[i] = 0;
}
}
