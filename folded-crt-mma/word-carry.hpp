// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define CARRY_INLINE __host__ __device__ __forceinline__
#else
#define CARRY_INLINE inline
#endif

namespace folded_carry {
// One subgroup covers 64 radix-2^32 columns. A lane owns eight consecutive
// columns. Shift each column's high half to its right-hand neighbor first.
struct Carry {
    bool generate;
    bool propagate;
};

CARRY_INLINE Carry normalize_tile(const uint64_t sums[8], uint32_t previous_high,
                                  uint32_t words[8]) {
    uint64_t carry = 0;
    bool propagate = true;
    for (unsigned k = 0; k < 8; ++k) {
        uint64_t value = uint64_t(uint32_t(sums[k])) + previous_high + carry;
        words[k] = uint32_t(value);
        carry = value >> 32;
        previous_high = uint32_t(sums[k] >> 32);
        propagate = propagate && words[k] == UINT32_MAX;
    }
    return {carry != 0, propagate};
}

CARRY_INLINE Carry compose(Carry left, Carry right) {
    return {right.generate || (right.propagate && left.generate),
            right.propagate && left.propagate};
}

CARRY_INLINE void apply_carry(uint32_t words[8], bool incoming) {
    for (unsigned k = 0; k < 8; ++k) {
        uint32_t prior = words[k];
        words[k] += uint32_t(incoming);
        incoming = incoming && prior == UINT32_MAX;
    }
}
}
#undef CARRY_INLINE
