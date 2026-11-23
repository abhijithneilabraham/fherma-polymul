// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
namespace bytecrt::warp_alpha {
__device__ __forceinline__ float partial(const uint32_t* terms, const float* inverse,
        unsigned P, unsigned lane, unsigned start=0, unsigned mask=0) {
    float total = 0;
    for (unsigned j = lane; j < P; j += 32)
        total = __fadd_rn(total, __fmul_rn(__uint2float_rn(terms[(start+j)^mask]), inverse[j]));
    return total;
}
__device__ __forceinline__ float sum(const uint32_t* terms, const float* inverse,
        unsigned P, unsigned lane, unsigned start=0, unsigned mask=0) {
    float total = partial(terms, inverse, P, lane, start, mask);
    // All 32 lanes in the row's warp execute every shuffle. For P<=114,
    // four serial terms and five tree levels have a conservative P*gamma_16
    // error < 0.000109, below the centered-CRT rounding margin of 0.25.
    for (unsigned delta = 16; delta; delta >>= 1)
        total = __fadd_rn(total, __shfl_down_sync(0xffffffffu, total, delta));
    return total;
}
} // namespace bytecrt::warp_alpha
