// SPDX-License-Identifier: Apache-2.0
// Exact residue from a bounded FP32 quotient estimate and integer corrections.
#pragma once
#include <cstdint>
namespace bytecrt::fast_encoding {
__device__ __forceinline__ uint32_t reduce(const int32_t* sums, uint32_t p, float inverse) {
    const uint64_t x = uint64_t{static_cast<uint32_t>(sums[0])} +
                       (uint64_t{static_cast<uint32_t>(sums[1])} << 8) +
                       (uint64_t{static_cast<uint32_t>(sums[2])} << 16) +
                       (uint64_t{static_cast<uint32_t>(sums[3])} << 24);
    // x/p < 4*55*255 = 56100. Four conservative RN error factors give
    // absolute quotient error < 0.013376, so floor(q_est) differs by <= 1.
    const uint32_t quotient = __float2uint_rz(__fmul_rn(__ull2float_rn(x), inverse));
    int64_t remainder = static_cast<int64_t>(x) - static_cast<int64_t>(quotient) * p;
    if (remainder < 0) remainder += p;
    if (remainder >= p) remainder -= p;
    return static_cast<uint32_t>(remainder);
}
} // namespace bytecrt::fast_encoding
