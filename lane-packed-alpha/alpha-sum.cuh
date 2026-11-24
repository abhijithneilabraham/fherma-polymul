// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
namespace bytecrt::warp_alpha {
__device__ __forceinline__ uint32_t fraction(uint32_t term,uint32_t reciprocal) {
#ifdef __CUDA_ARCH__
    return __umulhi(term,reciprocal);
#else
    return uint32_t((uint64_t(term)*reciprocal)>>32);
#endif
}
__device__ __forceinline__ uint32_t pair(uint32_t low,uint32_t high,uint32_t reciprocal) {
    return fraction(low,reciprocal) | (fraction(high,reciprocal)<<16);
}
__device__ __forceinline__ uint32_t round_sum(uint32_t units) {
    return (units+512)>>10;
}
}
