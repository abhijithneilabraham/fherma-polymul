// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace two_digit {
#ifdef __CUDACC__
#define WORD_INLINE __host__ __device__ __forceinline__
#else
#define WORD_INLINE inline
#endif
WORD_INLINE uint32_t reduce_dot(const int32_t* sums,uint32_t p,uint32_t reciprocal48) {
    const uint32_t bottom=uint32_t(sums[0])+(uint32_t(sums[1])<<8);
    const uint32_t high16=uint32_t(sums[2])+(uint32_t(sums[3])<<8)+(bottom>>16);
    const uint64_t dot=(uint64_t{high16}<<16)|(bottom&65535u);
#ifdef __CUDA_ARCH__
    const uint32_t quotient=__umulhi(high16,reciprocal48);
#else
    const uint32_t quotient=uint32_t((uint64_t(high16)*reciprocal48)>>32);
#endif
    const uint64_t residue=dot-uint64_t{quotient}*p;
    return uint32_t(residue>=p?residue-p:residue);
}
#undef WORD_INLINE
}
