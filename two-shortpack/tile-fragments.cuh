// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace bytecrt::resident_tile {
constexpr unsigned pairs_per_warp = 1;

// The caller supplies the warp MMA primitive so the same fragment loads and
// accumulator mapping can be checked with a synchronous CPU warp model.
template<unsigned P, unsigned Kpad, unsigned UsedWords=Kpad/4, class Mma>
__device__ __forceinline__ void accumulate(const uint32_t* a, unsigned stride,
        const uint8_t* b, unsigned first_pair, unsigned lane,
        int32_t (&c)[pairs_per_warp][8], Mma mma) {
    static_assert(UsedWords==Kpad/4 || UsedWords+4==Kpad/4);
    unsigned g = lane / 4, t = lane % 4;
    #pragma unroll
    for (unsigned k = 0; k < Kpad; k += 32) {
        uint32_t a0 = a[g*stride + k/4 + t];
        uint32_t a1 = a[(g+8)*stride + k/4 + t];
        bool has_upper_words=k/4+4<UsedWords;
        uint32_t a2 = has_upper_words?a[g*stride + k/4 + t+4]:0;
        uint32_t a3 = has_upper_words?a[(g+8)*stride + k/4 + t+4]:0;
        uint32_t a4 = a[(g+16)*stride + k/4 + t];
        uint32_t a5 = a[(g+24)*stride + k/4 + t];
        uint32_t a6 = has_upper_words?a[(g+16)*stride + k/4 + t+4]:0;
        uint32_t a7 = has_upper_words?a[(g+24)*stride + k/4 + t+4]:0;
        #pragma unroll
        for (unsigned j=0; j<pairs_per_warp; ++j) {
            unsigned pair=first_pair+j;
            if (2*pair >= P) continue;
            const auto* tile=reinterpret_cast<const uint32_t*>(b)+128*((k/32)*(P/2)+pair);
            mma(c[j],a0,a1,a2,a3,tile[lane],tile[32+lane]);
            mma(c[j]+4,a4,a5,a6,a7,tile[64+lane],tile[96+lane]);
        }
    }
}
}
