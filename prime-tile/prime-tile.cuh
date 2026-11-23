// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace bytecrt::prime_tile {
constexpr unsigned pairs_per_warp = 3;

// The caller supplies the warp MMA primitive so the same fragment loads and
// accumulator mapping can be checked with a synchronous CPU warp model.
template<unsigned P, unsigned Kpad, class Mma>
__device__ __forceinline__ void accumulate(const uint32_t* a, unsigned stride,
        const uint8_t* b, unsigned first_pair, unsigned lane,
        int32_t (&c)[pairs_per_warp][8], Mma mma) {
    unsigned g = lane / 4, t = lane % 4;
    #pragma unroll
    for (unsigned k = 0; k < Kpad; k += 32) {
        uint32_t a0 = a[g*stride + k/4 + t];
        uint32_t a1 = a[(g+8)*stride + k/4 + t];
        uint32_t a2 = a[g*stride + k/4 + t+4];
        uint32_t a3 = a[(g+8)*stride + k/4 + t+4];
        uint32_t a4 = a[(g+16)*stride + k/4 + t];
        uint32_t a5 = a[(g+24)*stride + k/4 + t];
        uint32_t a6 = a[(g+16)*stride + k/4 + t+4];
        uint32_t a7 = a[(g+24)*stride + k/4 + t+4];
        #pragma unroll
        for (unsigned j=0; j<pairs_per_warp; ++j) {
            unsigned pair=first_pair+j, column=8*pair+g;
            if (2*pair >= P) continue;
            uint32_t b0=0,b1=0;
            if (column<4*P) {
                const uint32_t* weights=reinterpret_cast<const uint32_t*>(b+column*Kpad+k);
                b0=weights[t]; b1=weights[t+4];
            }
            mma(c[j],a0,a1,a2,a3,b0,b1);
            mma(c[j]+4,a4,a5,a6,a7,b0,b1);
        }
    }
}
}
