// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace bytecrt::resident_tile {
constexpr unsigned pairs_per_warp = 1;

// The four matrices provide the low and high K halves of each eight-row
// quadrant. Lane addresses name rows, while the returned words use the
// four-lanes-per-row fragment layout of integer MMA.
__device__ __forceinline__ unsigned fragment_row_word(unsigned stride,unsigned k,unsigned lane) {
    return (lane%16)*stride+k/4+4*(lane/16);
}

__device__ __forceinline__ void load_a(const uint32_t* input,unsigned stride,
                                      unsigned k,unsigned lane,uint32_t (&fragment)[4]) {
#ifdef __CUDA_ARCH__
    const uint32_t* row=input+fragment_row_word(stride,k,lane);
    uint32_t shared=static_cast<uint32_t>(__cvta_generic_to_shared(row));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(fragment[0]),"=r"(fragment[1]),"=r"(fragment[2]),"=r"(fragment[3])
                 : "r"(shared) : "memory");
#else
    for(unsigned matrix=0;matrix<4;++matrix) {
        unsigned source_lane=8*matrix+lane/4;
        fragment[matrix]=input[fragment_row_word(stride,k,source_lane)+lane%4];
    }
#endif
}

// The caller supplies the warp MMA primitive so the same fragment loads and
// accumulator mapping can be checked with a synchronous CPU warp model.
template<unsigned P, unsigned Kpad, class Mma>
__device__ __forceinline__ void accumulate(const uint32_t* a, unsigned stride,
        const uint8_t* b, unsigned first_pair, unsigned lane,
        int32_t (&c)[pairs_per_warp][8], Mma mma) {
    #pragma unroll
    for (unsigned k = 0; k < Kpad; k += 32) {
        uint32_t low[4], high[4];
        load_a(a,stride,k,lane,low);
        load_a(a+16*stride,stride,k,lane,high);
        #pragma unroll
        for (unsigned j=0; j<pairs_per_warp; ++j) {
            unsigned pair=first_pair+j;
            if (2*pair >= P) continue;
            const auto* tile=reinterpret_cast<const uint32_t*>(b)+128*((k/32)*(P/2)+pair);
            mma(c[j],low[0],low[1],low[2],low[3],tile[lane],tile[32+lane]);
            mma(c[j]+4,high[0],high[1],high[2],high[3],tile[64+lane],tile[96+lane]);
        }
    }
}
}
