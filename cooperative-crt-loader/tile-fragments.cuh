// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace bytecrt::resident_tile {
constexpr unsigned pairs_per_warp = 1;

// Four consecutive lanes consume a 16-byte row. The last compact
// ranked tile has only the left half, so its load takes two matrices.
template<bool HasUpper,unsigned Stride>
__device__ __forceinline__ const uint32_t* fragment_row(const uint32_t (*rows)[Stride],
        unsigned word, unsigned lane) {
    static_assert(Stride%4==0);
    return rows[lane%16]+word+(HasUpper?4*(lane/16):0);
}

template<bool HasUpper,unsigned Stride>
__device__ __forceinline__ void load_a(const uint32_t (*rows)[Stride],
        unsigned word, unsigned lane, uint32_t (&v)[4]) {
#ifdef __CUDA_ARCH__
    const uint32_t address=static_cast<uint32_t>(
        __cvta_generic_to_shared(fragment_row<HasUpper>(rows,word,lane)));
    if constexpr(HasUpper) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v[0]),"=r"(v[1]),"=r"(v[2]),"=r"(v[3]) : "r"(address) : "memory");
    } else {
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                     : "=r"(v[0]),"=r"(v[1]) : "r"(address) : "memory");
        v[2]=v[3]=0;
    }
#else
    for(unsigned matrix=0;matrix<(HasUpper?4u:2u);++matrix) {
        const unsigned source_lane=8*matrix+lane/4;
        v[matrix]=fragment_row<HasUpper>(rows,word,source_lane)[lane%4];
    }
    if constexpr(!HasUpper) v[2]=v[3]=0;
#endif
}

// The caller supplies the warp MMA primitive so the same fragment loads and
// accumulator mapping can be checked with a synchronous CPU warp model.
template<unsigned P, unsigned Kpad, unsigned UsedWords, unsigned Stride, class Mma>
__device__ __forceinline__ void accumulate(const uint32_t (*a)[Stride],
        const uint8_t* b, unsigned first_pair, unsigned lane,
        int32_t (&c)[pairs_per_warp][8], Mma mma) {
    static_assert(UsedWords==Kpad/4 || UsedWords+4==Kpad/4);
    static_assert(Stride>=UsedWords);
    #pragma unroll
    for (unsigned k = 0; k < Kpad; k += 32) {
        uint32_t low[4],high[4];
        if(k/4+4<UsedWords) {
            load_a<true>(a,k/4,lane,low);
            load_a<true>(a+16,k/4,lane,high);
        } else {
            load_a<false>(a,k/4,lane,low);
            load_a<false>(a+16,k/4,lane,high);
        }
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
