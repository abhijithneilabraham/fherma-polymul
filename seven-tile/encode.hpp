// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "rader7.hpp"

namespace seven_encode {
using namespace rader7;
struct Field {
    Mod mod;
    uint64_t reciprocal;
    Twiddle digit_twist[7];
    Weights forward;
};
#ifdef __CUDACC__
#define SEVEN_INLINE __device__ __forceinline__
#else
#define SEVEN_INLINE inline
#endif

// Input rows may end at the final limb of the allocation.
template<unsigned W>
SEVEN_INLINE U digit_word(const U* row,unsigned digit,unsigned word) {
    unsigned lo=(digit*W+6)/7, hi=((digit+1)*W+6)/7;
    unsigned bit=lo+32*word;
    if(bit>=hi) return 0;
    unsigned shift=bit%32;
    U value=row[bit/32]>>shift;
    if(shift && bit/32+1<(W+31)/32) value|=row[bit/32+1]<<(32-shift);
    unsigned remaining=hi-bit;
    return remaining<32 ? value & ((U{1}<<remaining)-1) : value;
}

struct Fragments { U a[7][4]; };
template<unsigned W>
SEVEN_INLINE Fragments load_fragments(const U* low,const U* high,unsigned t) {
    Fragments f;
    #pragma unroll
    for(unsigned d=0;d<7;++d) {
        f.a[d][0]=digit_word<W>(low,d,t);
        f.a[d][1]=digit_word<W>(high,d,t);
        f.a[d][2]=digit_word<W>(low,d,t+4);
        f.a[d][3]=digit_word<W>(high,d,t+4);
    }
    return f;
}
template<class Operations>
SEVEN_INLINE void evaluate(const Fragments& f,unsigned lane,const Field& field,
                           U weight_lo,U weight_hi,U (&result)[2][7], Operations ops) {
    U digits[2][7];
    #pragma unroll
    for(unsigned d=0;d<7;++d) {
        int32_t acc[4]={};
        ops.mma(acc,f.a[d][0],f.a[d][1],f.a[d][2],f.a[d][3],weight_lo,weight_hi);
        #pragma unroll
        for(unsigned h=0;h<2;++h) {
            int32_t sums[4]={acc[2*h],acc[2*h+1],ops.neighbor(acc[2*h]),ops.neighbor(acc[2*h+1])};
            if(!(lane&1))
                digits[h][d]=multiply_twiddle(ops.reduce(sums,field),field.digit_twist[d],field.mod);
        }
    }
    if(!(lane&1)) {
        transform(digits[0],field.forward,field.mod,result[0]);
        transform(digits[1],field.forward,field.mod,result[1]);
    }
}
#undef SEVEN_INLINE
}
