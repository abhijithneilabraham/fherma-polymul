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

// Each row includes one zero sentinel after its final input limb.
template<unsigned W>
SEVEN_INLINE U digit_word(const U* row,unsigned digit,unsigned word) {
    unsigned lo=(digit*W+6)/7, hi=((digit+1)*W+6)/7;
    unsigned bit=lo+32*word;
    if(bit>=hi) return 0;
    unsigned shift=bit%32;
    U value=row[bit/32]>>shift;
    if(shift) value|=row[bit/32+1]<<(32-shift);
    unsigned remaining=hi-bit;
    return remaining<32 ? value & ((U{1}<<remaining)-1) : value;
}

// Each warp handles 16 coefficients and two primes. An even lane owns two
// coefficients for one prime. Its odd neighbor contributes two weight bytes.
template<unsigned W,unsigned L,class Operations>
SEVEN_INLINE void evaluate(const U (&raw)[16][L+1],unsigned lane,const Field& field,
                           U weight_lo,U weight_hi,U (&result)[2][7], Operations ops) {
    unsigned row=lane/4, t=lane%4;
    U digits[2][7];
    #pragma unroll
    for(unsigned d=0;d<7;++d) {
        int32_t acc[4]={};
        ops.mma(acc,digit_word<W>(raw[row],d,t),digit_word<W>(raw[row+8],d,t),
                    digit_word<W>(raw[row],d,t+4),digit_word<W>(raw[row+8],d,t+4),
                    weight_lo,weight_hi);
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
