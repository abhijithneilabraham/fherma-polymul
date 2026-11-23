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

template<unsigned W>
constexpr unsigned digit_words=(W+7*32-1)/(7*32);

template<unsigned Count>
inline void set_weights(uint32_t* matrix,unsigned digit,unsigned prime,uint32_t scale,uint32_t p) {
    uint32_t weight=scale;
    uint32_t* pair=matrix+digit*Count*32+(prime/2)*64;
    for(unsigned byte=0;byte<32;++byte) {
        for(unsigned out=0;out<4;++out) {
            unsigned slot=32*(byte/16)+4*(4*(prime%2)+out)+(byte%16)/4;
            pair[slot]|=((weight>>(8*out))&255u)<<(8*(byte%4));
        }
        weight=uint64_t(weight)*256%p;
    }
}

// Each warp handles 16 coefficients and two primes. An even lane owns two
// coefficients for one prime. Its odd neighbor contributes two weight bytes.
template<unsigned W,unsigned Count,class Operations>
SEVEN_INLINE void evaluate(const U (&packed)[16][7*digit_words<W>],unsigned lane,const Field& field,
                           const U* weights,U (&result)[2][7], Operations ops) {
    unsigned row=lane/4, t=lane%4;
    U digits[2][7];
    #pragma unroll
    for(unsigned d=0;d<7;++d) {
        int32_t acc[4]={};
        constexpr unsigned words=digit_words<W>;
        const unsigned word=d*words+t;
        const U* weight=weights+d*Count*32;
        ops.mma(acc,packed[row][word],packed[row+8][word],
                words==8?packed[row][word+4]:0,words==8?packed[row+8][word+4]:0,
                weight[lane],weight[32+lane]);
        #pragma unroll
        for(unsigned h=0;h<2;++h) {
            int32_t sums[4]={acc[2*h],acc[2*h+1],ops.neighbor(acc[2*h]),ops.neighbor(acc[2*h+1])};
            if(!(lane&1))
                digits[h][d]=ops.reduce(sums,field);
        }
    }
    if(!(lane&1)) {
        transform(digits[0],field.forward,field.mod,result[0]);
        transform(digits[1],field.forward,field.mod,result[1]);
    }
}
#undef SEVEN_INLINE
}
