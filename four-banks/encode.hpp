// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "tile-layout.hpp"

namespace four_encode {
using namespace radix4;
struct Field { Mod mod; uint64_t reciprocal; Twiddle forward; };

template<unsigned W>
FOUR_INLINE U digit_word(const U* raw,unsigned row,unsigned digit,unsigned word) {
    unsigned lo=(digit*W+3)/4,hi=((digit+1)*W+3)/4;
    unsigned abs_word=lo/32+word;
    if(32*abs_word>=hi)return 0;
    U x=raw[raw_limb<W>(row,abs_word)];
    if(word==0)x&=UINT32_MAX<<(lo%32);
    if(hi-32*abs_word<32)x&=(U{1}<<(hi%32))-1;
    return x;
}

template<unsigned W,unsigned L,unsigned Count,class Operations>
FOUR_INLINE void evaluate(const U (&raw)[16][raw_stride<W>],unsigned lane,const Field& field,
                          const U* weights,U (&result)[2][4],Operations ops) {
    constexpr unsigned chunks=W==1747?2:1;
    U digits[2][4];
    unsigned row=lane/4,t=lane%4;
    #pragma unroll
    for(unsigned d=0;d<4;++d){
        int32_t acc[4]={};
        #pragma unroll
        for(unsigned c=0;c<chunks;++c){
            unsigned word=8*c+t;
            const U* b=weights+size_t(d*chunks+c)*(Count/2)*64;
            ops.mma(acc,digit_word<W>(raw[row],row,d,word),digit_word<W>(raw[row+8],row+8,d,word),
                    digit_word<W>(raw[row],row,d,word+4),digit_word<W>(raw[row+8],row+8,d,word+4),
                    b[lane],b[32+lane]);
        }
        #pragma unroll
        for(unsigned h=0;h<2;++h){
            int32_t sums[4]={acc[2*h],acc[2*h+1],ops.neighbor(acc[2*h]),ops.neighbor(acc[2*h+1])};
            if(!(lane&1))digits[h][d]=ops.reduce(sums,field);
        }
    }
    if(!(lane&1)){
        transform(digits[0],field.forward,field.mod,result[0]);
        transform(digits[1],field.forward,field.mod,result[1]);
    }
}
}
