// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace bytecrt::warp_carry {
template<unsigned L,unsigned Rows,unsigned Stride>
__host__ __device__ inline void store_dot(uint32_t (&lo)[Rows][Stride],uint32_t (&hi)[Rows][Stride],
        unsigned lane,unsigned warp,unsigned group,unsigned part,uint32_t a,uint32_t b,uint32_t c,uint32_t d) {
    unsigned limb=14*group+2*warp+(lane%4)/2;
    if(!(lane&1) && limb<L) {
        unsigned row=lane/4+8*part;
        uint64_t sum=uint64_t(a)+(uint64_t(b)<<8)+(uint64_t(c)<<16)+(uint64_t(d)<<24);
        lo[row][limb]=uint32_t(sum);hi[row][limb]=uint32_t(sum>>32);
    }
}
struct CarryMask { uint32_t incoming, outgoing; };
// g marks a word whose sum overflowed, p marks a sum of UINT32_MAX.
// Adding (g|p) to g reproduces that generate/propagate recurrence in
// all 32 bit positions. XOR with p gives every word's incoming carry.
__host__ __device__ inline CarryMask carries(uint32_t g,uint32_t p,uint32_t first) {
    uint32_t a=g|p, sum=a+g, full=sum+first;
    return {full^p,uint32_t(sum<a || full<sum)};
}
template<class Warp>
__device__ __forceinline__ uint32_t normalize(uint32_t a,uint32_t b,uint32_t first,
        unsigned lane,Warp op,uint32_t& outgoing) {
    uint32_t sum=a+b;
    uint32_t g=op.ballot(sum<a),p=op.ballot(sum==UINT32_MAX);
    CarryMask c=carries(g,p,first);
    outgoing=c.outgoing;
    return sum+((c.incoming>>lane)&1u);
}
template<unsigned W,unsigned L,class Warp>
__device__ __forceinline__ void finish(const uint32_t* lows,const uint32_t* highs,
        uint32_t gap,uint32_t* out,unsigned lane,Warp op) {
    static_assert(L==28 || L==55);
    constexpr unsigned top=W%32;
    constexpr uint32_t mask=(uint32_t{1}<<top)-1;
    uint32_t v[2],carry=0;
    #pragma unroll
    for(unsigned tile=0;tile<2;++tile) {
        unsigned j=tile*32+lane;
        uint32_t a=j<L?lows[j]:0;
        uint32_t b=j>0 && j<=L?highs[j-1]:0;
        v[tile]=normalize(a,b,carry,lane,op,carry);
    }
    #pragma unroll
    for(unsigned pass=0;pass<2;++pass) {
        const uint32_t tail=op.get(v[(L-1)/32],(L-1)%32);
        const uint32_t extra=op.get(v[L/32],L%32);
        const uint32_t last=op.get(v[(L+1)/32],(L+1)%32);
        const uint64_t high=(uint64_t(tail)>>top)|(uint64_t(extra)<<(32-top))|(uint64_t(last)<<(64-top));
        if(!high) break; // The three broadcasts give every lane the same decision.
        const uint64_t low_product=uint64_t(uint32_t(high))*gap;
        const uint64_t upper=(high>>32)*gap+(low_product>>32);
        carry=0;
        #pragma unroll
        for(unsigned tile=0;tile<2;++tile) {
            unsigned j=tile*32+lane;
            uint32_t addend=j==0?uint32_t(low_product):j==1?uint32_t(upper):j==2?uint32_t(upper>>32):0;
            uint32_t low=j<L-1?v[tile]:j==L-1?v[tile]&mask:0;
            v[tile]=normalize(low,addend,carry,lane,op,carry);
        }
    }
    bool meets=true;
    #pragma unroll
    for(unsigned tile=0;tile<2;++tile) {
        unsigned j=32*tile+lane;
        if(j<L) meets &= j==0 ? v[tile]>=uint32_t(0)-gap : v[tile]==(j==L-1?mask:UINT32_MAX);
    }
    const bool subtract=op.ballot(meets)==UINT32_MAX;
    #pragma unroll
    for(unsigned tile=0;tile<2;++tile) {
        unsigned j=tile*32+lane;
        if(j<L) out[j]=subtract?(j==0?v[tile]+gap:0):v[tile];
    }
}
}
