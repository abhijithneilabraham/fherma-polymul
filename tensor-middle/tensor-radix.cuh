// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "warp-mod.hpp"
namespace bytecrt::warp_ntt {
struct TensorRow { uint32_t word[16][20]; };

// Each warp owns 16 independent length-16 vectors. Two m16n8k32
// instructions compute the four byte sums for two output frequencies.
template<bool First,class WarpOps>
__device__ __forceinline__ void radix16(TensorRow& s, const uint32_t* weights,
        const Twiddle* roots, Mod m, uint64_t reciprocal, unsigned lane,WarpOps ops) {
    unsigned g=lane/4,t=lane%4;
    uint32_t a[2][4];
    #pragma unroll
    for(unsigned half=0;half<2;++half) {
        a[half][0]=s.word[g][8*half+t];
        a[half][1]=s.word[g+8][8*half+t];
        a[half][2]=s.word[g][8*half+t+4];
        a[half][3]=s.word[g+8][8*half+t+4];
    }
    ops.sync();
    #pragma unroll 1
    for(unsigned pair=0;pair<8;++pair) {
        int32_t c[4]={};
        #pragma unroll
        for(unsigned half=0;half<2;++half) {
            const uint32_t* b=weights+(8*pair+g)*20+8*half+t;
            ops.mma(c,a[half],b);
        }
        #pragma unroll
        for(unsigned part=0;part<2;++part) {
            int32_t d0=c[2*part],d1=c[2*part+1];
            int32_t d2=ops.neighbor(d0);
            int32_t d3=ops.neighbor(d1);
            if (!(lane&1)) {
                unsigned row=g+8*part,col=2*pair+t/2;
                int32_t dots[4]={d0,d1,d2,d3};
                uint32_t v=ops.reduce(dots,m.p,reciprocal);
                if constexpr (First) s.word[col][row]=multiply_twiddle(v,roots[row*col],m);
                else s.word[row][col]=v;
            }
        }
    }
    ops.sync();
}

}
