// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "warp-mod.hpp"
namespace bytecrt::warp_ntt {
struct TensorRow {
    // Twenty words per matrix row and four extra words per warp keep both
    // the tensor loads and the later cross-warp transpose bank-aligned.
    uint32_t word[16*20+4];
    __device__ __forceinline__ uint32_t& at(unsigned row,unsigned col) {
        return word[20*row+col];
    }
};

// Each warp owns 16 independent length-16 vectors. Two m16n8k32
// instructions compute the four byte sums for two output frequencies.
template<bool First,class WarpOps>
__device__ __forceinline__ void radix16(TensorRow& s, const uint32_t* weights,
        const Twiddle* roots, Mod m, uint32_t reciprocal, unsigned lane,WarpOps ops) {
    unsigned g=lane/4,t=lane%4;
    uint32_t a[2][4];
    #pragma unroll
    for(unsigned half=0;half<2;++half) {
        a[half][0]=s.at(g,8*half+t);
        a[half][1]=s.at(g+8,8*half+t);
        a[half][2]=s.at(g,8*half+t+4);
        a[half][3]=s.at(g+8,8*half+t+4);
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
                if constexpr (First) s.at(col,row)=multiply_twiddle(v,roots[row*col],m);
                else s.at(row,col)=v;
            }
        }
    }
    ops.sync();
}

// The forward matrix also performs the inverse when its output is reversed.
// Read all permuted values before reusing the same tile as linear storage.
template<class WarpOps>
__device__ __forceinline__ void linearize_inverse(TensorRow& s,
        const uint32_t* phases,const Mod& mod,unsigned lane,WarpOps ops) {
    uint32_t values[8];
    #pragma unroll
    for(unsigned j=0;j<8;++j) {
        unsigned i=lane+32*j,reverse=(256-i)%256;
        values[j]=multiply(s.at(reverse%16,reverse/16),phases[i],mod);
    }
    ops.sync();
    #pragma unroll
    for(unsigned j=0;j<8;++j)s.word[lane+32*j]=values[j];
}

}
