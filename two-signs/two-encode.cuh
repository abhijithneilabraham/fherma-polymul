// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "sign-pair.hpp"
#include "tile-fragments.cuh"
#include "two-digit-word.hpp"
#include "warp-mod.hpp"

namespace two_digit {
struct Mma {
    __device__ __forceinline__ void operator()(int32_t* c,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1) const {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
            : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
};
struct Neighbor {
    __device__ __forceinline__ uint32_t operator()(uint32_t value) const {
        return __shfl_xor_sync(0xffffffffu,value,1);
    }
};
template<unsigned N,unsigned W,unsigned L,unsigned Count,unsigned Pad,unsigned Split>
__global__ void encode(const uint32_t* input,uint32_t* planes,const uint8_t* weights,
        const uint32_t* primes,const uint32_t* reciprocal) {
    constexpr unsigned stride=Pad/4+4, K=N/256;
    __shared__ uint32_t digits[32][stride];
    unsigned tid=threadIdx.x,lane=tid%32,warp=tid/32;
    unsigned id=blockIdx.x%256,first=16*(blockIdx.x/256);
    constexpr unsigned width=Pad/4;
    static_assert(width==32 || width==16);
    unsigned word=tid%width;
    for(unsigned row=tid/width;row<16;row+=blockDim.x/width){
        const uint32_t* input_row=input+size_t(id+256*(first+row))*L;
        auto loaded=load_words<W,L,width>(word,[&](unsigned i){return input_row[i];});
        unsigned active=__activemask();
        auto result=split_words<Split,width>(loaded,word,[&](uint32_t send,unsigned src){
            return __shfl_sync(active,send,src,width);
        });
        digits[row][word]=result.first;
        digits[row+16][word]=result.second;
    }
    __syncthreads();
    for(unsigned pair=warp;pair<Count/2;pair+=7) {
        int32_t c[1][8]{};
        bytecrt::resident_tile::accumulate<Count,Pad>(&digits[0][0],stride,weights,pair,lane,c,Mma{});
        unsigned pidx=2*pair+(lane%4)/2;
        uint32_t p=primes[pidx];uint32_t mu=reciprocal[pidx];
        #pragma unroll
        for(unsigned h=0;h<2;++h) {
            uint32_t value=evaluate_sign(c[0]+2*h,lane,p,mu,Neighbor{});
            size_t plane=pidx+(lane&1)*Count;
            planes[plane*N+id*K+first+lane/4+8*h]=value;
        }
    }
}
}
