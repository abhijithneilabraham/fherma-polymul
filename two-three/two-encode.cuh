// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "word-reduce.hpp"
#include "tile-fragments.cuh"
#include "two-digit-word.hpp"
#include "odd-fft.cuh"

namespace two_digit {
struct Mma {
    __device__ __forceinline__ void operator()(int32_t* c,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1) const {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
            : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
};
__device__ __forceinline__ uint32_t dot(int a,int b,unsigned lane,uint32_t p,uint32_t mu) {
    int c=__shfl_xor_sync(0xffffffffu,a,1),d=__shfl_xor_sync(0xffffffffu,b,1);
    int32_t sums[4]={a,b,c,d};
    return (lane&1)?0:reduce_dot(sums,p,mu);
}
template<unsigned N, unsigned Pad>
constexpr unsigned tile_bytes() {
    constexpr unsigned rows=N/256;
    return sizeof(uint32_t)*rows*(2*(Pad/4+4)+24);
}

// One block loads a whole column. Each MMA warp covers sixteen rows and two
// primes. Up to 24 warps then transform the resulting plus and minus planes.
template<unsigned N,unsigned W,unsigned L,unsigned Count,unsigned Pad,unsigned Split>
__global__ __launch_bounds__(4*(N/256)) void tile_forward(
        const uint32_t* a,const uint32_t* b,uint32_t* out,const uint8_t* weights,
        const uint32_t* primes,const uint32_t* reciprocal,
        const bytecrt::warp_ntt::Twiddle* theta,const uint32_t* phases,
        const bytecrt::warp_ntt::Twiddle* roots,const bytecrt::warp_ntt::Mod* mods) {
    using namespace bytecrt::warp_ntt;
    constexpr unsigned K=N/256,width=Pad/4,stride=width+4;
    extern __shared__ uint32_t storage[];
    uint32_t* values=storage+2*K*stride;
    unsigned tid=threadIdx.x,lane=tid%32,warp=tid/32,id=blockIdx.x,operand=blockIdx.y;
    const uint32_t* input=operand?b:a;
    for(unsigned row=tid/width;row<K;row+=blockDim.x/width) {
        unsigned word=tid%width;
        const uint32_t* source=input+size_t(id+256*row)*L;
        auto limbs=load_words<W,L,width>(word,[&](unsigned i){return source[i];});
        auto digits=split_words<Split,width>(limbs,word,[&](uint32_t send,unsigned src){
            return __shfl_sync(0xffffffffu,send,src,width);
        });
        unsigned position=32*(row/16)+row%16;
        storage[position*stride+word]=digits.first;
        storage[(position+16)*stride+word]=digits.second;
    }
    __syncthreads();
    for(unsigned first=0;first<Count;first+=12) {
        #pragma unroll 1
        for(unsigned step=0;step<12;step+=4) {
            unsigned local_pair=step/2+warp%2, pair=first/2+local_pair;
            if(2*pair>=Count) continue; // Uniform across this warp.
            int32_t accum[1][8]{};
            bytecrt::resident_tile::accumulate<Count,Pad>(storage+32*(warp/2)*stride,stride,
                weights,pair,lane,accum,Mma{});
            unsigned local=2*local_pair+(lane%4)/2, prime=first+local;
            uint32_t p=primes[prime],mu=reciprocal[prime];
            #pragma unroll
            for(unsigned half=0;half<2;++half) {
                uint32_t lo=dot(accum[0][2*half],accum[0][2*half+1],lane,p,mu);
                uint32_t hi=dot(accum[0][4+2*half],accum[0][5+2*half],lane,p,mu);
                if(!(lane&1)) {
                    hi=multiply_twiddle(hi,theta[prime],Mod{p,0,0,0});
                    unsigned row=16*(warp/2)+lane/4+8*half;
                    values[local*K+row]=add(lo,hi,p);
                    values[(local+12)*K+row]=sub(lo,hi,p);
                }
            }
        }
        __syncthreads();
        for(unsigned local=warp;local<24;local+=blockDim.x/32) {
            unsigned prime=first+local%12;
            if(prime>=Count) continue;
            unsigned plane=prime+Count*(local/12);
            uint32_t v[K/32];
            #pragma unroll
            for(unsigned j=0;j<K/32;++j) v[j]=values[local*K+32*j+lane];
            Mod mod=mods[prime];
            odd_fft<K,false>(v,roots+size_t(prime)*(2*K+256),mod,lane);
            #pragma unroll
            for(unsigned j=0;j<K/32;++j) {
                size_t index=id*K+32*j+lane;
                out[size_t(operand*2*Count+plane)*N+index]=multiply(v[j],phases[size_t(prime)*N+index],mod);
            }
        }
        __syncthreads();
    }
}
}
