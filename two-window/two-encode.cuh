// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "sign-pair.hpp"
#include "tile-fragments.cuh"
#include "two-digit-word.hpp"
#include "odd-fft.cuh"
#include "prime-window.hpp"

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
// Adjacent lanes represent two signs of two primes. Give each a different
// eight-bank group when publishing, then undo the same row permutation.
__device__ __forceinline__ unsigned local_row(unsigned plane,unsigned row) {
    return row ^ (8*(2*(plane%2)+plane/12));
}
template<unsigned N, unsigned Pad>
constexpr unsigned tile_bytes() {
    constexpr unsigned rows=N/256;
    return sizeof(uint32_t)*(64*2*(Pad/4+4)+rows*24);
}

// One block transforms up to twelve primes. Consume the raw column in 64-row chunks.
template<unsigned N,unsigned W,unsigned L,unsigned Count,unsigned Pad,unsigned Split>
__global__ __launch_bounds__(256) void tile_forward(
        const uint32_t* a,const uint32_t* b,uint32_t* out,const uint8_t* weights,
        const uint32_t* primes,const uint32_t* reciprocal,
        const uint32_t* phases,
        const bytecrt::warp_ntt::Twiddle* roots,const bytecrt::warp_ntt::Mod* mods,unsigned first) {
    using namespace bytecrt::warp_ntt;
    constexpr unsigned K=N/256,width=Pad/4,stride=width+4,chunk=64;
    static_assert(two_window::capacity==12);
    static_assert(K%chunk==0);
    extern __shared__ uint32_t storage[];
    uint32_t* values=storage+2*chunk*stride;
    unsigned tid=threadIdx.x,lane=tid%32,warp=tid/32,id=blockIdx.x,operand=blockIdx.y;
    const uint32_t* input=operand?b:a;
    for(unsigned base=0;base<K;base+=chunk) {
        for(unsigned row=tid/width;row<chunk;row+=blockDim.x/width) {
            unsigned word=tid%width;
            const uint32_t* source=input+size_t(id+256*(base+row))*L;
            auto limbs=load_words<W,L,width>(word,[&](unsigned i){return source[i];});
            auto digits=split_words<Split,width>(limbs,word,[&](uint32_t send,unsigned src){
                return __shfl_sync(0xffffffffu,send,src,width);
            });
            unsigned position=32*(row/16)+row%16;
            storage[position*stride+word]=digits.first;
            storage[(position+16)*stride+word]=digits.second;
        }
        __syncthreads();
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
                unsigned plane=local+12*(lane&1);
                unsigned row=base+16*(warp/2)+lane/4+8*half;
                values[plane*K+local_row(plane,row)]=evaluate_sign(accum[0]+2*half,lane,p,mu,Neighbor{});
            }
        }
        __syncthreads();
    }
    for(unsigned local=warp;local<24;local+=blockDim.x/32) {
        unsigned prime=first+local%12;
        if(prime>=Count) continue;
        unsigned plane=local;
        uint32_t v[K/32];
        #pragma unroll
        for(unsigned j=0;j<K/32;++j) v[j]=values[local*K+local_row(local,32*j+lane)];
        Mod mod=mods[prime];
        odd_fft<K,false>(v,roots+size_t(prime)*(2*K+256),mod,lane);
        #pragma unroll
        for(unsigned j=0;j<K/32;++j) {
            size_t index=id*K+32*j+lane;
            out[size_t(operand*2*two_window::capacity+plane)*N+index]=multiply(v[j],phases[size_t(prime)*N+index],mod);
        }
    }
}
}
