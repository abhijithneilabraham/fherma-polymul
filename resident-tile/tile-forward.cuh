// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "core/rns-encode-mma.cuh"
#include "odd-fft.cuh"
#include "tile-fragments.cuh"

namespace bytecrt::warp_ntt {
struct TileMma {
    __device__ __forceinline__ void operator()(int32_t* c, uint32_t a0,uint32_t a1,
        uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1) const {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
            : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
};

template<unsigned N,unsigned L,unsigned P,unsigned Kpad>
__global__ __launch_bounds__(4*(N/256)) void tile_forward(
        const uint32_t* left,const uint32_t* right,uint32_t* spectrum,
        const uint8_t* weights,const uint32_t* primes,const uint64_t* reciprocal,
        const uint32_t* phase,const Twiddle* roots,const Mod* mods) {
    constexpr unsigned K=N/256, stride=Kpad/4+4;
    extern __shared__ uint32_t storage[];
    uint32_t* raw=storage;
    uint32_t* residues=storage+K*stride;
    unsigned id=blockIdx.x,operand=blockIdx.y,tid=threadIdx.x;
    unsigned warp=tid/32,lane=tid%32,row_tile=warp/4,pair=warp%4;
    const uint32_t* input=operand?right:left;
    // Load all K coefficients for this id once. Every group reuses them.
    for(unsigned word=tid; word<K*stride; word+=blockDim.x) {
        unsigned row=word/stride, limb=word%stride;
        raw[word]=limb<L ? input[(id+256*row)*L+limb] : 0;
    }
    __syncthreads();
    for(unsigned first=0;first<P;first+=8) {
        if (first+2*pair<P) {
            int32_t c[1][8]{};
            resident_tile::accumulate<P,Kpad>(raw+32*row_tile*stride,stride,weights,
                                               first/2+pair,lane,c,TileMma{});
            #pragma unroll
            for(unsigned q=0;q<4;++q) {
                int32_t a=c[0][2*q],b=c[0][2*q+1];
                int32_t x=__shfl_xor_sync(0xffffffffu,a,1);
                int32_t y=__shfl_xor_sync(0xffffffffu,b,1);
                unsigned local=2*pair+(lane%4)/2,prime=first+local;
                if (!(lane&1) && prime<P) {
                    int32_t sums[4]={a,b,x,y};
                    unsigned row=32*row_tile+lane/4+8*q;
                    residues[local*K+row]=detail::encode_mma_reduce(sums,primes[prime],reciprocal[prime]);
                }
            }
        }
        __syncthreads();
        if (warp<8 && first+warp<P) {
            unsigned prime=first+warp;
            const Mod m=mods[prime];
            uint32_t v[K/32];
            #pragma unroll
            for(unsigned r=0;r<K/32;++r) v[r]=residues[warp*K+lane+32*r];
            odd_fft<K,false>(v, roots+size_t(prime)*(2*K+256),m,lane);
            #pragma unroll
            for(unsigned r=0;r<K/32;++r) {
                unsigned row=lane+32*r;
                size_t index=size_t(prime)*N+id*K+row;
                spectrum[size_t(operand)*P*N+index]=multiply(v[r],phase[index],m);
            }
        }
        __syncthreads();
    }
}
}
