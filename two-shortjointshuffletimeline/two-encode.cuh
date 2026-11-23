// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "sign-pair.hpp"
#include "tile-fragments.cuh"
#include "two-digit-word.hpp"
#include "odd-fft.cuh"
#include "register-fft.cuh"

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
using namespace bytecrt::warp_ntt;
// Each lane owns one column and rows row, row+8, row+16, row+24.
template<unsigned Distance=16>
__device__ __forceinline__ void outer_four(uint32_t (&v)[4],unsigned row,
                                           const Twiddle* roots,const Mod& mod) {
    constexpr unsigned groups=32/(2*Distance);
    #pragma unroll
    for(unsigned part=0;part<4;++part) {
        const unsigned index=row+8*part;
        if constexpr(Distance>=8) {
            if(part&(Distance/8))continue;
            uint32_t a=v[part],b=multiply_twiddle(v[part+Distance/8],roots[groups+index/(2*Distance)],mod);
            v[part]=add(a,b,mod.p);v[part+Distance/8]=sub(a,b,mod.p);
        } else {
            bool upper=(row&Distance)!=0;
            uint32_t value=v[part];
            if(upper)value=multiply_twiddle(value,roots[groups+index/(2*Distance)],mod);
            uint32_t peer=__shfl_xor_sync(0xffffffffu,value,4*Distance);
            v[part]=upper?sub(peer,value,mod.p):add(value,peer,mod.p);
        }
    }
    if constexpr(Distance>1)outer_four<Distance/2>(v,row,roots,mod);
}

constexpr unsigned input_warps=6;

template<unsigned N,unsigned W,unsigned L,unsigned Count,unsigned Pad,unsigned Split>
__global__ __launch_bounds__(32*input_warps) void tile_forward(
        const uint32_t* a,const uint32_t* b,uint32_t* planes,const uint8_t* weights,
        const uint32_t* primes,const uint32_t* reciprocal,const uint32_t* phases,
        const Twiddle* roots,const Mod* mods) {
    constexpr unsigned K=32,J=N/K,P=2*Count,width=Pad/4;
    constexpr unsigned used_words=W==1747?28:width;
    constexpr unsigned stride=W==1747?used_words:width+4;
    static_assert(used_words%4==0);
    static_assert(Split<=4*used_words && (W+7)/8-4*(Split/4)<=4*used_words);
    __shared__ uint32_t digits[4][2*K][stride];
    const unsigned tid=threadIdx.x,lane=tid%32,warp=tid/32;
    const unsigned start=4*blockIdx.x,operand=blockIdx.y;
    const auto* input=operand?b:a;
    for(unsigned entry=tid/width;entry<4*K;entry+=blockDim.x/width) {
        unsigned col=entry/K,row=entry%K,word=tid%width;
        const auto* coefficient=input+size_t(start+col+J*row)*L;
        auto loaded=load_words<W,L,width>(word,[&](unsigned i){return coefficient[i];});
        auto split=split_words<Split,width>(loaded,word,[&](uint32_t send,unsigned source){
            return __shfl_sync(0xffffffffu,send,source,width);
        });
        unsigned packed=32*(row/16)+row%16;
        if(word<used_words) {
            digits[col][packed][word]=split.first;
            digits[col][packed+16][word]=split.second;
        }
    }
    __syncthreads();

    const unsigned plane_lane=lane%4;
    for(unsigned pair=warp;pair<Count/2;pair+=input_warps) {
        unsigned prime=2*pair+plane_lane/2;
        uint32_t p=primes[prime],mu=reciprocal[prime];
        uint32_t residues[4][4];
        #pragma unroll
        for(unsigned half=0;half<2;++half) {
            #pragma unroll
            for(unsigned col=0;col<4;++col) {
                int32_t sums[1][8]{};
                bytecrt::resident_tile::accumulate<Count,Pad,used_words>(
                    &digits[col][32*half][0],stride,weights,pair,lane,sums,Mma{});
                residues[2*half][col]=evaluate_sign(sums[0],lane,p,mu,Neighbor{});
                residues[2*half+1][col]=evaluate_sign(sums[0]+2,lane,p,mu,Neighbor{});
            }
            // Swap the lane's sign/prime with the register's column index.
            transpose_bit<1>(residues[2*half],lane);
            transpose_bit<2>(residues[2*half],lane);
            transpose_bit<1>(residues[2*half+1],lane);
            transpose_bit<2>(residues[2*half+1],lane);
        }
        const unsigned col=lane%4,row=lane/4;
        #pragma unroll
        for(unsigned side=0;side<2;++side) {
            unsigned mod_index=2*pair+side;
            const Mod mod=mods[mod_index];
            uint32_t phase[4];
            #pragma unroll
            for(unsigned part=0;part<4;++part)
                phase[part]=phases[size_t(mod_index)*N+(start+col)*K+row+8*part];
            #pragma unroll
            for(unsigned sign=0;sign<2;++sign) {
                unsigned local=2*side+sign;
                unsigned plane=operand*P+sign*Count+mod_index;
                uint32_t values[4];
                #pragma unroll
                for(unsigned part=0;part<4;++part)
                    values[part]=residues[part][local];
                outer_four(values,row,roots+size_t(mod_index)*(2*K+J),mod);
                #pragma unroll
                for(unsigned part=0;part<4;++part) {
                    unsigned freq=row+8*part;
                    unsigned index=(freq/2)*(2*J)+2*start+4*(freq%2)+col;
                    planes[size_t(plane)*N+index]=multiply(values[part],phase[part],mod);
                }
            }
        }
    }
}
} // namespace two_digit
