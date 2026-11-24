// SPDX-License-Identifier: Apache-2.0
#pragma once


#include "warp-carry.cuh"
#include "odd-fft.cuh"
#include "inverse-field.hpp"
namespace four_decode {
struct CarryWarp {
    __device__ __forceinline__ uint32_t ballot(bool value) const {
        return __ballot_sync(0xffffffffu,value);
    }
    __device__ __forceinline__ uint32_t get(uint32_t value,unsigned lane) const {
        return __shfl_sync(0xffffffffu,value,lane);
    }
};

template<unsigned N,unsigned W,unsigned Count>
__global__ void inverse_decode(const uint32_t* input,uint32_t* out,const uint8_t* weights,
                       const uint32_t* alpha_reciprocal,uint32_t gap,const bytecrt::warp_ntt::Twiddle* roots,
                       const four_inverse::Field* fields) {
    constexpr unsigned MT=32,NT=56,L=(W+31)/32,groups=(L+13)/14;
    constexpr unsigned planes=4*Count, K=32, J=N/K, Kpad=32*((4*planes+4+31)/32);
    constexpr unsigned word_stride=Kpad/4+4,digit_stride=(L+2)|1;
    static_assert(Kpad*255u*255u<0x80000000u);
    __shared__ __align__(16) uint8_t shared_a[MT][4*word_stride];
    __shared__ uint32_t lows[MT][digit_stride],highs[MT][digit_stride];
    unsigned tid=threadIdx.x,lane=tid%32,warp=tid/32;
    unsigned g=lane/4,t=lane%4;
    unsigned id=blockIdx.x;
    auto* shared_words=reinterpret_cast<uint32_t*>(shared_a);
    for(unsigned plane=warp;plane<Kpad/4;plane+=8){
        uint32_t x=0;
        if(plane<planes){
            unsigned prime=plane%Count;
            unsigned address=(id/4)*K*4+lane*4+id%4;
            uint32_t v[1]={input[size_t(plane)*N+address]};
            bytecrt::warp_ntt::odd_fft<K,true>(v,roots+size_t(prime)*(2*K+J)+K,fields[prime].mod,lane);
            x=v[0];
        }
        shared_words[lane*word_stride+plane]=x;
    }
    __syncthreads();
    // One warp owns each coefficient row. Each active lane owns a prime.
    for(unsigned row=warp;row<MT;row+=8) {
        uint32_t* terms=shared_words+row*word_stride;
        uint32_t contribution[4]={};
        if(lane<Count) {
            uint32_t x[4],y[4];
            #pragma unroll
            for(unsigned d=0;d<4;++d)x[d]=terms[d*Count+lane];
            const auto& field=fields[lane];
            radix4::transform(x,field.inverse,field.mod,y);
            const uint32_t reciprocal=alpha_reciprocal[lane];
            #pragma unroll
            for(unsigned d=0;d<4;++d) {
                uint32_t v=bytecrt::warp_ntt::multiply_twiddle(y[d],field.recovery[d],field.mod);
                terms[d*Count+lane]=v;
#ifdef __CUDA_ARCH__
                contribution[d]=__umulhi(v,reciprocal);
#else
                contribution[d]=uint32_t((uint64_t(v)*reciprocal)>>32);
#endif
            }
        }
        #pragma unroll
        for(unsigned d=0;d<4;++d) {
            uint32_t total=__reduce_add_sync(0xffffffffu,contribution[d]);
            if(!lane)shared_a[row][4*planes+d]=uint8_t((total+(1u<<22))>>23);
        }
    }
    __syncthreads();
    if(warp<7) {
        for (unsigned group = 0; group < groups; ++group) {
            constexpr unsigned rows=groups*7,header=128*((4*(rows+1)+127)/128);
            constexpr unsigned weights_start=128*((header+8*rows*(Kpad/32)+127)/128);
            const auto* offsets=reinterpret_cast<const uint32_t*>(weights);
            const unsigned slot=7*group+warp, stop=offsets[slot+1];
            int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            int32_t c4 = 0, c5 = 0, c6 = 0, c7 = 0;
            for (unsigned cursor=offsets[slot];cursor<stop;++cursor) {
                const uint8_t* indices=weights+header+8*cursor;
                unsigned first=4*indices[t],second=4*indices[4+t];
                const auto* tile=reinterpret_cast<const uint32_t*>(weights+weights_start)+64*cursor;
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(shared_a[g] + first);
                const uint32_t a1 = *reinterpret_cast<const uint32_t*>(shared_a[g + 8] + first);
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(shared_a[g] + second);
                const uint32_t a3 = *reinterpret_cast<const uint32_t*>(shared_a[g + 8] + second);
                uint32_t b0=tile[lane],b1=tile[32+lane];
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                const uint32_t a4 = *reinterpret_cast<const uint32_t*>(shared_a[g + 16] + first);
                const uint32_t a5 = *reinterpret_cast<const uint32_t*>(shared_a[g + 24] + first);
                const uint32_t a6 = *reinterpret_cast<const uint32_t*>(shared_a[g + 16] + second);
                const uint32_t a7 = *reinterpret_cast<const uint32_t*>(shared_a[g + 24] + second);
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+r"(c4), "+r"(c5), "+r"(c6), "+r"(c7)
                    : "r"(a4), "r"(a5), "r"(a6), "r"(a7), "r"(b0), "r"(b1));
            }
            const int32_t pairs[4][2]={{c0,c1},{c2,c3},{c4,c5},{c6,c7}};
            #pragma unroll
            for(unsigned part=0;part<4;++part) {
                uint32_t x=pairs[part][0],y=pairs[part][1];
                uint32_t z=__shfl_xor_sync(0xffffffffu,x,1),w=__shfl_xor_sync(0xffffffffu,y,1);
                bytecrt::warp_carry::store_dot<L>(lows,highs,lane,warp,group,part,x,y,z,w);
            }
        }

    }
    __syncthreads();
    for(unsigned r=warp;r<MT;r+=8) {
        unsigned coefficient=id+r*J;
        bytecrt::warp_carry::finish<W,L>(lows[r],highs[r],gap,out+size_t(coefficient)*L,lane,CarryWarp{});
    }
}
}
