// SPDX-License-Identifier: Apache-2.0
// ByteCRT MT32 CRT with a cooperative exact-alpha calculation.
#pragma once
#include "short-fold.cuh"
#include "core/g1-crt.cuh"
#include "alpha-sum.cuh"
#include "warp-mod.hpp"
#include "odd-fft.cuh"
namespace bytecrt {
template <unsigned W, unsigned L, unsigned P, unsigned Kpad>
__global__ __launch_bounds__(224, 3) void g1_crt32_warp_alpha(const uint32_t* __restrict__ partial,
                        const uint8_t* __restrict__ b,
                        const float* __restrict__ invp,
                        const uint32_t* primes,const warp_ntt::Twiddle* inverse_theta,
                        const warp_ntt::Twiddle* roots,const warp_ntt::Mod* mods,
                        uint32_t gap,
                        uint32_t* __restrict__ out,
                        unsigned block_begin = 0) {
    static_assert((W == 1747 && L == 55 && P == 116 && Kpad == 480) ||
                  (W == 868 && L == 28 && P == 60 && Kpad == 256),
                  "G1 full-width geometries");
    constexpr unsigned MT=32,threads=224,count=P/2,N=W==1747?65536:32768,J=N/32;
    constexpr unsigned split=W==1747?880:440,columns=W==1747?120:64;
    constexpr unsigned input_pad=((count+1)*4+31)/32*32;
    constexpr unsigned stride=input_pad/4+3,plane_words=columns/4;
    __shared__ __align__(16) uint32_t input[3][MT][stride];
    __shared__ uint32_t dots[3][plane_words][MT];
    __shared__ uint16_t upper[3][plane_words][MT];
    static_assert(input_pad<=256 && stride>=L+2);
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31, warp = tid >> 5;
    const unsigned g = lane >> 2, t = lane & 3;
    const unsigned k_lane = 4 * t;
    const unsigned id=block_begin+blockIdx.x;
    const unsigned column_offset=(id/4)*32*4+4*lane+id%4;
    for(unsigned prime=warp;prime<count;prime+=threads/32) {
        const warp_ntt::Mod mod=mods[prime];
        const auto* factors=roots+size_t(prime)*(64+J)+32;
        uint32_t plus[1]={partial[size_t(prime)*N+column_offset]};
        uint32_t minus[1]={partial[size_t(count+prime)*N+column_offset]};
        warp_ntt::odd_fft<32,true>(plus,factors,mod,lane);
        warp_ntt::odd_fft<32,true>(minus,factors,mod,lane);
        uint32_t low=warp_ntt::add(plus[0],minus[0],mod.p);
        uint32_t high=warp_ntt::multiply_twiddle(
            warp_ntt::sub(plus[0],minus[0],mod.p),inverse_theta[prime],mod);
        input[0][lane][prime]=low;
        input[1][lane][prime]=high;
        input[2][lane][prime]=low+high;
    }
    if(warp<3)
        for(unsigned j=count;j<input_pad/4;++j)input[warp][lane][j]=0;
    __syncthreads();
    for(unsigned row=warp;row<MT;row+=threads/32) {
        float low=warp_alpha::sum(input[0][row],invp,count,lane);
        float high=warp_alpha::sum(input[1][row],invp,count,lane);
        if(!lane) {
            uint32_t x=__float2uint_rn(low),y=__float2uint_rn(high);
            input[0][row][count]=x;input[1][row][count]=y;input[2][row][count]=x+y;
        }
    }
    __syncthreads();

    for(unsigned tile=warp;tile<3*columns/8;tile+=threads/32) {
        unsigned matrix=tile/(columns/8), col=(tile%(columns/8))*8;
        int32_t c0=0,c1=0,c2=0,c3=0,c4=0,c5=0,c6=0,c7=0;
        for(unsigned k=0;k<input_pad;k+=32) {
            unsigned j=k/4+t;
            uint32_t a0=input[matrix][g][j],a1=input[matrix][g+8][j];
            uint32_t a2=input[matrix][g][j+4],a3=input[matrix][g+8][j+4];
            const uint8_t* ptr=b+(matrix*columns+col+g)*input_pad+k+k_lane;
            uint32_t b0=*reinterpret_cast<const uint32_t*>(ptr);
            uint32_t b1=*reinterpret_cast<const uint32_t*>(ptr+16);
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            uint32_t a4=input[matrix][g+16][j],a5=input[matrix][g+24][j];
            uint32_t a6=input[matrix][g+16][j+4],a7=input[matrix][g+24][j+4];
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+r"(c4), "+r"(c5), "+r"(c6), "+r"(c7)
                : "r"(a4), "r"(a5), "r"(a6), "r"(a7), "r"(b0), "r"(b1));
        }
        // Adjacent lanes own the low and high two bytes of one output word.
        // The two-byte partial is below 2^32 even for 256 active input bytes.
        auto store=[&](unsigned row,uint32_t x,uint32_t y) {
            uint32_t pair=x+(y<<8);
            uint32_t peer=__shfl_xor_sync(0xffffffffu,pair,1);
            if(!(t&1)) {
                uint64_t word=uint64_t(pair)+(uint64_t(peer)<<16);
                unsigned limb=col/4+t/2;
                dots[matrix][limb][row]=uint32_t(word);
                upper[matrix][limb][row]=uint16_t(word>>32);
            }
        };
        store(g,c0,c1);store(g+8,c2,c3);store(g+16,c4,c5);store(g+24,c6,c7);
    }
    __syncthreads();
    if(warp<3) {
        uint32_t carry=0;
        for(unsigned j=0;j<plane_words;++j) {
            uint64_t value=uint64_t(dots[warp][j][lane])+carry;
            dots[warp][j][lane]=uint32_t(value);
            carry=upper[warp][j][lane]+unsigned(value>>32);
        }
    }
    __syncthreads();
    if(!warp) {
        int64_t borrow=0;
        for(unsigned j=0;j<plane_words;++j) {
            int64_t value=int64_t(dots[2][j][lane])-dots[0][j][lane]-dots[1][j][lane]+borrow;
            dots[2][j][lane]=uint32_t(value);borrow=value>>32;
        }
        auto shifted=[&](unsigned matrix,unsigned j,unsigned bits) {
            unsigned at=j-bits/32, shift=bits%32;
            uint32_t lo=at<plane_words?dots[matrix][at][lane]:0;
            uint32_t prev=at-1<plane_words?dots[matrix][at-1][lane]:0;
            return shift?(lo<<shift)|(prev>>(32-shift)):lo;
        };
        uint64_t carry=0;
        uint32_t* digits=input[0][lane];
        for(unsigned j=0;j<L+2;++j) {
            uint64_t v=carry+(j<plane_words?dots[0][j][lane]:0);
            v+=uint64_t(gap)*shifted(1,j,2*split-W);
            v+=shifted(2,j,split);
            digits[j]=uint32_t(v);carry=v>>32;
        }
        short_fold::reduce<W,L>(digits,gap);
    }
    __syncthreads();
    for (unsigned word = tid; word < MT * L; word += threads) {
        const unsigned row = word / L;
        const unsigned coefficient=id+row*J;
        out[size_t(coefficient) * L + word % L] = input[0][row][word % L];
    }
}

inline void launch_g1_crt32_warp_alpha_transposed(unsigned W, unsigned M,
        const uint32_t* a, const uint8_t* b, const float* inverse, const uint32_t* primes, const warp_ntt::Twiddle* inverse_theta, const warp_ntt::Twiddle* roots,const warp_ntt::Mod* mods,
        uint32_t gap, uint32_t* output, cudaStream_t stream) {
    if (W == 1747 && M == 65536)
        g1_crt32_warp_alpha<1747,55,116,480><<<M/32,224,0,stream>>>(a,b,inverse,primes,inverse_theta,roots,mods,gap,output);
    else if (W == 868 && M == 32768)
        g1_crt32_warp_alpha<868,28,60,256><<<M/32,224,0,stream>>>(a,b,inverse,primes,inverse_theta,roots,mods,gap,output);
    else throw std::runtime_error("Cooperative alpha requires full staged geometries");
}
} // namespace bytecrt
