// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "folded-kernels.cuh"
#include "joint-terms.cuh"
#include "warp-carry.cuh"
namespace folded {
struct CarryWarp {
    __device__ __forceinline__ uint32_t ballot(bool value) const {
        return __ballot_sync(0xffffffffu,value);
    }
    __device__ __forceinline__ uint32_t get(uint32_t value,unsigned lane) const {
        return __shfl_sync(0xffffffffu,value,lane);
    }
};

__global__ void decode_joint(const uint32_t* weighted,uint32_t* out,const Plan* plan,
        const Mod* mods,const JointWeights* weights) {
    constexpr unsigned MT=32,NT=56,Kpad=joint_pad,groups=(L+13)/14;
    constexpr unsigned word_stride=joint_terms+4,digit_stride=(L+2)|1;
    static_assert(Kpad*255u*255u<0x80000000u);
    __shared__ __align__(16) uint8_t shared_a[MT][4*word_stride];
    __shared__ uint32_t lows[MT][digit_stride],highs[MT][digit_stride];
    unsigned tid=threadIdx.x,lane=tid%32,warp=tid/32;
    unsigned g=lane/4,t=lane%4,k_lane=4*t;
    unsigned id=blockIdx.x%256,first=32*(blockIdx.x/256);
    auto* shared_words=reinterpret_cast<uint32_t*>(shared_a);
    for(unsigned i=tid;i<128*MT;i+=256) {
        unsigned plane=i/MT,row=i%MT;
        shared_words[row*word_stride+plane]=weighted[(size_t(id)*128+plane)*256+first+row];
    }
    __syncthreads();
    unsigned row=tid/8,digit=tid%8;
    uint32_t* terms=shared_words+row*word_stride;
    joint_terms_for_digit(terms,digit,plan,mods,weights);
    __syncthreads();
    if(warp<7) {
        for (unsigned group = 0; group < groups; ++group) {
            const unsigned b_col = group * NT + warp * 8 + g;
            int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            int32_t c4 = 0, c5 = 0, c6 = 0, c7 = 0;
            for (unsigned k = 0; k < Kpad; k += 32) {
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(shared_a[g] + k + k_lane);
                const uint32_t a1 = *reinterpret_cast<const uint32_t*>(shared_a[g + 8] + k + k_lane);
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(shared_a[g] + k + k_lane + 16);
                const uint32_t a3 = *reinterpret_cast<const uint32_t*>(shared_a[g + 8] + k + k_lane + 16);
                uint32_t b0 = 0, b1 = 0;
                if (b_col < 4 * L) {
                    const uint8_t* column = weights->matrix[b_col] + k + k_lane;
                    b0 = *reinterpret_cast<const uint32_t*>(column);
                    b1 = *reinterpret_cast<const uint32_t*>(column + 16);
                }
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                const uint32_t a4 = *reinterpret_cast<const uint32_t*>(shared_a[g + 16] + k + k_lane);
                const uint32_t a5 = *reinterpret_cast<const uint32_t*>(shared_a[g + 24] + k + k_lane);
                const uint32_t a6 = *reinterpret_cast<const uint32_t*>(shared_a[g + 16] + k + k_lane + 16);
                const uint32_t a7 = *reinterpret_cast<const uint32_t*>(shared_a[g + 24] + k + k_lane + 16);
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
        unsigned coefficient=id+(first+r)*256;
        bytecrt::warp_carry::finish<W,L>(lows[r],highs[r],gap,out+size_t(coefficient)*L,lane,CarryWarp{});
    }
}
}
