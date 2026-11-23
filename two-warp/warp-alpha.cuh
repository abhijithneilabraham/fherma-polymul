// SPDX-License-Identifier: Apache-2.0
// ByteCRT MT32 CRT with a cooperative exact-alpha calculation.
#pragma once
#include "core/g1-crt.cuh"
#include "warp-carry.cuh"
#include "warp-mod.hpp"
namespace bytecrt {
struct CarryWarp {
    __device__ __forceinline__ uint32_t ballot(bool value) const {
        return __ballot_sync(0xffffffffu,value);
    }
    __device__ __forceinline__ uint32_t get(uint32_t value,unsigned lane) const {
        return __shfl_sync(0xffffffffu,value,lane);
    }
};
template <unsigned W, unsigned L, unsigned P, unsigned Kpad, bool Transposed = false>
__global__ __launch_bounds__(224, 3) void g1_crt32_warp_alpha(const uint8_t* __restrict__ a,
                        const uint8_t* __restrict__ b,
                        const uint32_t* __restrict__ reciprocal48,
                        const uint32_t* primes,const warp_ntt::Twiddle* inverse_theta,
                        const uint32_t* __restrict__ q,
                        uint32_t gap,
                        uint32_t* __restrict__ out,
                        unsigned block_begin = 0) {
    static_assert((W == 1747 && L == 55 && P == 116 && Kpad == 480) ||
                  (W == 868 && L == 28 && P == 60 && Kpad == 256),
                  "G1 full-width geometries");
    constexpr unsigned MT = 32, NT = 56, threads = 224;
    constexpr unsigned digit_stride = (L + 2) | 1;
    constexpr unsigned groups = (L + 13) / 14;
    // Four padding words make the m16n8k32 A gather bank-bijective:
    // stride_words mod 32 is 28 (main) or 4 (aux), bank = g*stride+t.
    constexpr unsigned row_words = Kpad / 4;
    constexpr unsigned shared_stride = Kpad + 16;
    constexpr unsigned shared_word_stride = shared_stride / 4;
    constexpr unsigned transpose_m = 256;
    constexpr unsigned transpose_k = W == 1747 ? 256 : 128;
    __shared__ __align__(16) uint8_t shared_a[MT][shared_stride];
    __shared__ uint32_t lows[MT][digit_stride],highs[MT][digit_stride];

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31, warp = tid >> 5;
    const unsigned g = lane >> 2, t = lane & 3;
    const unsigned k_lane = 4 * t;
    const unsigned block = block_begin + blockIdx.x;
    const unsigned id = Transposed ? block % transpose_m : 0;
    const unsigned first_row = (Transposed ? block / transpose_m : block) * MT;
    auto* shared_words = reinterpret_cast<uint32_t*>(shared_a);
    const auto* a_words = reinterpret_cast<const uint32_t*>(a);
    for (unsigned word = tid; word < MT * row_words; word += threads) {
        if constexpr (Transposed) {
            // A_t[id][prime_word][i]. A full warp reads 32 consecutive i.
            const unsigned row = word % MT, prime_word = word / MT;
            shared_words[row * shared_word_stride + prime_word] =
                prime_word < P ? a_words[(size_t(id) * row_words + prime_word) * transpose_k + first_row + row] : 0;
        } else {
            const unsigned row = word / row_words, prime_word = word % row_words;
            shared_words[row * shared_word_stride + prime_word] =
                a_words[(size_t(first_row) + row) * row_words + prime_word];
        }
    }
    __syncthreads();

    constexpr unsigned count=P/2;
    // Assign both signs of a prime to the same lane. Consume the new
    // residues in registers while accumulating the centered CRT quotient.
    for(unsigned row=warp;row<MT;row+=threads/32) {
        uint32_t* terms=shared_words+row*shared_word_stride;
        uint64_t low_sum=0,high_sum=0;
        for(unsigned j=lane;j<count;j+=32){
            uint32_t p=primes[j],plus=terms[j],minus=terms[count+j];
            uint32_t lo=warp_ntt::add(plus,minus,p);
            uint32_t hi=warp_ntt::multiply_twiddle(warp_ntt::sub(plus,minus,p),
                                                    inverse_theta[j],warp_ntt::Mod{p,0,0,0});
            terms[j]=lo;terms[count+j]=hi;
            low_sum+=uint64_t(lo)*reciprocal48[j];
            high_sum+=uint64_t(hi)*reciprocal48[j];
        }
        for(unsigned distance=16;distance;distance/=2){
            low_sum+=__shfl_down_sync(0xffffffffu,low_sum,distance,32);
            high_sum+=__shfl_down_sync(0xffffffffu,high_sum,distance,32);
        }
        if(!lane){
            terms[P]=(low_sum+(uint64_t{1}<<47))>>48;
            terms[P+1]=(high_sum+(uint64_t{1}<<47))>>48;
        }
    }
    __syncthreads();
    // C=4*(P+2)*255 < 2^17: lo <= C*(2^32-1), carry <= C-1.
    // Thus total fits u64; B4 already includes every input-byte shift.

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
                const uint8_t* column = b + b_col * Kpad + k + k_lane;
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
            warp_carry::store_dot<L>(lows,highs,lane,warp,group,part,x,y,z,w);
        }
    }
    __syncthreads();
    for(unsigned row=warp;row<MT;row+=threads/32) {
        unsigned coefficient=Transposed?id+(first_row+row)*transpose_m:first_row+row;
        warp_carry::finish<W,L>(lows[row],highs[row],gap,out+size_t(coefficient)*L,lane,CarryWarp{});
    }
}

inline void launch_g1_crt32_warp_alpha_transposed(unsigned W, unsigned M,
        const uint8_t* a, const uint8_t* b, const uint32_t* reciprocal48, const uint32_t* primes, const warp_ntt::Twiddle* inverse_theta, const uint32_t* q,
        uint32_t gap, uint32_t* output, cudaStream_t stream) {
    if (W == 1747 && M == 65536)
        g1_crt32_warp_alpha<1747,55,116,480,true><<<M/32,224,0,stream>>>(a,b,reciprocal48,primes,inverse_theta,q,gap,output);
    else if (W == 868 && M == 32768)
        g1_crt32_warp_alpha<868,28,60,256,true><<<M/32,224,0,stream>>>(a,b,reciprocal48,primes,inverse_theta,q,gap,output);
    else throw std::runtime_error("Cooperative alpha requires full staged geometries");
}
} // namespace bytecrt
