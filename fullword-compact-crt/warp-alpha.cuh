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
                        const uint32_t* __restrict__ correction,
                        const float* __restrict__ invp,
                        const uint32_t* primes,const warp_ntt::Twiddle* inverse_theta,
                        const warp_ntt::Twiddle* roots,const warp_ntt::Mod* mods,
                        uint32_t gap,
                        uint32_t* __restrict__ out,
                        unsigned block_begin = 0) {
    static_assert((W == 1747 && L == 55 && P == 112 && Kpad == 448) ||
                  (W == 868 && L == 28 && P == 60 && Kpad == 256),
                  "G1 full-width geometries");
    constexpr unsigned MT = 32, NT = 56, threads = 224;
    constexpr unsigned digit_stride = (L + 2) | 1;
    constexpr unsigned groups = (L + 13) / 14;
    // Four padding words make the m16n8k32 A gather bank-bijective:
    // Both row strides in words are 4 mod 8.
    constexpr unsigned row_words = Kpad / 4;
    constexpr unsigned shared_stride = Kpad + 16;
    constexpr unsigned shared_word_stride = shared_stride / 4;
    static_assert(P+2 <= shared_word_stride);
    constexpr unsigned N=W==1747?65536:32768;
    constexpr unsigned J=N/32;
    __shared__ __align__(16) uint8_t shared_a[MT][shared_stride];
    __shared__ int32_t shared_c[MT][NT + 1];
    __shared__ uint32_t digits[MT][digit_stride];

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31, warp = tid >> 5;
    const unsigned g = lane >> 2, t = lane & 3;
    const unsigned k_lane = 4 * t;
    const unsigned id=block_begin+blockIdx.x;
    const unsigned column_offset=(id/4)*32*4+4*lane+id%4;
    auto* shared_words=reinterpret_cast<uint32_t*>(shared_a);
    uint32_t* row_words_ptr=shared_words+lane*shared_word_stride;

    constexpr unsigned count=P/2;
    for(unsigned prime=warp;prime<count;prime+=threads/32) {
        const warp_ntt::Mod mod=mods[prime];
        const auto* factors=roots+size_t(prime)*(64+J)+32;
        uint32_t plus[1]={partial[size_t(prime)*N+column_offset]};
        uint32_t minus[1]={partial[size_t(count+prime)*N+column_offset]};
        warp_ntt::odd_fft<32,true>(plus,factors,mod,lane);
        warp_ntt::odd_fft<32,true>(minus,factors,mod,lane);
        row_words_ptr[prime]=warp_ntt::add(plus[0],minus[0],mod.p);
        row_words_ptr[count+prime]=warp_ntt::multiply_twiddle(
            warp_ntt::sub(plus[0],minus[0],mod.p),inverse_theta[prime],mod);
    }
    if(warp==0)
        for(unsigned j=P;j<row_words;++j)row_words_ptr[j]=0;
    __syncthreads();
    for (unsigned row=warp;row<MT;row+=threads/32) {
        uint32_t* terms=shared_words+row*shared_word_stride;
        float low=warp_alpha::sum(terms,invp,count,lane);
        float high=warp_alpha::sum(terms+count,invp,count,lane);
        if(!lane) { terms[P]=__float2uint_rn(low); terms[P+1]=__float2uint_rn(high); }
    }
    __syncthreads();

    // C=4*(P+2)*255 < 2^17: lo <= C*(2^32-1), carry <= C-1.
    // Thus total fits u64; B4 already includes every input-byte shift.
    uint64_t carry = 0;
    uint32_t alpha_low=0,alpha_high=0;
    if constexpr(W==1747) {
        if(tid<MT) {
            alpha_low=shared_words[tid*shared_word_stride+P];
            alpha_high=shared_words[tid*shared_word_stride+P+1];
        }
    }
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
        const unsigned c_col = warp * 8 + 2 * t;
        shared_c[g][c_col] = c0;
        shared_c[g][c_col + 1] = c1;
        shared_c[g + 8][c_col] = c2;
        shared_c[g + 8][c_col + 1] = c3;
        shared_c[g + 16][c_col] = c4;
        shared_c[g + 16][c_col + 1] = c5;
        shared_c[g + 24][c_col] = c6;
        shared_c[g + 24][c_col + 1] = c7;
        __syncthreads();

        if (tid < MT) {
            for (unsigned local_limb = 0; local_limb < 14; ++local_limb) {
                const unsigned limb = group * 14 + local_limb;
                if (limb < L) {
                    const int32_t* sums = shared_c[tid] + 4 * local_limb;
                    const uint64_t lo = uint64_t{static_cast<uint32_t>(sums[0])} +
                        (uint64_t{static_cast<uint32_t>(sums[1])} << 8) +
                        (uint64_t{static_cast<uint32_t>(sums[2])} << 16) +
                        (uint64_t{static_cast<uint32_t>(sums[3])} << 24);
                    uint64_t total = carry + lo;
                    if constexpr(W==1747) {
                        total+=uint64_t(alpha_low)*correction[limb];
                        total+=uint64_t(alpha_high)*correction[L+limb];
                    }
                    digits[tid][limb] = static_cast<uint32_t>(total);
                    carry = total >> 32;
                }
            }
        }
        // All readers finish before the next group overwrites shared C.
        __syncthreads();
    }

    if (tid < MT) {
        digits[tid][L] = static_cast<uint32_t>(carry);
        digits[tid][L + 1] = static_cast<uint32_t>(carry >> 32);
        short_fold::reduce<W, L>(digits[tid], gap);
    }
    __syncthreads();
    for (unsigned word = tid; word < MT * L; word += threads) {
        const unsigned row = word / L;
        const unsigned coefficient=id+row*J;
        out[size_t(coefficient) * L + word % L] = digits[row][word % L];
    }
}

inline void launch_g1_crt32_warp_alpha_transposed(unsigned W, unsigned M,
        const uint32_t* a, const uint8_t* b, const uint32_t* correction, const float* inverse, const uint32_t* primes, const warp_ntt::Twiddle* inverse_theta, const warp_ntt::Twiddle* roots,const warp_ntt::Mod* mods,
        uint32_t gap, uint32_t* output, cudaStream_t stream) {
    if (W == 1747 && M == 65536)
        g1_crt32_warp_alpha<1747,55,112,448><<<M/32,224,0,stream>>>(a,b,correction,inverse,primes,inverse_theta,roots,mods,gap,output);
    else if (W == 868 && M == 32768)
        g1_crt32_warp_alpha<868,28,60,256><<<M/32,224,0,stream>>>(a,b,correction,inverse,primes,inverse_theta,roots,mods,gap,output);
    else throw std::runtime_error("Cooperative alpha requires full staged geometries");
}
} // namespace bytecrt
