// SPDX-License-Identifier: Apache-2.0
// ByteCRT MT32 CRT with a cooperative exact-alpha calculation.
#pragma once
#include "short-fold.cuh"
#include "core/g1-crt.cuh"
#include "alpha-sum.cuh"
#include "warp-mod.hpp"
namespace bytecrt {
template <unsigned W, unsigned L, unsigned P, unsigned Kpad, bool Transposed = false>
__global__ __launch_bounds__(224, 3) void g1_crt32_warp_alpha(const uint8_t* __restrict__ a,
                        const uint8_t* __restrict__ b,
                        const float* __restrict__ invp,
                        const uint32_t* primes,const warp_ntt::Twiddle* inverse_theta,
                        const uint32_t* __restrict__ q,
                        uint32_t gap,
                        uint32_t* __restrict__ out,
                        unsigned block_begin = 0) {
    static_assert((W == 1747 && L == 55 && P == 120 && Kpad == 512) ||
                  (W == 868 && L == 28 && P == 64 && Kpad == 288),
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
    __shared__ int32_t shared_c[MT][NT + 1];
    __shared__ uint32_t digits[MT][digit_stride];

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
    for(unsigned i=tid;i<MT*count;i+=threads) {
        uint32_t* terms=shared_words+(i/count)*shared_word_stride;
        unsigned j=i%count;
        uint32_t p=primes[j],plus=terms[j],minus=terms[count+j];
        terms[j]=warp_ntt::add(plus,minus,p);
        terms[count+j]=warp_ntt::multiply_twiddle(warp_ntt::sub(plus,minus,p),inverse_theta[j],warp_ntt::Mod{p,0,0,0});
    }
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
                    const uint64_t total = carry + lo;
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
        const unsigned coefficient = Transposed ? id + (first_row + row) * transpose_m
                                                : first_row + row;
        out[size_t(coefficient) * L + word % L] = digits[row][word % L];
    }
}

inline void launch_g1_crt32_warp_alpha_transposed(unsigned W, unsigned M,
        const uint8_t* a, const uint8_t* b, const float* inverse, const uint32_t* primes, const warp_ntt::Twiddle* inverse_theta, const uint32_t* q,
        uint32_t gap, uint32_t* output, cudaStream_t stream) {
    if (W == 1747 && M == 65536)
        g1_crt32_warp_alpha<1747,55,120,512,true><<<M/32,224,0,stream>>>(a,b,inverse,primes,inverse_theta,q,gap,output);
    else if (W == 868 && M == 32768)
        g1_crt32_warp_alpha<868,28,64,288,true><<<M/32,224,0,stream>>>(a,b,inverse,primes,inverse_theta,q,gap,output);
    else throw std::runtime_error("Cooperative alpha requires full staged geometries");
}
} // namespace bytecrt
