// SPDX-License-Identifier: Apache-2.0
// ByteCRT MT32 CRT with a cooperative exact-alpha calculation.
#pragma once
#include "short-fold.cuh"
#include "core/g1-crt.cuh"
#include "alpha-sum.cuh"
#include "split-inverse.cuh"
namespace bytecrt {
template <unsigned W, unsigned L, unsigned P, unsigned Kpad>
__global__ __launch_bounds__(224) void inverse_crt(const uint32_t* __restrict__ partial,
                        const warp_ntt::Twiddle* roots,const warp_ntt::Mod* mods,
                        const uint8_t* __restrict__ b,
                        const float* __restrict__ invp,
                        const uint32_t* primes,const warp_ntt::Twiddle* inverse_theta,
                        const uint32_t* __restrict__ q,
                        uint32_t gap,
                        uint32_t* __restrict__ out) {
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
    __shared__ __align__(16) uint8_t planes[2][MT][shared_stride];
    __shared__ int32_t shared_c[MT][NT + 1];
    __shared__ uint32_t digits[MT][digit_stride];

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31, warp = tid >> 5;
    const unsigned g = lane >> 2, t = lane & 3;
    const unsigned k_lane = 4 * t;
    constexpr unsigned K=transpose_k,group=warp_ntt::first_group<K>,N=256*K;
    unsigned base=2*(blockIdx.x%128),low=blockIdx.x/128;
    unsigned side=(lane%8)/4,row=4*(lane/8)+lane%4;
    auto* words=reinterpret_cast<uint32_t*>(planes);
    for(unsigned prime=warp;prime<row_words;prime+=threads/32) {
        uint32_t value[2]{};
        if(prime<P) {
            #pragma unroll
            for(unsigned j=0;j<2;++j)value[j]=partial[size_t(prime)*N+warp_ntt::partial_index(base+side,low,row+16*j)];
            warp_ntt::inverse_pair(value,row,roots+size_t(prime)*(2*K+256)+K,mods[prime]);
        }
        #pragma unroll
        for(unsigned j=0;j<2;++j)words[(side*MT+row+16*j)*shared_word_stride+prime]=value[j];
    }
    __syncthreads();

    for(unsigned column=0;column<2;++column) {
        auto& shared_a=planes[column];
        auto* shared_words=reinterpret_cast<uint32_t*>(shared_a);
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
            const unsigned coefficient=(base+column)+256*(low+group*row);
            out[size_t(coefficient) * L + word % L] = digits[row][word % L];
        }
        __syncthreads();
    }
}

} // namespace bytecrt
