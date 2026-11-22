// SPDX-License-Identifier: Apache-2.0
// ByteCRT MT32 byte dots fused with the unchanged cuPQC first transform stage.
#pragma once
#include "core/rns-encode-mma.cuh"
#include "core/ntt-pipeline.cuh"
#include "warp-output.cuh"
#include "fused-layout.hpp"

namespace bytecrt {
template<unsigned N, unsigned P, unsigned GroupPrimes>
__global__ void fused_input_first(const uint32_t* input_a, const uint32_t* input_b,
        const uint8_t* b, const uint32_t* primes, const uint64_t* mu, uint32_t* ab,
        const uint32_t* tw, const uint32_t* psi,
        const cupqc::nttConst<uint32_t>* constants) {
    static_assert(GroupPrimes == 8, "four complete MMA warps and the retained 128-thread NTT");
    constexpr unsigned K = N / 256, L = N == 65536 ? 55 : 28;
    constexpr unsigned Kpad = N == 65536 ? 224 : 128;
    constexpr unsigned MT = 32, threads = 128;
    constexpr unsigned shared_stride = Kpad + 16, word_stride = shared_stride / 4;
    static_assert((N == 65536 && P == 114) || (N == 32768 && P == 57), "exact RNS plans");
    static_assert(4 * L * 255 * 255 < (uint32_t{1} << 31), "exact signed dot sums");
    __shared__ __align__(16) uint8_t shared_a[MT][shared_stride];
    // SDK first-stage workspace is exactly K uint32 words per prime.
    extern __shared__ uint32_t residues[];
    const unsigned tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const unsigned g = lane >> 2, t = lane & 3, k_lane = 4 * t;
    const unsigned id = blockIdx.x, first_prime = blockIdx.y * GroupPrimes;
    const unsigned count = P - first_prime < GroupPrimes ? P - first_prime : GroupPrimes;
    const unsigned operand = blockIdx.z;
    const uint32_t* inputs = operand ? input_b : input_a;
    auto* shared_words = reinterpret_cast<uint32_t*>(shared_a);
    for (unsigned first_row = 0; first_row < K; first_row += MT) {
        for (unsigned word = tid; word < MT * word_stride; word += threads) {
            const unsigned row = first_row + word / word_stride;
            const unsigned limb = word % word_stride;
            shared_words[word] = limb < L
                ? inputs[fused_layout::input_word(id, row, limb, L)] : 0;
        }
        __syncthreads();
        // Uniform within each warp. All block threads reach both tile barriers.
        if (2 * warp < count) {
            const unsigned b_col = 4 * first_prime + warp * 8 + g;
            int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            int32_t c4 = 0, c5 = 0, c6 = 0, c7 = 0;
            #pragma unroll
            for (unsigned k = 0; k < Kpad; k += 32) {
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(shared_a[g] + k + k_lane);
                const uint32_t a1 = *reinterpret_cast<const uint32_t*>(shared_a[g + 8] + k + k_lane);
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(shared_a[g] + k + k_lane + 16);
                const uint32_t a3 = *reinterpret_cast<const uint32_t*>(shared_a[g + 8] + k + k_lane + 16);
                uint32_t b0 = 0, b1 = 0;
                if (b_col < 4 * P) {
                    const uint8_t* column = b + std::size_t{b_col} * Kpad + k + k_lane;
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
            // Local prime offsets use the same paired-lane reducer. Its input
            // primes and reciprocals point to this block's global prime group.
            warp_encoding::store_pair<false>(c0, c1, lane, warp, 0, first_row, 0,
                0, K, count, 0, primes + first_prime, mu + first_prime, residues);
            warp_encoding::store_pair<false>(c2, c3, lane, warp, 0, first_row, 0,
                1, K, count, 0, primes + first_prime, mu + first_prime, residues);
            warp_encoding::store_pair<false>(c4, c5, lane, warp, 0, first_row, 0,
                2, K, count, 0, primes + first_prime, mu + first_prime, residues);
            warp_encoding::store_pair<false>(c6, c7, lane, warp, 0, first_row, 0,
                3, K, count, 0, primes + first_prime, mu + first_prime, residues);
        }
        // Protect immutable MMA input until every warp finishes, and publish
        // every residue before the first transform reads the shared buffer.
        __syncthreads();
    }
    ntt::Staged<N, cupqc::nttDirection::FORWARD> f;
    for (unsigned local = 0; local < count; ++local) {
        const unsigned prime = first_prime + local;
        const auto mc = constants[prime];
        auto* data = residues + local * K;
        for (unsigned i = tid; i < K; i += threads)
            data[i] = mont_mul_word(f.to_mont(data[i], mc),
                psi[std::size_t{prime} * N + id * K + i], mc.p, mc.n_dash);
        __syncthreads();
        f.stage_1_execute(data, tw + std::size_t{prime} * N, mc.p);
        __syncthreads();
        f.stage_1_store(data, ab + fused_layout::plane(operand, prime, P, N), id);
        __syncthreads();
    }
}

template<unsigned N, unsigned P, unsigned GroupPrimes>
inline void launch_fused_input(const uint32_t* a, const uint32_t* b,
        const uint8_t* powers, const uint32_t* primes, const uint64_t* mu,
        uint32_t* ab, const uint32_t* tw, const uint32_t* psi,
        const cupqc::nttConst<uint32_t>* constants, cudaStream_t stream) {
    constexpr unsigned K = N / 256;
    constexpr auto workspace = cupqc::fwd_stage_1_ntt_shared_workspace_size<N, 256, uint32_t>();
    static_assert(workspace == K * sizeof(uint32_t), "non-overlapping SDK workspace per prime");
    fused_input_first<N, P, GroupPrimes><<<dim3(256, (P + GroupPrimes - 1) / GroupPrimes, 2),
        128, GroupPrimes * workspace, stream>>>(a, b, powers, primes, mu, ab, tw, psi, constants);
}
} // namespace bytecrt
