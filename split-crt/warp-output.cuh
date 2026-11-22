// SPDX-License-Identifier: Apache-2.0
// Paired-lane output for ByteCRT unsigned m16n8k32 accumulators.
#pragma once
#include <cstddef>
#include <cstdint>
#include "fp32-reduce.cuh"

namespace bytecrt::warp_encoding {

template<bool Transposed>
__device__ __forceinline__ void store_pair(int32_t c0, int32_t c1,
        unsigned lane, unsigned warp, unsigned group, unsigned first_row, unsigned id,
        unsigned quarter, unsigned N, unsigned P, unsigned operand,
        const uint32_t* primes, const float* inverse, uint32_t* output) {
    // Every lane reaches both exchanges. Even lanes own digits 0/1; their
    // odd neighbors own digits 2/3 of the same row and prime.
    const int32_t c2 = __shfl_xor_sync(0xffffffffu, c0, 1);
    const int32_t c3 = __shfl_xor_sync(0xffffffffu, c1, 1);
    const unsigned prime = group * 14 + 2 * warp + (lane & 3) / 2;
    const unsigned row = first_row + (lane >> 2) + 8 * quarter;
    const unsigned coefficient = Transposed ? id + row * 256 : row;
    if (!(lane & 1) && prime < P && coefficient < N) {
        const int32_t sums[4] = {c0, c1, c2, c3};
        const uint32_t residue = fast_encoding::reduce(sums, primes[prime], inverse[prime]);
        const unsigned destination = Transposed ? id * (N / 256) + row : coefficient;
        output[(std::size_t{operand} * P + prime) * N + destination] = residue;
    }
}
} // namespace bytecrt::warp_encoding
