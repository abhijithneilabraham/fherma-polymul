// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <cstring>

namespace bytecrt::resident_tile {
struct WeightFragment {
    uint32_t low0, low1, high0, high1;
};
static_assert(sizeof(WeightFragment) == 16);

template<unsigned P, unsigned Kpad>
__device__ __forceinline__ void load_weights(const uint8_t* b, unsigned pair,
        unsigned lane, WeightFragment (&weights)[Kpad/32]) {
    #pragma unroll
    for (unsigned k = 0; k < Kpad/32; ++k) {
        const auto* owner = b + 16 * (32 * (k * (P/2) + pair) + lane);
#if defined(__CUDA_ARCH__)
        const auto packed = *reinterpret_cast<const uint4*>(owner);
        weights[k] = {packed.x, packed.y, packed.z, packed.w};
#else
        std::memcpy(&weights[k], owner, sizeof(WeightFragment));
#endif
    }
}

// The caller supplies the warp MMA primitive so the same fragment loads and
// accumulator mapping can be checked with a synchronous CPU warp model.
template<unsigned P, unsigned Kpad, unsigned UsedWords=Kpad/4, class Mma>
__device__ __forceinline__ void accumulate(const uint32_t* a, unsigned stride,
        const WeightFragment (&weights)[Kpad/32], unsigned lane,
        int32_t (&c)[1][8], Mma mma) {
    static_assert(P%2 == 0);
    static_assert(UsedWords==Kpad/4 || UsedWords+4==Kpad/4);
    unsigned g = lane / 4, t = lane % 4;
    #pragma unroll
    for (unsigned k = 0; k < Kpad; k += 32) {
        uint32_t a0 = a[g*stride + k/4 + t];
        uint32_t a1 = a[(g+8)*stride + k/4 + t];
        bool has_upper_words=k/4+4<UsedWords;
        uint32_t a2 = has_upper_words?a[g*stride + k/4 + t+4]:0;
        uint32_t a3 = has_upper_words?a[(g+8)*stride + k/4 + t+4]:0;
        uint32_t a4 = a[(g+16)*stride + k/4 + t];
        uint32_t a5 = a[(g+24)*stride + k/4 + t];
        uint32_t a6 = has_upper_words?a[(g+16)*stride + k/4 + t+4]:0;
        uint32_t a7 = has_upper_words?a[(g+24)*stride + k/4 + t+4]:0;
        auto weight = weights[k/32];
        mma(c[0],a0,a1,a2,a3,weight.low0,weight.low1);
        mma(c[0]+4,a4,a5,a6,a7,weight.high0,weight.high1);
    }
}
} // namespace bytecrt::resident_tile
