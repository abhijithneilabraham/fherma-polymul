// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "warp-fft.cuh"

namespace bytecrt::warp_ntt {

// Exchange a bit of the local register index with the same bit of the lane.
// The two matching quadrants stay local. The other two cross lanes together.
template<unsigned Bit, unsigned Values>
__device__ __forceinline__ void transpose_bit(uint32_t (&v)[Values], unsigned lane) {
    static_assert(Values == 4 || Values == 8);
    static_assert(Bit < Values);
    #pragma unroll
    for (unsigned low = 0; low < Values; ++low) {
        if (low & Bit) continue;
        const unsigned high = low + Bit;
        const bool upper = (lane & Bit) != 0;
        uint32_t send = upper ? v[low] : v[high];
        uint32_t receive = __shfl_xor_sync(0xffffffffu, send, Bit);
        if (upper) v[low] = receive;
        else v[high] = receive;
    }
}

__device__ __forceinline__ void transpose_low_bits(uint32_t (&v)[8], unsigned lane) {
    transpose_bit<1>(v, lane);
    transpose_bit<2>(v, lane);
    transpose_bit<4>(v, lane);
}

// Three low index bits now select registers. Each lane owns both outputs.
template<unsigned Distance, bool Inverse, unsigned Stride>
__device__ __forceinline__ void register_step(uint32_t (&v)[8],
        const Twiddle* roots, const Mod& m) {
    #pragma unroll
    for (unsigned low = 0; low < 8; ++low) {
        if (low & Distance) continue;
        unsigned index = (low & (Distance - 1)) * (256 / (2 * Distance));
        uint32_t a = v[low], b = v[low + Distance];
        if constexpr (Inverse) {
            if (index) b = multiply_twiddle(b, roots[Stride*(256-index)], m);
            v[low] = add(a, b, m.p);
            v[low + Distance] = sub(a, b, m.p);
        } else {
            v[low] = add(a, b, m.p);
            v[low + Distance] = index ? scaled_difference(a, b, roots[Stride*index], m) : sub(a,b,m.p);
        }
    }
}

template<bool Inverse,unsigned Stride>
__device__ __forceinline__ void middle_fft(uint32_t (&v)[8], const Twiddle* roots,
        const Mod& m, unsigned lane) {
    if constexpr (Inverse) {
        register_step<1,true,Stride>(v, roots, m);
        register_step<2,true,Stride>(v, roots, m);
        register_step<4,true,Stride>(v, roots, m);
        transpose_low_bits(v, lane);
        fft<256,true,Stride,8>(v, roots, m, lane);
    } else {
        step<256,128,false,Stride>(v, roots, m, lane);
        step<256,64,false,Stride>(v, roots, m, lane);
        step<256,32,false,Stride>(v, roots, m, lane);
        step<256,16,false,Stride>(v, roots, m, lane);
        step<256,8,false,Stride>(v, roots, m, lane);
        transpose_low_bits(v, lane);
        register_step<4,false,Stride>(v, roots, m);
        register_step<2,false,Stride>(v, roots, m);
        register_step<1,false,Stride>(v, roots, m);
    }
}

} // namespace bytecrt::warp_ntt
