// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "warp-mod.hpp"

namespace bytecrt::warp_ntt {
// Evaluate at the odd powers of a primitive 2*Len root. Each group of
// 2*Distance entries shares one root, including the cross-lane butterflies.
template<unsigned Len, unsigned Distance, bool Inverse>
__device__ __forceinline__ void odd_step(uint32_t (&v)[Len/32], const Twiddle* roots,
                                         const Mod& mod, unsigned lane) {
    constexpr unsigned groups = Len / (2 * Distance);
    #pragma unroll
    for (unsigned r = 0; r < Len / 32; ++r) {
        unsigned i = lane + 32 * r;
        if constexpr (Distance >= 32) {
            if (r & (Distance / 32)) continue;
            uint32_t a = v[r], b = v[r + Distance/32];
            Twiddle root = roots[groups + i / (2 * Distance)];
            if constexpr (Inverse) {
                v[r] = sum_twice(a,b,mod.p);
                v[r + Distance/32] = lazy_twiddle(a+2*mod.p-b,root,mod);
            } else {
                a = reduce_once(a,2*mod.p);
                b = lazy_twiddle(b,root,mod);
                v[r] = a+b;
                v[r + Distance/32] = a+2*mod.p-b;
            }
        } else {
            bool upper = (lane & Distance) != 0;
            Twiddle root = roots[groups + i / (2 * Distance)];
            uint32_t value = v[r];
            if constexpr (!Inverse) {
                value = upper?lazy_twiddle(value,root,mod):reduce_once(value,2*mod.p);
            }
            uint32_t peer = __shfl_xor_sync(0xffffffffu, value, Distance);
            if constexpr(Inverse) {
                v[r] = upper?lazy_twiddle(peer+2*mod.p-value,root,mod):sum_twice(value,peer,mod.p);
            } else {
                v[r] = upper?peer+2*mod.p-value:value+peer;
            }
        }
    }
}

template<unsigned Len, bool Inverse, unsigned Distance = Inverse ? 1 : Len/2>
__device__ __forceinline__ void odd_fft(uint32_t (&v)[Len/32], const Twiddle* roots,
                                        const Mod& mod, unsigned lane) {
    odd_step<Len, Distance, Inverse>(v, roots, mod, lane);
    if constexpr (Inverse && Distance < Len/2) odd_fft<Len, Inverse, Distance*2>(v, roots, mod, lane);
    if constexpr (!Inverse && Distance > 1) odd_fft<Len, Inverse, Distance/2>(v, roots, mod, lane);
}
}
