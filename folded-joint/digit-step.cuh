// SPDX-License-Identifier: MIT
#pragma once
#include "folded-math.hpp"
#include "warp-mod.hpp"
namespace folded {
using bytecrt::warp_ntt::add;
using bytecrt::warp_ntt::sub;
using bytecrt::warp_ntt::multiply;
using bytecrt::warp_ntt::Mod;

// DIF produces physical bit-reversed frequencies; DIT consumes that order.
template<bool inverse, int distance>
__device__ __forceinline__ uint32_t digit_step(uint32_t x, unsigned lane, const Prime& p, const Mod& mod) {
    const unsigned index = (lane % distance) * (4 / distance);
    const bool upper = (lane & distance) != 0;
    if constexpr(inverse) {
        if (upper) x = multiply(x, p.roots[(8-index)%8], mod);
    }
    uint32_t peer = __shfl_xor_sync(0xffffffff, x, distance, 8);
    if (upper) {
        x = sub(peer, x, mod.p);
        if constexpr(!inverse) x = multiply(x, p.roots[index], mod);
    } else x = add(x,peer,mod.p);
    return x;
}

}
