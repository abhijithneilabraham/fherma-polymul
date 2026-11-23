// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "warp-mod.hpp"
#ifdef __CUDACC__
#define PHASE_HD __host__ __device__ __forceinline__
#else
#define PHASE_HD inline
#endif
namespace bytecrt::warp_ntt {
constexpr unsigned phase_radix=256;
PHASE_HD constexpr unsigned forward_phase_words(unsigned n) {
    return phase_radix+2*n/phase_radix;
}
PHASE_HD uint32_t forward_phase_at(const uint32_t* table,unsigned exponent,const Mod& mod) {
    return multiply(table[exponent%phase_radix],table[phase_radix+exponent/phase_radix],mod);
}
PHASE_HD constexpr unsigned reverse_five(unsigned row) {
    return ((row&1)<<4)|((row&2)<<2)|(row&4)|((row&8)>>2)|((row&16)>>4);
}
}
#undef PHASE_HD
