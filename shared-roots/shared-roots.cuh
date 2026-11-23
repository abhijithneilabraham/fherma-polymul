// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "warp-mod.hpp"
namespace bytecrt::warp_ntt {
// Every radix-two stage reads k in [0,127] in strides of a power of two.
// Fold bit 5 into bit 0, bit 6 into bit 1, etc., before indexing each
// 32-bit bank. Keep the inverse in its own plane at the same logical k.
struct SharedCyclicRoots {
    uint32_t values[2][128], reciprocals[2][128];
    __device__ static __forceinline__ unsigned slot(unsigned k) { return k^(k>>5); }
    __device__ __forceinline__ void stage(unsigned thread,const Twiddle* roots) {
        unsigned direction=thread/128,k=thread%128;
        Twiddle w=roots[direction ? ((256-k)&255) : k];
        values[direction][slot(k)]=w.value;
        reciprocals[direction][slot(k)]=w.reciprocal;
    }
};

template<unsigned Len,bool Inverse>
__device__ __forceinline__ Twiddle read_root(const SharedCyclicRoots& roots,unsigned k) {
    static_assert(Len==256);
    unsigned i=SharedCyclicRoots::slot(k);
    return {roots.values[Inverse][i],roots.reciprocals[Inverse][i]};
}
template<unsigned Len,bool Inverse>
__device__ __forceinline__ Twiddle read_root(const Twiddle* roots,unsigned k) {
    return roots[Inverse ? ((Len-k)&(Len-1)) : k];
}
}
