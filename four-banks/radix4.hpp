// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "warp-mod.hpp"
#include <stdexcept>

namespace radix4 {
using namespace bytecrt::warp_ntt;
using U=uint32_t;
inline Twiddle prepare(U root,const Mod& m) {
    if(uint64_t(root)*root%m.p!=m.p-1)throw std::invalid_argument("fourth root");
    return {root,U((uint64_t(root)<<32)/m.p)};
}
#ifdef __CUDACC__
#define FOUR_INLINE __device__ __forceinline__
#else
#define FOUR_INLINE inline
#endif
FOUR_INLINE void transform(const U (&a)[4],Twiddle root,const Mod& m,U (&out)[4]) {
    U even=add(a[0],a[2],m.p),odd=add(a[1],a[3],m.p);
    U delta=sub(a[0],a[2],m.p),rot=multiply_twiddle(sub(a[1],a[3],m.p),root,m);
    out[0]=add(even,odd,m.p);out[2]=sub(even,odd,m.p);
    out[1]=add(delta,rot,m.p);out[3]=sub(delta,rot,m.p);
}
}
