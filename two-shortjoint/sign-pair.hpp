// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "word-reduce.hpp"
#include "warp-mod.hpp"

namespace two_digit {
template<class Neighbor>
#ifdef __CUDACC__
__device__ __forceinline__
#else
inline
#endif
uint32_t evaluate_sign(const int32_t* c,unsigned lane,uint32_t p,uint32_t mu,Neighbor neighbor) {
    const bool high=(lane&1)!=0;
    // The even lane needs its partner's low-digit columns 2 and 3.
    // The odd lane needs its partner's scaled high-digit columns 0 and 1.
    int32_t first=neighbor(high?c[0]:c[4]);
    int32_t second=neighbor(high?c[1]:c[5]);
    int32_t columns[]={high?first:c[0],high?second:c[1],
                       high?c[4]:first,high?c[5]:second};
    uint32_t own=reduce_dot(columns,p,mu),peer=neighbor(own);
    return high?bytecrt::warp_ntt::sub(peer,own,p):bytecrt::warp_ntt::add(own,peer,p);
}
}
