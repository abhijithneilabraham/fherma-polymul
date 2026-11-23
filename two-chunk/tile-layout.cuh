// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
namespace bytecrt::warp_ntt {
template<unsigned K>
__device__ __forceinline__ void gather_spectrum(const uint32_t* input,
        uint32_t (&tile)[32][257],unsigned first,unsigned warp,unsigned lane) {
    for(unsigned id=warp;id<256;id+=32) tile[lane][id]=input[id*K+first+lane];
}
}
