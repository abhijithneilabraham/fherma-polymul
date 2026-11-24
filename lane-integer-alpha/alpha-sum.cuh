// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
namespace bytecrt::warp_alpha {
__device__ __forceinline__ uint32_t partial(const uint32_t* terms,const uint32_t* reciprocal,
        unsigned count,unsigned lane) {
    uint32_t total=0;
    for(unsigned prime=lane;prime<count;prime+=32) {
#ifdef __CUDA_ARCH__
        total+=__umulhi(terms[prime],reciprocal[prime]);
#else
        total+=uint32_t((uint64_t(terms[prime])*reciprocal[prime])>>32);
#endif
    }
    return total;
}
__device__ __forceinline__ uint32_t sum(const uint32_t* terms,const uint32_t* reciprocal,
        unsigned count,unsigned lane) {
    uint32_t fixed=__reduce_add_sync(0xffffffffu,partial(terms,reciprocal,count,lane));
    return (fixed+(1u<<22))>>23;
}
}
