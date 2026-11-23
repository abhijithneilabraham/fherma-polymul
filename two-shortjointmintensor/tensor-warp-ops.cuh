// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "word-reduce.hpp"
#include "tensor-radix.cuh"

namespace bytecrt::warp_ntt {
struct WarpOps {
    __device__ __forceinline__ void sync() const { __syncwarp(); }
    __device__ __forceinline__ int32_t neighbor(int32_t value) const {
        return __shfl_xor_sync(0xffffffffu,value,1);
    }
    __device__ __forceinline__ uint32_t reduce(const int32_t* dots,uint32_t p,uint32_t mu) const {
        return two_digit::reduce_dot(dots,p,mu);
    }
    __device__ __forceinline__ void mma(int32_t* c,const uint32_t* a,const uint32_t* b) const {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
                : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),
                  "r"(b[0]),"r"(b[4]));
    }
};
} // namespace bytecrt::warp_ntt
