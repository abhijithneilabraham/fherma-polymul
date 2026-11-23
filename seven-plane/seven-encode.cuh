#pragma once
// SPDX-License-Identifier: Apache-2.0
#include "core/rns-encode-mma.cuh"
#include "encode.hpp"
namespace seven_encode {
struct Native {
    __device__ __forceinline__ void mma(int32_t* c,U a0,U a1,U a2,U a3,U b0,U b1) const {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
                     "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                    : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
                    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    __device__ __forceinline__ int32_t neighbor(int32_t v) const {
        return __shfl_xor_sync(0xffffffffu,v,1);
    }
    __device__ __forceinline__ U reduce(const int32_t* sums,const Field& field) const {
        return bytecrt::detail::encode_mma_reduce(sums,field.mod.p,field.reciprocal);
    }
};
template<unsigned N,unsigned W,unsigned Count>
__global__ void encode(const U* input,U* output,const U* weights,const Field* fields) {
    constexpr unsigned L=(W+31)/32,K=N/256;
    __shared__ U raw[16][L+1];
    unsigned id=blockIdx.x%256, first=(blockIdx.x/256)*16;
    unsigned tid=threadIdx.x,lane=tid%32,pair=tid/32;
    for(unsigned i=tid;i<16*(L+1);i+=blockDim.x) {
        unsigned row=i/(L+1),limb=i%(L+1);
        raw[row][limb]=limb==L?0:input[size_t(id+256*(first+row))*L+limb];
    }
    __syncthreads();
    if(pair>=Count/2)return;
    unsigned prime=2*pair+(lane%4)/2;
    const U* column=weights+8*(8*pair+lane/4);
    U result[2][7];
    evaluate<W,L>(raw,lane,fields[prime],column[lane%4],column[4+lane%4],result,Native{});
    if(!(lane&1)) {
        #pragma unroll
        for(unsigned h=0;h<2;++h) {
            #pragma unroll
            for(unsigned d=0;d<7;++d)
                output[(size_t(d)*Count+prime)*N+id*K+first+lane/4+8*h]=result[h][d];
        }
    }
}
}
