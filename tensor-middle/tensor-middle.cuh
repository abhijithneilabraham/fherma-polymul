// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "core/rns-encode-mma.cuh"
#include "tensor-radix.cuh"

namespace bytecrt::warp_ntt {
struct WarpOps {
    __device__ __forceinline__ void sync() const { __syncwarp(); }
    __device__ __forceinline__ int32_t neighbor(int32_t value) const {
        return __shfl_xor_sync(0xffffffffu,value,1);
    }
    __device__ __forceinline__ uint32_t reduce(const int32_t* dots,uint32_t p,uint64_t mu) const {
        return detail::encode_mma_reduce(dots,p,mu);
    }
    __device__ __forceinline__ void mma(int32_t* c,const uint32_t* a,const uint32_t* b) const {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
                : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),
                  "r"(b[0]),"r"(b[4]));
    }
};
template<unsigned N,unsigned P>
__global__ void middle(const uint32_t* input,uint32_t* output,const uint32_t* inverse_phase,
        const Twiddle* all_roots,const Mod* mods,const uint32_t* packed,const uint64_t* reciprocal) {
    constexpr unsigned K=N/256;
    __shared__ uint32_t weights[64*20];
    __shared__ TensorRow scratch[8];
    __shared__ uint32_t output_tile[8][260];
    unsigned tid=threadIdx.x,warp=tid/32,lane=tid%32;
    unsigned prime=blockIdx.y,first=blockIdx.x*8,row=first+warp;
    for(unsigned i=tid;i<1024;i+=256) weights[(i/16)*20+i%16]=packed[prime*1024+i];
    __syncthreads();
    TensorRow& s=scratch[warp];
    const Mod m=mods[prime];
    const Twiddle* roots=all_roots+size_t(prime)*(2*K+256)+2*K;
    uint64_t mu=reciprocal[prime];
    uint32_t left[8];
    #pragma unroll
    for(unsigned operand=0;operand<2;++operand) {
        #pragma unroll
        for(unsigned j=0;j<8;++j) {
            unsigned i=lane+32*j;
            s.word[i%16][i/16]=input[size_t(operand*P+prime)*N+row*256+i];
        }
        __syncwarp();
        radix16<true>(s,weights,roots,m,mu,lane,WarpOps{});
        radix16<false>(s,weights,roots,m,mu,lane,WarpOps{});
        #pragma unroll
        for(unsigned j=0;j<8;++j) {
            unsigned i=lane+32*j;
            uint32_t v=s.word[i%16][i/16];
            if (!operand) left[j]=v;
            else s.word[i%16][i/16]=multiply(left[j],v,m);
        }
        __syncwarp();
    }
    radix16<true>(s,weights,roots,m,mu,lane,WarpOps{});
    radix16<false>(s,weights,roots,m,mu,lane,WarpOps{});
    #pragma unroll
    for(unsigned j=0;j<8;++j) {
        unsigned i=lane+32*j,reverse=(256-i)%256;
        output_tile[warp][i]=multiply(s.word[reverse%16][reverse/16],
            inverse_phase[size_t(prime)*N+row*256+i],m);
    }
    __syncthreads();
    for(unsigned i=tid;i<2048;i+=256)
        output[size_t(prime)*N+(i/8)*K+first+i%8]=output_tile[i%8][i/8];
}
}
