#pragma once
// SPDX-License-Identifier: Apache-2.0
#include "core/rns-encode-mma.cuh"
#include "encode.hpp"
#include "odd-fft.cuh"
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
// A block owns a column and a group of prime pairs. Each warp loads its
// coefficient fragments once. The same 14 shared planes serve every pair.
template<unsigned N,unsigned W,unsigned Count>
__global__ void tile_forward(const U* input,U* output,const U* weights,const Field* fields,
                             const U* phases,const Twiddle* roots) {
    constexpr unsigned L=(W+31)/32,K=N/256,pairs=N==65536?3:5;
    static_assert(Count%(2*pairs)==0);
    __shared__ U spectrum[14][K];
    unsigned id=blockIdx.x,group=blockIdx.y,tid=threadIdx.x,lane=tid%32,warp=tid/32;
    Fragments fragments{};
    if(warp<K/16) {
        unsigned row=16*warp+lane/4;
        const U* low=input+size_t(id+256*row)*L;
        fragments=load_fragments<W>(low,low+8*256*L,lane%4);
    }
    #pragma unroll 1
    for(unsigned j=0;j<pairs;++j) {
        unsigned pair=group*pairs+j;
        if(warp<K/16) {
            unsigned side=(lane%4)/2,prime=2*pair+side;
            const U* column=weights+8*(8*pair+lane/4);
            U result[2][7];
            evaluate(fragments,lane,fields[prime],column[lane%4],column[4+lane%4],result,Native{});
            if(!(lane&1)) {
                #pragma unroll
                for(unsigned h=0;h<2;++h) {
                    #pragma unroll
                    for(unsigned d=0;d<7;++d)
                        spectrum[2*d+side][16*warp+lane/4+8*h]=result[h][d];
                }
            }
        }
        __syncthreads();
        if(warp<14) {
            unsigned prime=2*pair+warp%2, plane=(warp/2)*Count+prime;
            const Mod m=fields[prime].mod;
            U v[K/32];
            #pragma unroll
            for(unsigned k=0;k<K/32;++k)v[k]=spectrum[warp][32*k+lane];
            odd_fft<K,false>(v,roots+size_t(prime)*(2*K+256),m,lane);
            #pragma unroll
            for(unsigned k=0;k<K/32;++k) {
                unsigned i=id*K+32*k+lane;
                output[size_t(plane)*N+i]=multiply(v[k],phases[size_t(prime)*N+i],m);
            }
        }
        __syncthreads();
    }
}
}
