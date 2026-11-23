#pragma once
// SPDX-License-Identifier: Apache-2.0
#include "core/rns-encode-mma.cuh"
#include "encode.hpp"
#include "odd-fft.cuh"
namespace four_encode {
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
template<unsigned W>
constexpr unsigned shared_bytes() {
    constexpr unsigned L=(W+31)/32, parallel=W==1747?3:2;
    return sizeof(U)*(4*32*(L+1)+parallel*8*4*33);
}
// One block owns four outer columns and all digit planes of one operand.
// A pair of rows and four columns fill one aligned 32-byte sector.
template<unsigned N,unsigned W,unsigned Count>
__global__ __launch_bounds__(W==1747?768:512) void tile_forward(
        const U* a,const U* b,U* output,const U* weights,const Field* fields,
        const U* phases,const Twiddle* roots) {
    constexpr unsigned K=32,J=N/K,L=(W+31)/32,P=4*Count,parallel=W==1747?3:2;
    static_assert(Count%(2*parallel)==0);
    extern __shared__ U storage[];
    auto* raw=reinterpret_cast<U (*)[K][L+1]>(storage);
    auto* tile=reinterpret_cast<U (*)[4][33]>(storage+4*K*(L+1));
    unsigned base=4*blockIdx.x,operand=blockIdx.y,tid=threadIdx.x,lane=tid%32,warp=tid/32;
    const U* input=operand?b:a;
    for(unsigned i=tid;i<4*K*(L+1);i+=blockDim.x){
        unsigned limb=i%(L+1),entry=i/(L+1),id=base+entry/K,row=entry%K;
        storage[i]=limb==L?0:input[size_t(id+J*row)*L+limb];
    }
    __syncthreads();
    unsigned slot=warp/8,column=(warp%8)/2,half=warp%2;
    #pragma unroll 1
    for(unsigned start=0;start<Count/2;start+=parallel) {
        unsigned pair=start+slot,side=(lane%4)/2,prime=2*pair+side;
        const U* weight=weights+64*pair;
        U result[2][4];
        auto* rows=reinterpret_cast<U (*)[16][L+1]>(&raw[column][16*half]);
        evaluate<W,L,Count>(*rows,lane,fields[prime],weight,result,Native{});
        if(!(lane&1)){
            #pragma unroll
            for(unsigned h=0;h<2;++h) {
                #pragma unroll
                for(unsigned d=0;d<4;++d)tile[8*slot+2*d+side][column][16*half+lane/4+8*h]=result[h][d];
            }
        }
        __syncthreads();
        for(unsigned task=warp;task<parallel*8*4;task+=blockDim.x/32) {
            unsigned local=task/4,col=task%4,modulus=2*(start+local/8)+local%2;
            const Mod m=fields[modulus].mod;
            U v[1]={tile[local][col][lane]};
            odd_fft<K,false>(v,roots+size_t(modulus)*(2*K+J),m,lane);
            unsigned index=(base+col)*K+lane;
            tile[local][col][lane]=multiply(v[0],phases[size_t(modulus)*N+index],m);
        }
        __syncthreads();
        for(unsigned i=tid;i<parallel*8*4*K;i+=blockDim.x) {
            unsigned local=i/(4*K),row=(i/4)%K,col=i%4;
            unsigned modulus=2*(start+local/8)+local%2, digit=(local%8)/2;
            unsigned plane=operand*P+digit*Count+modulus;
            size_t index=size_t(row/2)*(2*J)+size_t(base)*2+4*(row%2)+col;
            output[size_t(plane)*N+index]=tile[local][col][row];
        }
        __syncthreads();
    }
}
}
