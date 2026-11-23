// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "warp-fft.cuh"
#include "odd-fft.cuh"

namespace bytecrt::warp_ntt {
// A 32-point outer transform leaves a J=N/32 cyclic product. Split J into
// 32 columns and S rows so each local transform uses at most two registers
// per lane. The row/column exchanges stay inside the block.
template<unsigned N,unsigned P>
__global__ void middle(const uint32_t* input,uint32_t* output,const uint32_t* inverse_phases,
                       const Twiddle* roots,const Mod* mods) {
    constexpr unsigned K=32,J=N/K,R=32,S=J/R;
    unsigned first_row=2*blockIdx.x, plane=blockIdx.y, prime=plane%(P/4);
    unsigned tid=threadIdx.x,lane=tid%32,half=tid/256,warp=(tid%256)/32;
    const Mod m=mods[prime];
    const Twiddle* table=roots+size_t(prime)*(2*K+J)+2*K;
    __shared__ uint32_t left[2][S][R+1],right[2][S][R+1];
    for(unsigned i=tid;i<2*J;i+=blockDim.x) {
        unsigned side=(i/4)%2,id=4*(i/8)+i%4;
        size_t index=size_t(first_row)*J+i;
        left[side][id/R][id%R]=input[size_t(plane)*N+index];
        right[side][id/R][id%R]=input[size_t(P+plane)*N+index];
    }
    __syncthreads();
    for(unsigned u=warp;u<R;u+=8){
        uint32_t a[S/32],b[S/32];
        #pragma unroll
        for(unsigned j=0;j<S/32;++j){a[j]=left[half][j*32+lane][u];b[j]=right[half][j*32+lane][u];}
        fft<S,false,R>(a,table,m,lane);fft<S,false,R>(b,table,m,lane);
        #pragma unroll
        for(unsigned j=0;j<S/32;++j){
            unsigned storage=j*32+lane, freq=__brev(storage)>>(32-(S==64?6:5));
            Twiddle rotation=table[u*freq];
            left[half][storage][u]=multiply_twiddle(a[j],rotation,m);
            right[half][storage][u]=multiply_twiddle(b[j],rotation,m);
        }
    }
    __syncthreads();
    for(unsigned storage=warp;storage<S;storage+=8){
        uint32_t a[1]={left[half][storage][lane]},b[1]={right[half][storage][lane]};
        fft<R,false,S>(a,table,m,lane);fft<R,false,S>(b,table,m,lane);
        a[0]=multiply(a[0],b[0],m);
        fft<R,true,S>(a,table,m,lane);
        unsigned freq=__brev(storage)>>(32-(S==64?6:5));
        left[half][storage][lane]=multiply_twiddle(a[0],table[(J-lane*freq)&(J-1)],m);
    }
    __syncthreads();
    for(unsigned u=warp;u<R;u+=8){
        uint32_t v[S/32];
        #pragma unroll
        for(unsigned j=0;j<S/32;++j)v[j]=left[half][j*32+lane][u];
        fft<S,true,R>(v,table,m,lane);
        #pragma unroll
        for(unsigned j=0;j<S/32;++j)left[half][j*32+lane][u]=v[j];
    }
    __syncthreads();
    for(unsigned i=tid;i<2*J;i+=blockDim.x) {
        unsigned side=(i/4)%2,id=4*(i/8)+i%4,row=first_row+side;
        unsigned dst=(id/4)*K*4+row*4+id%4;
        output[size_t(plane)*N+dst]=multiply(left[side][id/R][id%R],
            inverse_phases[(size_t(prime)*K+row)*J+id],m);
    }
}
}
