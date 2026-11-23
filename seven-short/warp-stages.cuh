// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "warp-fft.cuh"
#include "odd-fft.cuh"

namespace bytecrt::warp_ntt {
template<unsigned N,unsigned P>
__global__ void forward(const uint32_t* input,uint32_t* output,
                        const uint32_t* phases,const Twiddle* roots,const Mod* mods){
    constexpr unsigned K=32,J=N/K;
    __shared__ uint32_t tile[8][K+4];
    unsigned lane=threadIdx.x&31,warp=threadIdx.x>>5,first=(blockIdx.x/14)*8,id=first+warp;
    unsigned prime=blockIdx.y,plane=prime+(blockIdx.x%14)*(P/7);const Mod m=mods[prime];
    uint32_t v[K/32];
    #pragma unroll
    for(unsigned r=0;r<K/32;++r){unsigned i=lane+32*r;
        v[r]=input[size_t(plane)*N+id*K+i];}
    odd_fft<K,false>(v,roots+size_t(prime)*(2*K+J),m,lane);
    #pragma unroll
    for(unsigned r=0;r<K/32;++r){unsigned i=lane+32*r;
        tile[warp][i]=multiply(v[r],phases[size_t(prime)*N+id*K+i],m);}
    __syncthreads();
    for(unsigned word=threadIdx.x;word<8*K;word+=256)
        output[size_t(plane)*N+size_t(word/8)*J+first+word%8]=tile[word%8][word/8];
}
// A 32-point outer transform leaves a J=N/32 cyclic product. Split J into
// 32 columns and S rows so each local transform uses at most two registers
// per lane. The row/column exchanges stay inside the block.
template<unsigned N,unsigned P>
__global__ void middle(const uint32_t* input,uint32_t* output,const uint32_t* inverse_phases,
                       const Twiddle* roots,const Mod* mods) {
    constexpr unsigned K=32,J=N/K,R=32,S=J/R;
    unsigned row=blockIdx.x%K, plane=blockIdx.y, prime=plane%(P/7);
    unsigned tid=threadIdx.x,lane=tid%32,warp=tid/32;
    const Mod m=mods[prime];
    const Twiddle* table=roots+size_t(prime)*(2*K+J)+2*K;
    __shared__ uint32_t left[S][R+1],right[S][R+1];
    for(unsigned k=tid;k<J;k+=blockDim.x){
        left[k/R][k%R]=input[(size_t(plane)*K+row)*J+k];
        right[k/R][k%R]=input[(size_t(P+plane)*K+row)*J+k];
    }
    __syncthreads();
    for(unsigned u=warp;u<R;u+=blockDim.x/32){
        uint32_t a[S/32],b[S/32];
        #pragma unroll
        for(unsigned j=0;j<S/32;++j){a[j]=left[j*32+lane][u];b[j]=right[j*32+lane][u];}
        fft<S,false,R>(a,table,m,lane);fft<S,false,R>(b,table,m,lane);
        #pragma unroll
        for(unsigned j=0;j<S/32;++j){
            unsigned storage=j*32+lane, freq=__brev(storage)>>(32-(S==64?6:5));
            Twiddle rotation=table[u*freq];
            left[storage][u]=multiply_twiddle(a[j],rotation,m);
            right[storage][u]=multiply_twiddle(b[j],rotation,m);
        }
    }
    __syncthreads();
    for(unsigned storage=warp;storage<S;storage+=blockDim.x/32){
        uint32_t a[1]={left[storage][lane]},b[1]={right[storage][lane]};
        fft<R,false,S>(a,table,m,lane);fft<R,false,S>(b,table,m,lane);
        a[0]=multiply(a[0],b[0],m);
        fft<R,true,S>(a,table,m,lane);
        unsigned freq=__brev(storage)>>(32-(S==64?6:5));
        left[storage][lane]=multiply_twiddle(a[0],table[(J-lane*freq)&(J-1)],m);
    }
    __syncthreads();
    for(unsigned u=warp;u<R;u+=blockDim.x/32){
        uint32_t v[S/32];
        #pragma unroll
        for(unsigned j=0;j<S/32;++j)v[j]=left[j*32+lane][u];
        fft<S,true,R>(v,table,m,lane);
        #pragma unroll
        for(unsigned j=0;j<S/32;++j)left[j*32+lane][u]=v[j];
    }
    __syncthreads();
    for(unsigned id=tid;id<J;id+=blockDim.x){
        unsigned dest=(id/8)*K*8+row*8+id%8;
        output[size_t(plane)*N+dest]=multiply(left[id/R][id%R],inverse_phases[(size_t(prime)*K+row)*J+id],m);
    }
}
}
