// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "warp-fft.cuh"
#include "odd-fft.cuh"
namespace bytecrt::warp_ntt {
template<unsigned N,unsigned P>
__global__ void forward(const uint32_t* input,uint32_t* output,
                        const Twiddle* roots,const Mod* mods){
    constexpr unsigned K=N/256;
    __shared__ uint32_t tile[8][K+4];
    unsigned lane=threadIdx.x&31,warp=threadIdx.x>>5,first=blockIdx.x*8,id=first+warp;
    unsigned plane=blockIdx.y,prime=plane%P;const Mod m=mods[prime];
    uint32_t v[K/32];
    #pragma unroll
    for(unsigned r=0;r<K/32;++r){unsigned i=lane+32*r;
        v[r]=input[size_t(plane)*N+id*K+i];}
    odd_fft<K,false>(v,roots+size_t(prime)*(2*K+N),m,lane);
    #pragma unroll
    for(unsigned r=0;r<K/32;++r){unsigned i=lane+32*r;
        tile[warp][i]=v[r];}
    __syncthreads();
    for(unsigned word=threadIdx.x;word<8*K;word+=256)
        output[size_t(plane)*N+size_t(word/8)*256+first+word%8]=tile[word%8][word/8];
}
template<unsigned N,unsigned P>
__global__ void middle(const uint32_t* input,uint32_t* output,
                       const Twiddle* roots,const Mod* mods){
    constexpr unsigned K=N/256;
    __shared__ uint32_t tile[8][260];
    unsigned lane=threadIdx.x&31,warp=threadIdx.x>>5,first=blockIdx.x*8,row=first+warp;
    unsigned prime=blockIdx.y;const Mod m=mods[prime];const Twiddle* table=roots+size_t(prime)*(2*K+N)+2*K;
    const Twiddle* forward_roots=table+256*row;
    const Twiddle* inverse_roots=table+256*(K-1-row);
    uint32_t a[8],b[8];
    #pragma unroll
    for(unsigned r=0;r<8;++r){unsigned i=lane+32*r;
        a[r]=input[size_t(prime)*N+row*256+i];b[r]=input[size_t(P+prime)*N+row*256+i];}
    odd_fft<256,false>(a,forward_roots,m,lane);odd_fft<256,false>(b,forward_roots,m,lane);
    #pragma unroll
    for(unsigned r=0;r<8;++r)a[r]=multiply(a[r],b[r],m);
    odd_fft<256,true,true>(a,inverse_roots,m,lane);
    #pragma unroll
    for(unsigned r=0;r<8;++r){unsigned i=lane+32*r;
        tile[warp][i]=a[r];}
    __syncthreads();
    for(unsigned word=threadIdx.x;word<8*256;word+=256)
        output[size_t(prime)*N+size_t(word/8)*K+first+word%8]=tile[word%8][word/8];
}
template<unsigned N,unsigned P,unsigned Kpad>
__global__ void inverse(const uint32_t* input,uint32_t* weighted,
                        const Twiddle* roots,const Mod* mods){
    constexpr unsigned K=N/256;
    unsigned lane=threadIdx.x&31,warp=threadIdx.x>>5,id=blockIdx.x,prime=blockIdx.y*8+warp;
    if(prime>=P)return; // Whole warp exits; all active warps retain all 32 lanes.
    const Mod m=mods[prime];uint32_t v[K/32];
    #pragma unroll
    for(unsigned r=0;r<K/32;++r)v[r]=input[size_t(prime)*N+id*K+lane+32*r];
    odd_fft<K,true>(v,roots+size_t(prime)*(2*K+N)+K,m,lane);
    #pragma unroll
    for(unsigned r=0;r<K/32;++r){unsigned i=lane+32*r;
        weighted[(size_t(id)*(Kpad/4)+prime)*K+i]=multiply(v[r],m.normalize,m);}
}
} // namespace bytecrt::warp_ntt
