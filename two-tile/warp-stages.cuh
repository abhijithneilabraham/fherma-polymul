// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "register-fft.cuh"
#include "odd-fft.cuh"
#include "tile-layout.cuh"
namespace bytecrt::warp_ntt {
template<unsigned N,unsigned P>
__global__ void middle(const uint32_t* input,uint32_t* output,const uint32_t* inverse_phases,
                       const Twiddle* roots,const Mod* mods){
    constexpr unsigned K=N/256;
    __shared__ uint32_t tile[8][260];
    unsigned lane=threadIdx.x&31,warp=threadIdx.x>>5,first=blockIdx.x*8,row=first+warp;
    unsigned prime=blockIdx.y;const Mod m=mods[prime];const Twiddle* table=roots+size_t(prime)*(2*K+256)+2*K;
    __shared__ uint32_t right[8][260];
    // Each eight-word strip comes from adjacent frequencies of the same id.
    // Transpose it locally so every transform warp owns one frequency row.
    for (unsigned word=threadIdx.x;word<8*256;word+=blockDim.x) {
        gather_middle_word<K>(input+size_t(prime)*N, tile, first, word);
        gather_middle_word<K>(input+size_t(P+prime)*N, right, first, word);
    }
    __syncthreads();
    uint32_t a[8],b[8];
    #pragma unroll
    for(unsigned r=0;r<8;++r) {
        a[r]=tile[warp][lane+32*r];
        b[r]=right[warp][lane+32*r];
    }
    __syncthreads();
    middle_fft<false>(a,table,m,lane);middle_fft<false>(b,table,m,lane);
    #pragma unroll
    for(unsigned r=0;r<8;++r)a[r]=multiply(a[r],b[r],m);
    middle_fft<true>(a,table,m,lane);
    #pragma unroll
    for(unsigned r=0;r<8;++r){unsigned i=lane+32*r;
        tile[warp][i]=multiply(a[r],inverse_phases[size_t(prime)*N+row*256+i],m);}
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
    odd_fft<K,true>(v,roots+size_t(prime)*(2*K+256)+K,m,lane);
    #pragma unroll
    for(unsigned r=0;r<K/32;++r){unsigned i=lane+32*r;
        weighted[(size_t(id)*(Kpad/4)+prime)*K+i]=v[r];}
}
} // namespace bytecrt::warp_ntt
