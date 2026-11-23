// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "warp-fft.cuh"
#include "odd-fft.cuh"
#include "tile-layout.cuh"
namespace bytecrt::warp_ntt {
template<unsigned N,unsigned P>
__global__ void middle(const uint32_t* input,uint32_t* output,const uint32_t* inverse_phases,
                       const Twiddle* roots,const Mod* mods){
    constexpr unsigned K=N/256;
    __shared__ uint32_t tile[8][260];
    unsigned lane=threadIdx.x&31,warp=threadIdx.x>>5,first=(blockIdx.x/7)*8,row=first+warp;
    unsigned prime=blockIdx.y, plane=prime+(blockIdx.x%7)*(P/7);const Mod m=mods[prime];const Twiddle* table=roots+size_t(prime)*(2*K+256)+2*K;
    __shared__ uint32_t right[8][260];
    // Each eight-word strip comes from adjacent frequencies of the same id.
    // Transpose it locally so every transform warp owns one frequency row.
    for (unsigned word=threadIdx.x;word<8*256;word+=blockDim.x) {
        gather_middle_word<K>(input+size_t(plane)*N, tile, first, word);
        gather_middle_word<K>(input+size_t(P+plane)*N, right, first, word);
    }
    __syncthreads();
    uint32_t a[8],b[8];
    #pragma unroll
    for(unsigned r=0;r<8;++r) {
        a[r]=tile[warp][lane+32*r];
        b[r]=right[warp][lane+32*r];
    }
    __syncthreads();
    fft<256,false>(a,table,m,lane);fft<256,false>(b,table,m,lane);
    #pragma unroll
    for(unsigned r=0;r<8;++r)a[r]=multiply(a[r],b[r],m);
    fft<256,true>(a,table,m,lane);
    #pragma unroll
    for(unsigned r=0;r<8;++r){unsigned i=lane+32*r;
        tile[warp][i]=multiply(a[r],inverse_phases[size_t(prime)*N+row*256+i],m);}
    __syncthreads();
    for(unsigned word=threadIdx.x;word<8*256;word+=256)
        output[size_t(plane)*N+size_t(word/8)*K+first+word%8]=tile[word%8][word/8];
}
}
