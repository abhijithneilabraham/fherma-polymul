// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "register-fft.cuh"
#include "split-inverse.cuh"
namespace bytecrt::warp_ntt {
template<unsigned N,unsigned P>
__global__ void forward(const uint32_t* input,uint32_t* output,
                        const uint32_t* phases,const Twiddle* roots,const Mod* mods){
    constexpr unsigned K=N/256;
    __shared__ uint32_t tile[8][K+4];
    unsigned lane=threadIdx.x&31,warp=threadIdx.x>>5,first=blockIdx.x*8,id=first+warp;
    unsigned plane=blockIdx.y,prime=plane%P;const Mod m=mods[prime];
    uint32_t v[K/32];
    #pragma unroll
    for(unsigned r=0;r<K/32;++r){unsigned i=lane+32*r;
        v[r]=input[size_t(plane)*N+id*K+i];}
    odd_fft<K,false>(v,roots+size_t(prime)*(2*K+256),m,lane);
    #pragma unroll
    for(unsigned r=0;r<K/32;++r){unsigned i=lane+32*r;
        tile[warp][i]=multiply(v[r],phases[size_t(prime)*N+id*K+i],m);}
    __syncthreads();
    for(unsigned word=threadIdx.x;word<8*K;word+=256)
        output[size_t(plane)*N+size_t(word/8)*256+first+word%8]=tile[word%8][word/8];
}
template<unsigned N,unsigned P>
__global__ __launch_bounds__(1024) void middle(const uint32_t* input,uint32_t* output,const uint32_t* inverse_phases,
                       const Twiddle* roots,const Mod* mods){
    constexpr unsigned K=N/256;
    __shared__ uint32_t tile[32][260];
    unsigned lane=threadIdx.x&31,warp=threadIdx.x>>5,first=blockIdx.x*32,row=first+warp;
    unsigned prime=blockIdx.y;const Mod m=mods[prime];const Twiddle* table=roots+size_t(prime)*(2*K+256)+2*K;
    uint32_t a[8],b[8];
    #pragma unroll
    for(unsigned r=0;r<8;++r){unsigned i=lane+32*r;
        a[r]=input[size_t(prime)*N+row*256+i];b[r]=input[size_t(P+prime)*N+row*256+i];}
    middle_fft<false>(a,table,m,lane);middle_fft<false>(b,table,m,lane);
    #pragma unroll
    for(unsigned r=0;r<8;++r)a[r]=multiply(a[r],b[r],m);
    middle_fft<true>(a,table,m,lane);
    #pragma unroll
    for(unsigned r=0;r<8;++r){unsigned i=lane+32*r;
        tile[warp][i]=multiply(a[r],inverse_phases[size_t(prime)*N+row*256+i],m);}
    __syncthreads();
    constexpr unsigned group=first_group<K>;
    for(unsigned task=threadIdx.x;task<(32/group)*256;task+=blockDim.x) {
        unsigned id=task%256,base=(task/256)*group;
        uint32_t values[group];
        #pragma unroll
        for(unsigned r=0;r<group;++r)values[r]=tile[base+r][id];
        inverse_prefix<K>(values,first+base,roots+size_t(prime)*(2*K+256)+K,m);
        #pragma unroll
        for(unsigned r=0;r<group;++r)tile[base+r][id]=values[r];
    }
    __syncthreads();
    // Eight adjacent words hold two columns and four coarse rows. The
    // reconstruction warp reads the same sectors from both columns.
    for(unsigned i=threadIdx.x;i<32*256;i+=blockDim.x) {
        unsigned id=2*((i/8)%128)+(i%8)/4;
        unsigned low=(i/1024)%group,coarse=4*(i/(1024*group))+i%4;
        output[size_t(prime)*N+partial_index(id,low,first/group+coarse)]=tile[coarse*group+low][id];
    }
}
} // namespace bytecrt::warp_ntt
