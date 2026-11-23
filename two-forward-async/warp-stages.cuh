// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "register-fft.cuh"
#include "odd-fft.cuh"
namespace bytecrt::warp_ntt {
// The warp swaps the low bits of the lane and register numbers once.
template<unsigned K>
__device__ __forceinline__ unsigned natural_row(unsigned lane,unsigned reg) {
    constexpr unsigned mask=K/32-1;
    return 32*(lane&mask)+(lane&~mask)+reg;
}
template<unsigned K>
__device__ __forceinline__ unsigned tile_row(unsigned row) {
    return row ^ (row/32);
}
template<unsigned K,unsigned D=K/2>
__device__ __forceinline__ void forward_registers(uint32_t (&v)[K/32],const Twiddle* roots,
                                                  const Mod& mod,unsigned lane) {
    constexpr unsigned units=K/32;
    if constexpr(D>=units) {
        odd_step<K,D,false>(v,roots,mod,lane);
    } else {
        if constexpr(D==units/2) {
            transpose_bit<1>(v,lane);
            transpose_bit<2>(v,lane);
            if constexpr(units==8)transpose_bit<4>(v,lane);
        }
        #pragma unroll
        for(unsigned low=0;low<units;++low) {
            if(low&D)continue;
            unsigned row=natural_row<K>(lane,low);
            uint32_t a=v[low],b=multiply_twiddle(v[low+D],roots[K/(2*D)+row/(2*D)],mod);
            v[low]=add(a,b,mod.p);
            v[low+D]=sub(a,b,mod.p);
        }
    }
    if constexpr(D>1)forward_registers<K,D/2>(v,roots,mod,lane);
}
template<unsigned N,unsigned P>
__global__ void forward(const uint32_t* input,uint32_t* output,
                        const uint32_t* phases,const Twiddle* roots,const Mod* mods){
    constexpr unsigned K=N/256;
    __shared__ uint32_t tile[8][K+4];
    __shared__ __align__(16) uint32_t phase_tile[8*K];
    unsigned lane=threadIdx.x&31,warp=threadIdx.x>>5,first=blockIdx.x*8,id=first+warp;
    unsigned plane=blockIdx.y,prime=plane%P;const Mod m=mods[prime];
    constexpr unsigned copy_words=4;
    constexpr unsigned threads=256;
    static_assert((8*K)%(copy_words*threads)==0);
    #pragma unroll
    for(unsigned part=0;part<8*K/(copy_words*threads);++part) {
        unsigned word=copy_words*(threadIdx.x+threads*part);
        unsigned offset=(word/8)*256+first+word%8;
        const auto* source=phases+size_t(prime)*N+offset;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        const auto dest=static_cast<uint32_t>(__cvta_generic_to_shared(phase_tile+word));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                     :: "r"(dest), "l"(source) : "memory");
#else
        for(unsigned entry=0;entry<4;++entry)phase_tile[word+entry]=source[entry];
#endif
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
    uint32_t v[K/32];
    #pragma unroll
    for(unsigned r=0;r<K/32;++r){unsigned i=lane+32*r;
        v[r]=input[size_t(plane)*N+id*K+i];}
    forward_registers<K>(v,roots+size_t(prime)*(2*K+256),m,lane);
    #pragma unroll
    for(unsigned r=0;r<K/32;++r)
        tile[warp][tile_row<K>(natural_row<K>(lane,r))]=v[r];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
    __syncthreads();
    for(unsigned word=threadIdx.x;word<8*K;word+=256) {
        unsigned offset=(word/8)*256+first+word%8;
        output[size_t(plane)*N+offset]=multiply(tile[word%8][tile_row<K>(word/8)],
                                               phase_tile[word],m);
    }
}
template<unsigned N,unsigned P>
__global__ void middle(const uint32_t* input,uint32_t* output,const uint32_t* inverse_phases,
                       const Twiddle* roots,const Mod* mods){
    constexpr unsigned K=N/256;
    __shared__ uint32_t tile[8][260];
    unsigned lane=threadIdx.x&31,warp=threadIdx.x>>5,first=blockIdx.x*8,row=first+warp;
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
