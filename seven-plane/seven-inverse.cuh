#pragma once
// SPDX-License-Identifier: Apache-2.0
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
#include "inverse-field.hpp"
#include "odd-fft.cuh"

namespace seven_inverse {
using namespace rader7;
// A block owns all seven evaluations of one prime at a single outer index.
// The inverse polynomial phase upstream supplies R^2/N and the CRT cofactor inverse.
template<unsigned N,unsigned Count>
__global__ void inverse(const U* input,U* weighted,const Twiddle* roots,const Field* fields) {
    constexpr unsigned K=N/256;
    unsigned lane=threadIdx.x%32,warp=threadIdx.x/32;
    unsigned prime=blockIdx.y, id=blockIdx.x;
    const Field& f=fields[prime];
    __shared__ U transformed[7][K];
    if(warp<7) {
        unsigned plane=warp*Count+prime;
        U v[K/32];
        #pragma unroll
        for(unsigned j=0;j<K/32;++j) v[j]=input[size_t(plane)*N+id*K+32*j+lane];
        odd_fft<K,true>(v,roots+size_t(prime)*(2*K+256)+K,f.mod,lane);
        #pragma unroll
        for(unsigned j=0;j<K/32;++j) transformed[warp][32*j+lane]=v[j];
    }
    __syncthreads();
    unsigned coefficient=threadIdx.x;
    if(coefficient<K) {
        U x[7],y[7];
        #pragma unroll
        for(unsigned d=0;d<7;++d)x[d]=transformed[d][coefficient];
        transform(x,f.inverse,f.mod,y);
        #pragma unroll
        for(unsigned d=0;d<7;++d){
            unsigned plane=d*Count+prime;
            weighted[(size_t(id)*7*Count+plane)*K+coefficient]=multiply_twiddle(y[d],f.recovery[d],f.mod);
        }
    }
}
}
