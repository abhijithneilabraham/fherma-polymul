// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "odd-fft.cuh"
#include "register-fft.cuh"

namespace bytecrt::warp_ntt {
// The short dimension is entirely local to one thread. Its roots are
// powers of the 256th power of the primitive length-J root.
template<unsigned S,bool Inverse,unsigned D=Inverse?1:S/2>
__device__ __forceinline__ void column_fft(uint32_t (&v)[S],const Twiddle* roots,const Mod& mod) {
    #pragma unroll
    for(unsigned i=0;i<S;++i) {
        if(i&D)continue;
        const unsigned exponent=(i&(D-1))*(S*256/(2*D));
        uint32_t a=v[i],b=v[i+D];
        if constexpr(Inverse) {
            if(exponent)b=multiply_twiddle(b,roots[S*256-exponent],mod);
            v[i]=add(a,b,mod.p);
            v[i+D]=sub(a,b,mod.p);
        } else {
            v[i]=add(a,b,mod.p);
            v[i+D]=exponent?scaled_difference(a,b,roots[exponent],mod):sub(a,b,mod.p);
        }
    }
    if constexpr(Inverse && D<S/2)column_fft<S,true,2*D>(v,roots,mod);
    if constexpr(!Inverse && D>1)column_fft<S,false,D/2>(v,roots,mod);
}

template<unsigned N,unsigned P>
__global__ __launch_bounds__(512) void middle(const uint32_t* input,uint32_t* output,
        const uint32_t* inverse_phases,const Twiddle* roots,const Mod* mods) {
    constexpr unsigned K=32,J=N/K,R=256,S=J/R;
    static_assert(S==4 || S==8);
    unsigned tid=threadIdx.x,lane=tid%32,first=2*blockIdx.x,plane=blockIdx.y,prime=plane%(P/2);
    const Mod mod=mods[prime];
    const Twiddle* table=roots+size_t(prime)*(2*K+J)+2*K;
    __shared__ uint32_t left[2][S][R],right[2][S][R];
    unsigned side=(tid/4)%2,u=4*(tid/8)+tid%4,bank=u^(16*side);
    uint32_t a[S],b[S];
    #pragma unroll
    for(unsigned i=0;i<S;++i) {
        size_t offset=size_t(plane)*N+size_t(first)*J+2*R*i+tid;
        a[i]=input[offset];b[i]=input[size_t(P)*N+offset];
    }
    column_fft<S,false>(a,table,mod);
    column_fft<S,false>(b,table,mod);
    #pragma unroll
    for(unsigned i=0;i<S;++i) {
        unsigned frequency=__brev(i)>>(32-(S==8?3:2));
        if(frequency) {
            Twiddle rotation=table[u*frequency];
            a[i]=multiply_twiddle(a[i],rotation,mod);
            b[i]=multiply_twiddle(b[i],rotation,mod);
        }
        left[side][i][bank]=a[i];right[side][i][bank]=b[i];
    }
    __syncthreads();
    unsigned half=tid/256,row=(tid%256)/32;
    if(row<S) {
        uint32_t x[8],y[8];
        #pragma unroll
        for(unsigned i=0;i<8;++i) {
            unsigned col=(32*i+lane)^(16*half);
            x[i]=left[half][row][col];y[i]=right[half][row][col];
        }
        middle_fft<false,S>(x,table,mod,lane);
        middle_fft<false,S>(y,table,mod,lane);
        #pragma unroll
        for(unsigned i=0;i<8;++i)x[i]=multiply(x[i],y[i],mod);
        middle_fft<true,S>(x,table,mod,lane);
        const unsigned frequency=__brev(row)>>(32-(S==8?3:2));
        #pragma unroll
        for(unsigned i=0;i<8;++i) {
            unsigned col=32*i+lane;
            if(frequency)x[i]=multiply_twiddle(x[i],table[(J-col*frequency)&(J-1)],mod);
            left[half][row][col^(16*half)]=x[i];
        }
    }
    __syncthreads();
    uint32_t value[S];
    #pragma unroll
    for(unsigned i=0;i<S;++i)value[i]=left[side][i][bank];
    column_fft<S,true>(value,table,mod);
    #pragma unroll
    for(unsigned i=0;i<S;++i) {
        unsigned id=i*R+u,outer=first+side;
        unsigned dst=(id/4)*K*4+outer*4+id%4;
        output[size_t(plane)*N+dst]=multiply(value[i],inverse_phases[(size_t(prime)*K+outer)*J+id],mod);
    }
}
} // namespace bytecrt::warp_ntt
