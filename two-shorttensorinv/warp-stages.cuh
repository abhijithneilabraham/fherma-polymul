// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "odd-fft.cuh"
#include "register-fft.cuh"
#ifdef __CUDACC__
#include "two-encode.cuh"
#endif

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
            uint32_t difference=sub(a,b,mod.p);
            v[i+D]=exponent?multiply_twiddle(difference,roots[exponent],mod):difference;
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
        unsigned dst=(id/16)*K*16+outer*16+id%16;
        output[size_t(plane)*N+dst]=multiply(value[i],inverse_phases[(size_t(prime)*K+outer)*J+id],mod);
    }
}
// Four warps share the input tile. Each warp computes eight inverse rows
// for two sets of sixteen columns and writes them to the CRT order.
template<unsigned N,unsigned P,unsigned Kpad>
__global__ __launch_bounds__(128) void inverse(const uint32_t* input,uint32_t* weighted,
        const uint32_t* matrices,const uint32_t* primes,const uint32_t* reciprocals) {
    constexpr unsigned K=32,cols=32,stride=36;
    __shared__ uint32_t source[cols][stride],output[cols][stride];
    const unsigned tid=threadIdx.x,lane=tid%32,warp=tid/32,g=lane/4,t=lane%4;
    const unsigned block=blockIdx.x,plane=blockIdx.y,prime=plane%(P/2);
    const auto* weights=matrices+size_t(prime)*K*K*4;
    for(unsigned word=tid;word<cols*K;word+=128) {
        const unsigned col=16*(word/(16*K))+word%16;
        const unsigned freq=(word/16)%K;
        source[col][freq]=input[size_t(plane)*N+block*cols*K+word];
    }
    __syncthreads();
    two_digit::Mma mma;
    two_digit::Neighbor exchange;
    const uint32_t p=primes[prime],mu=reciprocals[prime];
    #pragma unroll 1
    for(unsigned pair=0;pair<4;++pair) {
        const unsigned group=4*warp+pair;
        int32_t accumulators[2][4]{};
        #pragma unroll
        for(unsigned k=0;k<K;k+=8) {
            const auto* column=weights+(8*group+g)*K+k+t;
            const uint32_t b0=column[0],b1=column[4];
            #pragma unroll
            for(unsigned half=0;half<2;++half) {
                unsigned lo=16*half+g,hi=lo+8;
                mma(accumulators[half],source[lo][k+t],source[hi][k+t],
                    source[lo][k+t+4],source[hi][k+t+4],b0,b1);
            }
        }
        const bool upper=(lane&1)!=0;
        const unsigned row=2*group+t/2;
        #pragma unroll
        for(unsigned half=0;half<2;++half) {
            const auto& c=accumulators[half];
            const auto x=static_cast<int32_t>(exchange(upper?c[0]:c[2]));
            const auto y=static_cast<int32_t>(exchange(upper?c[1]:c[3]));
            const int32_t dot[4]={upper?x:c[0],upper?y:c[1],upper?c[2]:x,upper?c[3]:y};
            output[16*half+g+8*unsigned(upper)][row]=two_digit::reduce_dot(dot,p,mu);
        }
    }
    __syncthreads();
    for(unsigned word=tid;word<cols*K;word+=128) {
        const unsigned col=word/K,row=word%K,id=cols*block+col;
        weighted[(size_t(id)*(Kpad/4)+plane)*K+row]=output[col][row];
    }
}
} // namespace bytecrt::warp_ntt
