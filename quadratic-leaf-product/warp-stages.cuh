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
            uint32_t difference=sub(a,b,mod.p);
            v[i+D]=exponent?multiply_twiddle(difference,roots[exponent],mod):difference;
        }
    }
    if constexpr(Inverse && D<S/2)column_fft<S,true,2*D>(v,roots,mod);
    if constexpr(!Inverse && D>1)column_fft<S,false,D/2>(v,roots,mod);
}

// Two canonical prime-field products fit below p*2^32. They can share
// one Montgomery correction, including when the second input equals p.
__device__ __forceinline__ uint32_t dot_two(uint32_t a,uint32_t b,
        uint32_t c,uint32_t d,const Mod& m) {
    const uint64_t sum=uint64_t(a)*b+uint64_t(c)*d;
    const uint32_t low=uint32_t(sum)*m.negative_inverse;
    const uint32_t high=uint32_t((sum+uint64_t(low)*m.p)>>32);
    return high>=m.p?high-m.p:high;
}

// Product modulo z^4-1 from the two scalar evaluations and one
// product modulo z^2+1. The factor of four is the omitted inverse FFT scale.
__device__ __forceinline__ void quadratic_leaf(uint32_t* a,const uint32_t* b,const Mod& m) {
    const uint32_t p=m.p;
    uint32_t ae=add(a[0],a[2],p), ao=add(a[1],a[3],p);
    uint32_t be=add(b[0],b[2],p), bo=add(b[1],b[3],p);
    uint32_t at_one=multiply(add(ae,ao,p),add(be,bo,p),m);
    uint32_t at_minus_one=multiply(sub(ae,ao,p),sub(be,bo,p),m);
    uint32_t u=sub(a[0],a[2],p),v=sub(a[1],a[3],p);
    uint32_t x=sub(b[0],b[2],p),y=sub(b[1],b[3],p);
    uint32_t real=dot_two(u,x,p-v,y,m);
    uint32_t imag=dot_two(u,y,v,x,m);
    real=add(real,real,p);imag=add(imag,imag,p);
    uint32_t even=add(at_one,at_minus_one,p),odd=sub(at_one,at_minus_one,p);
    a[0]=add(even,real,p);a[2]=sub(even,real,p);
    a[1]=add(odd,imag,p);a[3]=sub(odd,imag,p);
}

template<unsigned N,unsigned P>
__global__ __launch_bounds__(512) void middle(const uint32_t* input,uint32_t* output,
        const uint32_t* phases,const Twiddle* roots,const Mod* mods) {
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
        middle_fft<false,S,true>(x,table,mod,lane);
        middle_fft<false,S,true>(y,table,mod,lane);
        #pragma unroll
        for(unsigned i=0;i<8;i+=4)quadratic_leaf(x+i,y+i,mod);
        middle_fft<true,S,true>(x,table,mod,lane);
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
        output[size_t(plane)*N+dst]=multiply(value[i],phases[phase_address(prime,N,J-1-id,outer)],mod);
    }
}
} // namespace bytecrt::warp_ntt
