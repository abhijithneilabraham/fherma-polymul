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
            if(exponent)b=lazy_twiddle(b,roots[S*256-exponent],mod);
            v[i]=sum_twice(a,b,mod.p);
            v[i+D]=diff_twice(a,b,mod.p);
        } else {
            v[i]=sum_twice(a,b,mod.p);
            uint32_t difference=diff_twice(a,b,mod.p);
            v[i+D]=exponent?lazy_twiddle(difference,roots[exponent],mod):difference;
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
            a[i]=lazy_twiddle(a[i],rotation,mod);
            b[i]=lazy_twiddle(b[i],rotation,mod);
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
            if(frequency)x[i]=lazy_twiddle(x[i],table[(J-col*frequency)&(J-1)],mod);
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
// Four columns share a warp. Small distances exchange lanes and the
// last two stages operate on the four register rows of each lane.
template<unsigned Distance=1>
__device__ __forceinline__ void inverse_four(uint32_t (&v)[4], unsigned row,
                                             const Twiddle* roots, const Mod& mod) {
    constexpr unsigned groups=32/(2*Distance);
    #pragma unroll
    for(unsigned part=0;part<4;++part) {
        unsigned index=row+8*part;
        const Twiddle root=roots[groups+index/(2*Distance)];
        if constexpr(Distance>=8) {
            if(part&(Distance/8))continue;
            uint32_t a=v[part],b=v[part+Distance/8];
            v[part]=add(a,b,mod.p);
            v[part+Distance/8]=multiply_twiddle(sub(a,b,mod.p),root,mod);
        } else {
            uint32_t value=v[part],peer=__shfl_xor_sync(0xffffffffu,v[part],4*Distance);
            v[part]=(row&Distance)?multiply_twiddle(sub(peer,value,mod.p),root,mod):add(value,peer,mod.p);
        }
    }
    if constexpr(Distance<16)inverse_four<2*Distance>(v,row,roots,mod);
}

template<unsigned N,unsigned P,unsigned Kpad>
__global__ void inverse(const uint32_t* input,uint32_t* weighted,
                        const Twiddle* roots,const Mod* mods) {
    constexpr unsigned K=32,J=N/K;
    unsigned lane=threadIdx.x%32,warp=threadIdx.x/32,col=lane%4,row=lane/4;
    unsigned group=blockIdx.x,plane=blockIdx.y*8+warp;
    if(plane>=P)return;
    unsigned prime=plane%(P/2);
    uint32_t values[4];
    #pragma unroll
    for(unsigned part=0;part<4;++part)
        values[part]=input[size_t(plane)*N+group*K*4+4*(row+8*part)+col];
    inverse_four(values,row,roots+size_t(prime)*(2*K+J)+K,mods[prime]);
    unsigned id=4*group+col;
    #pragma unroll
    for(unsigned part=0;part<4;++part)
        weighted[(size_t(id)*(Kpad/4)+plane)*K+row+8*part]=values[part];
}
} // namespace bytecrt::warp_ntt
