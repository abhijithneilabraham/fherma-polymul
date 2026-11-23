// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "warp-mod.hpp"
namespace bytecrt::warp_ntt {
template<unsigned Len,unsigned Distance,bool Inverse,unsigned Stride>
__device__ __forceinline__ void step(uint32_t (&v)[Len/32],const Twiddle* roots,
                                    const Mod& m,unsigned lane){
    static_assert(Len==32 || Len==64 || Len==256);
    if constexpr(Distance>=32){
        #pragma unroll
        for(unsigned r=0;r<Len/32;++r){
            if(!(r&(Distance/32))){
                unsigned index=((lane+32*r)&(Distance-1))*(Len/(2*Distance));
                Twiddle twiddle=roots[Stride*(Inverse?((Len-index)&(Len-1)):index)];
                uint32_t a=v[r],b=v[r+Distance/32];
                if constexpr(Inverse){b=lazy_twiddle(b,twiddle,m);v[r]=sum_twice(a,b,m.p);v[r+Distance/32]=diff_twice(a,b,m.p);}
                else {v[r]=sum_twice(a,b,m.p);v[r+Distance/32]=lazy_twiddle(diff_twice(a,b,m.p),twiddle,m);}
            }
        }
    } else {
        unsigned index=(lane&(Distance-1))*(Len/(2*Distance));
        Twiddle twiddle=roots[Stride*(Inverse?((Len-index)&(Len-1)):index)];
        #pragma unroll
        for(unsigned r=0;r<Len/32;++r){
            uint32_t value=v[r];
            if constexpr(Inverse){if(lane&Distance)value=lazy_twiddle(value,twiddle,m);}
            uint32_t partner=__shfl_xor_sync(0xffffffffu,value,Distance);
            if constexpr(Inverse)v[r]=(lane&Distance)?diff_twice(partner,value,m.p):sum_twice(value,partner,m.p);
            else v[r]=(lane&Distance)?lazy_twiddle(diff_twice(partner,value,m.p),twiddle,m):sum_twice(value,partner,m.p);
        }
    }
}
template<unsigned Len,bool Inverse,unsigned Stride,unsigned Distance=Inverse?1:Len/2>
__device__ __forceinline__ void fft(uint32_t (&v)[Len/32],const Twiddle* roots,
                                  const Mod& m,unsigned lane){
    step<Len,Distance,Inverse,Stride>(v,roots,m,lane);
    if constexpr(Inverse){if constexpr(Distance<Len/2)fft<Len,Inverse,Stride,Distance*2>(v,roots,m,lane);}
    else {if constexpr(Distance>1)fft<Len,Inverse,Stride,Distance/2>(v,roots,m,lane);}
}
} // namespace bytecrt::warp_ntt
