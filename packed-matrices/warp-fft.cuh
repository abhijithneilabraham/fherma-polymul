// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "warp-mod.hpp"
namespace bytecrt::warp_ntt {
template<unsigned Len,unsigned Distance,bool Inverse>
__device__ __forceinline__ void step(uint32_t (&v)[Len/32],const uint32_t* roots,
                                    const Mod& m,unsigned lane){
    static_assert(Len==128 || Len==256,"one warp and four/eight register coefficients");
    if constexpr(Distance>=32){
        #pragma unroll
        for(unsigned r=0;r<Len/32;++r){
            if(!(r&(Distance/32))){
                unsigned index=((lane+32*r)&(Distance-1))*(Len/(2*Distance));
                uint32_t twiddle=roots[Inverse?((Len-index)&(Len-1)):index];
                uint32_t a=v[r],b=v[r+Distance/32];
                if constexpr(Inverse){b=multiply(b,twiddle,m);v[r]=add(a,b,m.p);v[r+Distance/32]=sub(a,b,m.p);}
                else {v[r]=add(a,b,m.p);v[r+Distance/32]=multiply(sub(a,b,m.p),twiddle,m);}
            }
        }
    } else {
        unsigned index=(lane&(Distance-1))*(Len/(2*Distance));
        uint32_t twiddle=roots[Inverse?((Len-index)&(Len-1)):index];
        #pragma unroll
        for(unsigned r=0;r<Len/32;++r){
            uint32_t value=v[r];
            if constexpr(Inverse){if(lane&Distance)value=multiply(value,twiddle,m);}
            uint32_t partner=__shfl_xor_sync(0xffffffffu,value,Distance);
            if constexpr(Inverse)v[r]=(lane&Distance)?sub(partner,value,m.p):add(value,partner,m.p);
            else v[r]=(lane&Distance)?multiply(sub(partner,value,m.p),twiddle,m):add(value,partner,m.p);
        }
    }
}
template<unsigned Len,bool Inverse,unsigned Distance=Inverse?1:Len/2>
__device__ __forceinline__ void fft(uint32_t (&v)[Len/32],const uint32_t* roots,
                                  const Mod& m,unsigned lane){
    step<Len,Distance,Inverse>(v,roots,m,lane);
    if constexpr(Inverse){if constexpr(Distance<Len/2)fft<Len,Inverse,Distance*2>(v,roots,m,lane);}
    else {if constexpr(Distance>1)fft<Len,Inverse,Distance/2>(v,roots,m,lane);}
}
} // namespace bytecrt::warp_ntt
