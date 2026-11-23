// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "shared-roots.cuh"
namespace bytecrt::warp_ntt {
template<unsigned Len,unsigned Distance,bool Inverse,class RootTable>
__device__ __forceinline__ void step(uint32_t (&v)[Len/32],const RootTable& roots,
                                    const Mod& m,unsigned lane){
    static_assert(Len==128 || Len==256,"one warp and four/eight register coefficients");
    if constexpr(Distance>=32){
        #pragma unroll
        for(unsigned r=0;r<Len/32;++r){
            if(!(r&(Distance/32))){
                unsigned index=((lane+32*r)&(Distance-1))*(Len/(2*Distance));
                Twiddle twiddle=read_root<Len,Inverse>(roots,index);
                uint32_t a=v[r],b=v[r+Distance/32];
                if constexpr(Inverse){b=multiply_twiddle(b,twiddle,m);v[r]=add(a,b,m.p);v[r+Distance/32]=sub(a,b,m.p);}
                else {v[r]=add(a,b,m.p);v[r+Distance/32]=multiply_twiddle(sub(a,b,m.p),twiddle,m);}
            }
        }
    } else {
        unsigned index=(lane&(Distance-1))*(Len/(2*Distance));
        Twiddle twiddle=read_root<Len,Inverse>(roots,index);
        #pragma unroll
        for(unsigned r=0;r<Len/32;++r){
            uint32_t value=v[r];
            if constexpr(Inverse){if(lane&Distance)value=multiply_twiddle(value,twiddle,m);}
            uint32_t partner=__shfl_xor_sync(0xffffffffu,value,Distance);
            if constexpr(Inverse)v[r]=(lane&Distance)?sub(partner,value,m.p):add(value,partner,m.p);
            else v[r]=(lane&Distance)?multiply_twiddle(sub(partner,value,m.p),twiddle,m):add(value,partner,m.p);
        }
    }
}
template<unsigned Len,bool Inverse,unsigned Distance=Inverse?1:Len/2,class RootTable>
__device__ __forceinline__ void fft(uint32_t (&v)[Len/32],const RootTable& roots,
                                  const Mod& m,unsigned lane){
    step<Len,Distance,Inverse>(v,roots,m,lane);
    if constexpr(Inverse){if constexpr(Distance<Len/2)fft<Len,Inverse,Distance*2>(v,roots,m,lane);}
    else {if constexpr(Distance>1)fft<Len,Inverse,Distance/2>(v,roots,m,lane);}
}
} // namespace bytecrt::warp_ntt
