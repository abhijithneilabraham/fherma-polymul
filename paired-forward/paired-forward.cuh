// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "warp-mod.hpp"
namespace bytecrt::warp_ntt {
template<unsigned Len,unsigned D>
__device__ __forceinline__ void paired_step(uint32_t (&v)[2][Len/32],
        const Twiddle* roots,const Mod& m,unsigned lane) {
    #pragma unroll
    for(unsigned r=0;r<Len/32;++r) {
        if constexpr(D>=32) if(r&(D/32)) continue;
        Twiddle w=roots[Len/(2*D)+(lane+32*r)/(2*D)];
        #pragma unroll
        for(unsigned side=0;side<2;++side) {
            uint32_t a=v[side][r];
            if constexpr(D>=32) {
                uint32_t b=multiply_twiddle(v[side][r+D/32],w,m);
                v[side][r]=add(a,b,m.p);
                v[side][r+D/32]=sub(a,b,m.p);
            } else {
                bool upper=(lane&D)!=0;
                if(upper) a=multiply_twiddle(a,w,m);
                uint32_t b=__shfl_xor_sync(0xffffffffu,a,D);
                v[side][r]=upper?sub(b,a,m.p):add(a,b,m.p);
            }
        }
    }
}
template<unsigned Len,unsigned D=Len/2>
__device__ __forceinline__ void paired_forward(uint32_t (&v)[2][Len/32],
        const Twiddle* roots,const Mod& mod,unsigned lane) {
    paired_step<Len,D>(v,roots,mod,lane);
    if constexpr(D>1) paired_forward<Len,D/2>(v,roots,mod,lane);
}
}
