// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "odd-fft.cuh"

namespace bytecrt::warp_ntt {
// Number of adjacent frequency rows reduced inside the middle kernel.
template<unsigned K>
constexpr unsigned first_group=K/32;

// Four neighboring coarse rows of each of two adjacent columns share a sector.
__device__ __forceinline__ unsigned partial_index(unsigned id,unsigned low,unsigned coarse) {
    return ((low*128+id/2)*8+coarse/4)*8+4*(id%2)+coarse%4;
}

template<unsigned K,unsigned Distance=1>
__device__ __forceinline__ void inverse_prefix(uint32_t (&v)[first_group<K>],unsigned first,
        const Twiddle* inverse_roots,const Mod& m) {
    #pragma unroll
    for(unsigned i=0;i<first_group<K>;++i) {
        if(i&Distance) continue;
        Twiddle root=inverse_roots[K/(2*Distance)+(first+i)/(2*Distance)];
        uint32_t a=v[i],b=v[i+Distance];
        v[i]=add(a,b,m.p);
        v[i+Distance]=multiply_twiddle(sub(a,b,m.p),root,m);
    }
    if constexpr(2*Distance<first_group<K>) inverse_prefix<K,Distance*2>(v,first,inverse_roots,m);
}

// Each half of the logical warp transforms one column. Physical bit two
// selects the column, so a four-row group occupies a contiguous sector pair.
template<unsigned Distance=1>
__device__ __forceinline__ void inverse_pair(uint32_t (&v)[2],unsigned row,
        const Twiddle* roots,const Mod& m) {
    constexpr unsigned groups=32/(2*Distance);
    if constexpr(Distance==16) {
        uint32_t a=v[0],b=v[1];
        v[0]=add(a,b,m.p);
        v[1]=multiply_twiddle(sub(a,b,m.p),roots[1],m);
    } else {
        #pragma unroll
        for(unsigned j=0;j<2;++j) {
            uint32_t a=v[j];
            uint32_t b=__shfl_xor_sync(0xffffffffu,a,Distance<4?Distance:2*Distance);
            Twiddle w=roots[groups+(row+16*j)/(2*Distance)];
            v[j]=(row&Distance)?multiply_twiddle(sub(b,a,m.p),w,m):add(a,b,m.p);
        }
        inverse_pair<2*Distance>(v,row,roots,m);
    }
}

}
