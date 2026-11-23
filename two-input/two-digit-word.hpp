// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define TWO_HD __host__ __device__
#else
#define TWO_HD
#endif
namespace two_digit {
struct WordPair { uint32_t first,second; };
template<unsigned W,unsigned L,unsigned Width,class Load>
TWO_HD inline WordPair load_words(unsigned lane,Load load) {
    static_assert(Width<L && L<=2*Width);
    uint32_t front=load(lane),back=0;
    if(lane+Width<L){
        back=load(lane+Width);
        constexpr unsigned bytes=(W+7)/8;
        if constexpr(bytes%4)
            if(lane+Width==L-1)back&=(1u<<(8*(bytes%4)))-1;
    }
    return {front,back};
}
template<unsigned Split,unsigned Width,class Exchange>
TWO_HD inline WordPair split_words(WordPair raw,unsigned lane,Exchange exchange) {
    constexpr unsigned first=Split/4,offset=8*(Split%4);
    static_assert(first<Width && offset!=0);
    constexpr uint32_t mask=(1u<<offset)-1;
    uint32_t lo=lane<first?raw.first:(lane==first?raw.first&mask:0);
    uint32_t hi=exchange(lane<first?raw.second:raw.first,(lane+first)%Width);
    return {lo,lane==0?hi&~mask:hi};
}
}
#undef TWO_HD
