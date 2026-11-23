// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define TWO_HD __host__ __device__
#else
#define TWO_HD
#endif
namespace two_digit {
template<unsigned W,unsigned L,unsigned Split>
TWO_HD inline uint32_t digit_word(const uint32_t* raw,unsigned side,unsigned index) {
    constexpr unsigned bytes=(W+7)/8;
    unsigned pos=4*index;
    unsigned length=side?bytes-Split:Split;
    if(pos>=length) return 0;
    unsigned start=pos+(side?Split:0),offset=(start%4)*8;
    uint32_t value=raw[start/4]>>offset;
    if(offset && start/4+1<L) value |= raw[start/4+1]<<(32-offset);
    unsigned remaining=length-pos;
    if(remaining<4) value &= (1u<<(8*remaining))-1;
    return value;
}
}
#undef TWO_HD
