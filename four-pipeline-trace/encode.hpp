// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "radix4.hpp"

namespace four_encode {
using namespace radix4;
struct Field { Mod mod; uint32_t reciprocal48; Twiddle forward; };

template<unsigned W>
FOUR_INLINE U digit_word(const U* raw,unsigned digit,unsigned word) {
    unsigned lo=(digit*W+3)/4,hi=((digit+1)*W+3)/4;
    unsigned abs_word=lo/32+word;
    if(32*abs_word>=hi)return 0;
    U x=raw[abs_word];
    if(word==0)x&=UINT32_MAX<<(lo%32);
    if(hi-32*abs_word<32)x&=(U{1}<<(hi%32))-1;
    return x;
}

}
