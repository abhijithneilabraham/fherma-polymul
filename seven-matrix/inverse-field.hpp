// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "warp-mod.hpp"

namespace seven_inverse {
struct alignas(128) Field {
    uint32_t tiles[4][64]{};
    bytecrt::warp_ntt::Mod mod;
};

// Pack a column for each of the four bytes of a recovered output digit.
// The missing eighth input and output digits retain zero weights.
inline void set_digit(Field& field,unsigned digit,uint32_t root_step,uint32_t recovery) {
    const uint32_t p=field.mod.p;
    uint32_t factor=recovery;
    for(unsigned input=0;input<7;++input) {
        uint32_t weight=factor;
        for(unsigned in_byte=0;in_byte<4;++in_byte) {
            for(unsigned out_byte=0;out_byte<4;++out_byte) {
                unsigned column=4*(digit%2)+out_byte;
                unsigned slot=32*(input/4)+4*column+input%4;
                field.tiles[digit/2][slot]|=((weight>>(8*out_byte))&255u)<<(8*in_byte);
            }
            weight=uint64_t(weight)*256%p;
        }
        factor=uint64_t(factor)*root_step%p;
    }
}
}
