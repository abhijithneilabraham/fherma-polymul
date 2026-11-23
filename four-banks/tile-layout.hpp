// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "radix4.hpp"
namespace four_encode {
template<unsigned W>
constexpr unsigned raw_stride = W==1747 ? 56 : 28;

template<unsigned W>
FOUR_INLINE unsigned raw_limb(unsigned row, unsigned limb) {
    // Ranked rows four apart start in the same bank. Exchange their
    // four-word groups so both halves of a matrix fragment can load together.
    return W==1747 ? limb ^ (row & 4u) : limb;
}

FOUR_INLINE unsigned tile_row(unsigned plane, unsigned column, unsigned row) {
    // Four consecutive columns use disjoint eight-bank groups. Within a
    // column the two prime lanes write to disjoint groups as well.
    return row ^ (column*8) ^ ((plane & 1u)*8);
}
}
