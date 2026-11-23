// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace bytecrt::warp_ntt {
// One contiguous eight-word strip per column. The destination row is
// then contiguous in the dimension consumed by the middle transform.
template<unsigned K>
__device__ __forceinline__ void gather_middle_word(const uint32_t* spectrum,
        uint32_t (&tile)[8][260], unsigned first, unsigned word) {
    const unsigned row = word % 8;
    const unsigned column = word / 8;
    tile[row][column] = spectrum[column*K + first + row];
}
}
