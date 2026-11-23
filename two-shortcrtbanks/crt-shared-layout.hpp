// SPDX-License-Identifier: Apache-2.0
#pragma once
#ifdef __CUDACC__
#define CRT_LAYOUT_HD __host__ __device__
#else
#define CRT_LAYOUT_HD
#endif
namespace bytecrt::crt_layout {
CRT_LAYOUT_HD constexpr unsigned mask(unsigned row) { return row >> 3; }
CRT_LAYOUT_HD constexpr unsigned word(unsigned row,unsigned word_index) {
    return word_index ^ mask(row);
}
} // namespace bytecrt::crt_layout
#undef CRT_LAYOUT_HD
