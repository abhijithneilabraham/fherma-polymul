// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>

namespace two_window {
constexpr unsigned capacity=12;
template<unsigned Count>
constexpr unsigned group_count=(Count+capacity-1)/capacity;

#if defined(__CUDACC__)
#define WINDOW_HD __host__ __device__
#else
#define WINDOW_HD
#endif

WINDOW_HD constexpr unsigned live_primes(unsigned count,unsigned first) {
    return count-first<capacity?count-first:capacity;
}

WINDOW_HD constexpr unsigned local_plane(unsigned plane,unsigned count) {
    return plane%count+capacity*(plane/count);
}

WINDOW_HD constexpr unsigned global_plane(unsigned plane,unsigned count,unsigned first,unsigned total) {
    return first+plane%count+total*(plane/count);
}

template<unsigned N>
constexpr std::size_t spectrum_words=4*std::size_t(N)*capacity;

#undef WINDOW_HD
}
