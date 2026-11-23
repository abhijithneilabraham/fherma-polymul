// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>

namespace short_window {
// Each input warp processes an even number of primes in pairs.
constexpr unsigned capacity(unsigned primes) {
    return 2*((primes+3)/4);
}
constexpr unsigned first(unsigned primes,unsigned group) {
    return group*capacity(primes);
}
constexpr unsigned count(unsigned primes,unsigned group) {
    return group==0?capacity(primes):primes-capacity(primes);
}
constexpr std::size_t words(unsigned n,unsigned primes) {
    return std::size_t(n)*4*capacity(primes);
}
} // namespace short_window
