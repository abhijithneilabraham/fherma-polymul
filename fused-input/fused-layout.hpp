// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define FUSED_HD __host__ __device__
#else
#define FUSED_HD
#endif

namespace bytecrt::fused_layout {
FUSED_HD constexpr unsigned coefficient(unsigned id, unsigned row) {
    return id + row * 256;
}
FUSED_HD constexpr std::size_t input_word(unsigned id, unsigned row, unsigned limb, unsigned L) {
    return std::size_t{coefficient(id, row)} * L + limb;
}
FUSED_HD constexpr std::size_t plane(unsigned operand, unsigned prime, unsigned P, unsigned N) {
    return (std::size_t{operand} * P + prime) * N;
}
} // namespace bytecrt::fused_layout
#undef FUSED_HD
