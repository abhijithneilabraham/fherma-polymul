// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace bytecrt {
// Store one 8-column, 32-byte K tile in the register order of an MMA warp.
// The last seven-warp block can contain inactive columns.
inline std::vector<uint8_t> pack_mma_matrix(const std::vector<uint8_t>& source,
                                            unsigned columns, unsigned k_bytes) {
    if (k_bytes % 32 || source.size() != size_t(columns) * k_bytes)
        throw std::invalid_argument("unexpected byte matrix dimensions");
    const unsigned padded_columns = (columns + 55) / 56 * 56;
    std::vector<uint8_t> packed(size_t(padded_columns) * k_bytes, 0);
    for (unsigned column = 0; column < columns; ++column) {
        for (unsigned k = 0; k < k_bytes; k += 4) {
            const unsigned lane = 4 * (column % 8) + (k % 16) / 4;
            const size_t tile = size_t(column / 8) * (k_bytes / 32) + k / 32;
            const size_t offset = tile * 256 + lane * 8 + (k % 32) / 16 * 4;
            std::memcpy(&packed[offset], &source[size_t(column) * k_bytes + k], 4);
        }
    }
    return packed;
}
} // namespace bytecrt
