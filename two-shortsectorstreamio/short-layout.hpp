// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace bytecrt::warp_ntt {
// A 32-byte sector contains four rows from each of two adjacent columns.
#ifdef __CUDACC__
__host__ __device__ __forceinline__
#else
inline
#endif
unsigned short_partial_index(unsigned column, unsigned row) {
    return (column / 2) * 64 + row * 2 + column % 2;
}
} // namespace bytecrt::warp_ntt
