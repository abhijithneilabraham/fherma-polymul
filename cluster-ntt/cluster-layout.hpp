// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace bytecrt::warp_ntt {
constexpr unsigned cluster_blocks = 8, ids_per_block = 32, cluster_threads = 1024;
#ifdef __CUDACC__
__host__ __device__
#endif
constexpr unsigned cluster_slot(unsigned row, unsigned local_id) {
    // A full warp accesses all 32 banks along either the local transpose
    // or the remote, segment-aligned row. The same permutation reverses itself.
    return row*ids_per_block + (local_id ^ (row % ids_per_block));
}
}
