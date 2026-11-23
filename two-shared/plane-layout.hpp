// SPDX-License-Identifier: Apache-2.0
#pragma once


namespace bytecrt::warp_ntt {
// Groups 0,1 are positive and negative evaluations of operand A.
// Groups 2,3 occupy the same two sections of operand B.
template<unsigned Planes>
#ifdef __CUDACC__
__host__ __device__
#endif
constexpr unsigned evaluation_plane(unsigned prime, unsigned group) {
    static_assert(Planes == 116 || Planes == 60, "two-digit evaluation layout");
    return prime + group*(Planes/2);
}
}
