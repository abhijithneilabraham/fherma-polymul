// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace four_decode {
// Each number names an eight-column CRT output tile. Five bits encode its
// one-based index. Zero terminates a warp's list.
constexpr uint32_t slots(unsigned a, unsigned b=0, unsigned c=0, unsigned d=0) {
    return a | (b<<5) | (c<<10) | (d<<15);
}
#ifdef __CUDACC__
#define SCHEDULE_DATA static __device__ __constant__ const
#else
#define SCHEDULE_DATA static constexpr
#endif
SCHEDULE_DATA uint32_t ranked_slots[8] = {
    slots(1,2,3), slots(7,8,4), slots(14,15,5), slots(21,22,6),
    slots(9,10,11,12), slots(13,16,17,18), slots(19,20,23,24), slots(25,26,27,28)
};
SCHEDULE_DATA uint32_t auxiliary_slots[8] = {
    slots(1,3), slots(2,6), slots(4,9), slots(5,10),
    slots(7,13), slots(8,14), slots(11), slots(12)
};
#undef SCHEDULE_DATA
}
