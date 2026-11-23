// Instantiate the ranked input kernel with identical flags for both layouts.
#include "two-encode.cuh"

namespace two_digit {
template __global__ void tile_forward<65536,1747,55,58,128,110>(
    const uint32_t*, const uint32_t*, uint32_t*, const uint8_t*,
    const uint32_t*, const uint32_t*, const uint32_t*, const Twiddle*, const Mod*);
}
