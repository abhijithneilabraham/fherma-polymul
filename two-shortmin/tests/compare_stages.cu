// Compare the exact ranked kernel specializations using identical options.
#include "two-encode.cuh"
#include "warp-stages.cuh"

namespace two_digit {
template __global__ void tile_forward<65536,1747,55,RANKED_PRIMES,128,110>(
    const uint32_t*, const uint32_t*, uint32_t*, const uint8_t*,
    const uint32_t*, const uint32_t*, const uint32_t*, const Twiddle*, const Mod*);
}
namespace bytecrt::warp_ntt {
template __global__ void middle<65536,2*RANKED_PRIMES>(
    const uint32_t*, uint32_t*, const uint32_t*, const Twiddle*, const Mod*);
template __global__ void inverse<65536,2*RANKED_PRIMES,480>(
    const uint32_t*, uint32_t*, const Twiddle*, const Mod*);
}
