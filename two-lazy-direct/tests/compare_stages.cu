// Instantiate identical ranked stage shapes for a standalone compiler comparison.
#include "warp-stages.cuh"
namespace bytecrt::warp_ntt {
template __global__ void forward<65536,120>(const uint32_t*,uint32_t*,const uint32_t*,const Twiddle*,const Mod*);
template __global__ void middle<65536,120>(const uint32_t*,uint32_t*,const uint32_t*,const Twiddle*,const Mod*);
template __global__ void inverse<65536,120,480>(const uint32_t*,uint32_t*,const Twiddle*,const Mod*);
}
