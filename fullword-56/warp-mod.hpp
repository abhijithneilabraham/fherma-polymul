// SPDX-License-Identifier: Apache-2.0
// Subtractive Montgomery reduction supports every odd modulus below 2^32.
#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define WARP_HD __host__ __device__ __forceinline__
#else
#define WARP_HD inline
#endif
namespace bytecrt::warp_ntt {
struct Mod { uint32_t p, inverse, r2, normalize; };
struct Twiddle { uint32_t scaled, factor; };
WARP_HD uint32_t add(uint32_t a,uint32_t b,uint32_t p){
    uint32_t gap=p-b;
    return a>=gap?a-gap:a+b;
}
WARP_HD uint32_t sub(uint32_t a,uint32_t b,uint32_t p){return a>=b?a-b:a+p-b;}
WARP_HD uint32_t high_word(uint32_t a,uint32_t b){
#ifdef __CUDA_ARCH__
    return __umulhi(a,b);
#else
    return uint32_t((uint64_t{a}*b)>>32);
#endif
}
WARP_HD uint32_t multiply(uint32_t a,uint32_t b,const Mod& m){
    uint32_t t=a*b*m.inverse;
    return sub(high_word(a,b),high_word(t,m.p),m.p);
}
WARP_HD uint32_t multiply_twiddle(uint32_t a,const Twiddle& w,const Mod& m){
    return sub(high_word(a,w.scaled),high_word(a*w.factor,m.p),m.p);
}
inline Twiddle make_twiddle(uint32_t value,const Mod& m){
    uint32_t scaled=(uint64_t{value}<<32)%m.p;
    return {scaled,scaled*m.inverse};
}
inline Mod make_mod(uint32_t p,uint32_t inverse_n){
    uint32_t inverse=1;for(unsigned i=0;i<5;++i)inverse*=2-p*inverse;
    uint64_t r=(uint64_t{1}<<32)%p;
    return {p,inverse,uint32_t(r*r%p),uint32_t(uint64_t{inverse_n}*r%p)};
}
} // namespace bytecrt::warp_ntt
#undef WARP_HD
