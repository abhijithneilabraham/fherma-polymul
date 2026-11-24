// SPDX-License-Identifier: Apache-2.0
// Montgomery multiplication retains ByteCRT's exact p<2^31 word formula.
#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define WARP_HD __host__ __device__ __forceinline__
#else
#define WARP_HD inline
#endif
namespace bytecrt::warp_ntt {
struct Mod { uint32_t p, negative_inverse, r2, normalize; };
struct Twiddle { uint32_t low, high; };
inline Twiddle make_twiddle(uint32_t value,uint32_t p){
    const uint64_t scaled=uint64_t{value}<<32;
    const uint32_t high=scaled/p;
    const uint64_t remainder=scaled%p;
    const uint32_t low=uint32_t(((remainder<<32)+p-1)/p);
    return {uint32_t(0)-low,high+uint32_t(low!=0)};
}
WARP_HD uint32_t add(uint32_t a,uint32_t b,uint32_t p){uint32_t x=a+b;return x>=p?x-p:x;}
WARP_HD uint32_t sub(uint32_t a,uint32_t b,uint32_t p){return a>=b?a-b:a+p-b;}
WARP_HD uint32_t multiply(uint32_t a,uint32_t b,const Mod& m){
    uint64_t x=uint64_t{a}*b;
    uint32_t t=static_cast<uint32_t>(x)*m.negative_inverse;
    uint32_t u=static_cast<uint32_t>((x+uint64_t{t}*m.p)>>32);
    return u>=m.p?u-m.p:u;
}
WARP_HD uint32_t multiply_twiddle(uint32_t a,const Twiddle& w,const Mod& m){
#ifdef __CUDA_ARCH__
    const uint32_t carry=__umulhi(a,w.low);
    return __umulhi(a*w.high-carry,m.p);
#else
    const uint32_t carry=uint32_t((uint64_t{a}*w.low)>>32);
    return uint32_t((uint64_t{uint32_t(a*w.high-carry)}*m.p)>>32);
#endif
}
// The fixed multiplier accepts every word, including a+p-b below 2p.
WARP_HD uint32_t scaled_difference(uint32_t a, uint32_t b, const Twiddle& w, const Mod& m) {
    return multiply_twiddle(a + m.p - b, w, m);
}
inline Mod make_mod(uint32_t p,uint32_t inverse_n){
    uint32_t inverse=1;for(unsigned i=0;i<5;++i)inverse*=2-p*inverse;
    uint64_t r=(uint64_t{1}<<32)%p;
    return {p,uint32_t{0}-inverse,uint32_t(r*r%p),uint32_t(uint64_t{inverse_n}*r%p)};
}
} // namespace bytecrt::warp_ntt
#undef WARP_HD
