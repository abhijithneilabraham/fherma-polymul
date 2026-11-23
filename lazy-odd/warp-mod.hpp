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
struct Twiddle { uint32_t value, reciprocal; };
WARP_HD uint32_t add(uint32_t a,uint32_t b,uint32_t p){uint32_t x=a+b;return x>=p?x-p:x;}
WARP_HD uint32_t sub(uint32_t a,uint32_t b,uint32_t p){return a>=b?a-b:a+p-b;}
// All transform registers hold representatives in [0,2p). Comparing with
// 2p-b handles a 32-bit wrap in the addition without a 64-bit sum.
WARP_HD uint32_t add_twice(uint32_t a,uint32_t b,uint32_t p) {
    uint32_t modulus=2*p, sum=a+b;
    return a>=modulus-b ? sum-modulus : sum;
}
WARP_HD uint32_t sub_twice(uint32_t a,uint32_t b,uint32_t p) {
    uint32_t difference=a-b;
    return a<b ? difference+2*p : difference;
}
WARP_HD uint32_t canonical(uint32_t a,uint32_t p) { return a>=p ? a-p : a; }
WARP_HD uint32_t multiply(uint32_t a,uint32_t b,const Mod& m){
    uint64_t x=uint64_t{a}*b;
    uint32_t t=static_cast<uint32_t>(x)*m.negative_inverse;
    uint32_t u=static_cast<uint32_t>((x+uint64_t{t}*m.p)>>32);
    return u>=m.p?u-m.p:u;
}
WARP_HD uint32_t lazy_twiddle(uint32_t a,const Twiddle& w,const Mod& m){
    #ifdef __CUDA_ARCH__
    uint32_t quotient=__umulhi(a,w.reciprocal);
#else
    uint32_t quotient=uint32_t((uint64_t{a}*w.reciprocal)>>32);
#endif
    // a<2p<2^32, so the reciprocal approximation misses by at most one.
    // The true nonnegative remainder fits in one word even if products wrap.
    return a*w.value-quotient*m.p;
}
inline Mod make_mod(uint32_t p,uint32_t inverse_n){
    uint32_t inverse=1;for(unsigned i=0;i<5;++i)inverse*=2-p*inverse;
    uint64_t r=(uint64_t{1}<<32)%p;
    return {p,uint32_t{0}-inverse,uint32_t(r*r%p),uint32_t(uint64_t{inverse_n}*r%p)};
}
} // namespace bytecrt::warp_ntt
#undef WARP_HD
