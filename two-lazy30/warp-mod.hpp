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
// Canonical residues and a modulus below 2^31 keep a sum below 2^32.
// A subtraction that wraps has its high bit set until corrected.
WARP_HD uint32_t reduce_once(uint32_t x,uint32_t p) {
    uint32_t corrected=x-p;
    return x<corrected?x:corrected;
}
WARP_HD uint32_t add(uint32_t a,uint32_t b,uint32_t p) {
    return reduce_once(a+b,p);
}
WARP_HD uint32_t sub(uint32_t a,uint32_t b,uint32_t p) {
    uint32_t x=a-b, corrected=x+p;
    return x<corrected?x:corrected;
}
WARP_HD uint32_t multiply(uint32_t a,uint32_t b,const Mod& m){
    uint64_t x=uint64_t{a}*b;
    uint32_t t=static_cast<uint32_t>(x)*m.negative_inverse;
    uint32_t u=static_cast<uint32_t>((x+uint64_t{t}*m.p)>>32);
    return reduce_once(u,m.p);
}
// Shoup's quotient estimate is exact or one below for any word a.
// The raw remainder is below 2p for canonical root w.
WARP_HD uint32_t lazy_twiddle(uint32_t a,const Twiddle& w,const Mod& m){
#ifdef __CUDA_ARCH__
    uint32_t quotient=__umulhi(a,w.reciprocal);
#else
    uint32_t quotient=uint32_t((uint64_t{a}*w.reciprocal)>>32);
#endif
    return a*w.value-quotient*m.p;
}
WARP_HD uint32_t multiply_twiddle(uint32_t a,const Twiddle& w,const Mod& m) {
    return reduce_once(lazy_twiddle(a,w,m),m.p);
}
WARP_HD uint32_t sum_twice(uint32_t a,uint32_t b,uint32_t p) {
    return reduce_once(a+b,2*p);
}
WARP_HD uint32_t diff_twice(uint32_t a,uint32_t b,uint32_t p) {
    return reduce_once(a+2*p-b,2*p);
}
WARP_HD uint32_t canonical_twice(uint32_t a,uint32_t p) {
    return reduce_once(a,p);
}
inline Mod make_mod(uint32_t p,uint32_t inverse_n){
    uint32_t inverse=1;for(unsigned i=0;i<5;++i)inverse*=2-p*inverse;
    uint64_t r=(uint64_t{1}<<32)%p;
    return {p,uint32_t{0}-inverse,uint32_t(r*r%p),uint32_t(uint64_t{inverse_n}*r%p)};
}
} // namespace bytecrt::warp_ntt
#undef WARP_HD
