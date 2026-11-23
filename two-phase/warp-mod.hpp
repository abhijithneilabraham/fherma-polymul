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
WARP_HD uint32_t multiply(uint32_t a,uint32_t b,const Mod& m){
    uint64_t x=uint64_t{a}*b;
    uint32_t t=static_cast<uint32_t>(x)*m.negative_inverse;
    uint32_t u=static_cast<uint32_t>((x+uint64_t{t}*m.p)>>32);
    return u>=m.p?u-m.p:u;
}
// For R=2^32 and 0<=value<p, this ceiling fits in one word.
inline uint32_t phase_code(uint32_t value,uint32_t p) {
    return uint32_t(((uint64_t(value)<<32)+p-1)/p);
}
WARP_HD uint32_t high_word(uint32_t a,uint32_t b) {
#ifdef __CUDA_ARCH__
    return __umulhi(a,b);
#else
    return uint32_t((uint64_t(a)*b)>>32);
#endif
}
WARP_HD uint32_t multiply_phase(uint32_t a,uint32_t code,uint32_t p) {
    // Both high products are independent. The quotient can overshoot by one.
    const uint32_t value=high_word(code,p);
    const uint32_t quotient=high_word(a,code);
    const uint32_t residue=a*value-quotient*p;
    return (residue>>31)?residue+p:residue;
}
WARP_HD uint32_t multiply_twiddle(uint32_t a,const Twiddle& w,const Mod& m){
    #ifdef __CUDA_ARCH__
    uint32_t quotient=__umulhi(a,w.reciprocal);
#else
    uint32_t quotient=uint32_t((uint64_t{a}*w.reciprocal)>>32);
#endif
    uint32_t remainder=a*w.value-quotient*m.p;
    return remainder>=m.p?remainder-m.p:remainder;
}
inline Mod make_mod(uint32_t p,uint32_t inverse_n){
    uint32_t inverse=1;for(unsigned i=0;i<5;++i)inverse*=2-p*inverse;
    uint64_t r=(uint64_t{1}<<32)%p;
    return {p,uint32_t{0}-inverse,uint32_t(r*r%p),uint32_t(uint64_t{inverse_n}*r%p)};
}
} // namespace bytecrt::warp_ntt
#undef WARP_HD
