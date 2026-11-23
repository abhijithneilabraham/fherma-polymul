// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "warp-mod.hpp"
#include <stdexcept>

namespace rader7 {
using namespace bytecrt::warp_ntt;
using U = uint32_t;
struct Weights { Twiddle branch[2][4]; };

inline Weights prepare(U root, const Mod& m) {
    auto power = [&](U x, U n) {
        U y=1;
        while (n) {
            if (n&1) y=uint64_t(y)*x%m.p;
            x=uint64_t(x)*x%m.p;
            n>>=1;
        }
        return y;
    };
    if (root==1 || power(root,7)!=1) throw std::invalid_argument("order-seven root");
    const U exponents[6]={1,3,2,6,4,5};
    U h[6];
    for(unsigned i=0;i<6;++i) h[i]=power(root,exponents[i]);
    Weights out{};
    const U scale=power(6,m.p-2);
    for(unsigned sign=0;sign<2;++sign) {
        U c[3];
        for(unsigned j=0;j<3;++j) {
            unsigned e=4*j%6, o=(e+3)%6;
            c[j]=sign?sub(h[e],h[o],m.p):add(h[e],h[o],m.p);
        }
        U u=sub(c[0],c[2],m.p), v=sub(c[1],c[2],m.p);
        U values[]={add(c[0],add(c[1],c[2],m.p),m.p),u,v,add(u,v,m.p)};
        for(unsigned j=0;j<4;++j) {
            U value=uint64_t(values[j])*scale%m.p;
            out.branch[sign][j]={value,U((uint64_t(value)<<32)/m.p)};
        }
    }
    return out;
}

#ifdef __CUDACC__
#define RADER_INLINE __device__ __forceinline__
#else
#define RADER_INLINE inline
#endif
// The factors of 1/2 and 1/3 from the two CRT interpolations are in the weights.
RADER_INLINE void cyclic_three(const U x[3],const Twiddle w[4],const Mod& m,U out[3]) {
    U a=sub(x[0],x[2],m.p), b=sub(x[1],x[2],m.p);
    U sum=multiply_twiddle(add(x[0],add(x[1],x[2],m.p),m.p),w[0],m);
    U ac=multiply_twiddle(a,w[1],m), bd=multiply_twiddle(b,w[2],m);
    U both=multiply_twiddle(add(a,b,m.p),w[3],m);
    U constant=sub(ac,bd,m.p), linear=sub(sub(both,ac,m.p),add(bd,bd,m.p),m.p);
    out[2]=sub(sub(sum,constant,m.p),linear,m.p);
    out[0]=add(out[2],add(constant,add(constant,constant,m.p),m.p),m.p);
    out[1]=add(out[2],add(linear,add(linear,linear,m.p),m.p),m.p);
}

RADER_INLINE void transform(const U x[7],const Weights& w,const Mod& m,U y[7]) {
    const unsigned powers[]={1,3,2,6,4,5};
    const unsigned inverse[]={1,5,4,6,2,3};
    U plus[3],minus[3],p[3],n[3];
    U dc=x[0];
    for(unsigned j=0;j<3;++j) {
        unsigned e=4*j%6, o=(e+3)%6;
        plus[j]=add(x[inverse[e]],x[inverse[o]],m.p);
        minus[j]=sub(x[inverse[e]],x[inverse[o]],m.p);
        dc=add(dc,plus[j],m.p);
    }
    cyclic_three(plus,w.branch[0],m,p);
    cyclic_three(minus,w.branch[1],m,n);
    y[0]=dc;
    for(unsigned j=0;j<3;++j) {
        unsigned e=4*j%6, o=(e+3)%6;
        y[powers[e]]=add(x[0],add(p[j],n[j],m.p),m.p);
        y[powers[o]]=add(x[0],sub(p[j],n[j],m.p),m.p);
    }
}
#undef RADER_INLINE
}
