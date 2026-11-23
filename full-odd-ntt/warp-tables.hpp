// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "warp-mod.hpp"
#include "core/rns-plan.hpp"
#include <vector>
namespace bytecrt::warp_ntt {
struct Tables {
    std::vector<Mod> mods;
    std::vector<Twiddle> roots;
};
inline unsigned reversed(unsigned value,unsigned length){
    unsigned result=0;for(length>>=1;length;length>>=1){result=(result<<1)|(value&1);value>>=1;}return result;
}
inline Tables make_tables(const RnsPlan& plan){
    unsigned N=plan.n,K=N/256,P=plan.pcount;
    if(!((N==65536 && P==114)||(N==32768 && P==57)))throw std::runtime_error("warp NTT point");
    Tables t;t.mods.resize(P);t.roots.resize(size_t(P)*(2*K+N));
    std::vector<uint32_t> powers(2*N);
    for(unsigned prime=0;prime<P;++prime){
        uint32_t p=plan.primes[prime];
        auto m=make_mod(p,plan.n_inverse[prime]);t.mods[prime]=m;
        powers[0] = 1;
        for (unsigned i=1; i<2*N; ++i)
            powers[i] = uint64_t(powers[i-1]) * plan.psi_roots[prime] % p;
        // Variable Montgomery multiplication introduces 1/R. One final
        // Montgomery product removes that factor and applies N^-1 and CRT.
        uint32_t scale = uint64_t(m.r2) * plan.n_inverse[prime] % p;
        t.mods[prime].normalize = uint64_t(scale) * plan.crt_inverse[prime] % p;
        const auto fixed = [&](uint32_t v) { return Twiddle{v,uint32_t((uint64_t{v}<<32)/p)}; };
        Twiddle* roots = t.roots.data() + size_t(prime)*(2*K+N);
        for (unsigned groups = 1; groups < K; groups *= 2) {
            unsigned distance = K / (2*groups);
            for (unsigned group = 0; group < groups; ++group) {
                unsigned exponent = 256 * distance * (2*reversed(group, groups)+1);
                roots[groups+group] = fixed(powers[exponent]);
                roots[K+groups+group] = fixed(powers[2*N-exponent]);
            }
        }
        // The second dimension evaluates its input at the roots for this
        // outer frequency. The reciprocal is the negative of the mirrored
        // forward root because psi^N = -1.
        for (unsigned row=0; row<K; ++row) {
            unsigned odd = 2*reversed(row,K)+1;
            Twiddle* inner = roots+2*K+256*row;
            for (unsigned groups=1; groups<256; groups*=2) {
                unsigned distance=256/(2*groups);
                for (unsigned g=0; g<groups; ++g) {
                    unsigned exponent=distance*(odd + 2*K*reversed(g,groups));
                    inner[groups+g]=fixed(powers[exponent]);
                }
            }
        }
    }
    return t;
}
} // namespace bytecrt::warp_ntt
