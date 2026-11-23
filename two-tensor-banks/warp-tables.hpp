// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "warp-mod.hpp"
#include "core/rns-plan.hpp"
#include <vector>
namespace bytecrt::warp_ntt {
struct Tables {
    std::vector<Mod> mods;
    std::vector<Twiddle> roots;
    std::vector<uint32_t> forward_phase,inverse_phase;
};
inline unsigned reversed(unsigned value,unsigned length){
    unsigned result=0;for(length>>=1;length;length>>=1){result=(result<<1)|(value&1);value>>=1;}return result;
}
inline Tables make_tables(const RnsPlan& plan){
    unsigned N=plan.n,K=N/256,P=plan.pcount;
    if(!((N==65536 && (P==114 || P==116))||(N==32768 && (P==57 || P==60))))throw std::runtime_error("warp NTT point");
    Tables t;t.mods.resize(P);t.roots.resize(size_t(P)*(2*K+256));
    t.forward_phase.resize(size_t(P)*N);t.inverse_phase.resize(size_t(P)*N);
    std::vector<uint32_t> powers(2*N);
    for(unsigned prime=0;prime<P;++prime){
        uint32_t p=plan.primes[prime],omega=plan.omega_roots[prime];
        auto m=make_mod(p,plan.n_inverse[prime]);t.mods[prime]=m;
        uint32_t R = (uint64_t{1} << 32) % p;
        powers[0] = 1;
        for (unsigned i = 1; i < 2*N; ++i)
            powers[i] = uint64_t(powers[i-1]) * plan.psi_roots[prime] % p;

        // Forward phase operates on ordinary residues. The pointwise Montgomery
        // product costs one factor of 1/R. Inverse phase restores it and
        // applies normalization and the prime's ordinary CRT weight.
        uint32_t scale = uint64_t(m.r2) * plan.n_inverse[prime] % p;
        scale = uint64_t(scale) * plan.crt_inverse[prime] % p;
        for (unsigned id = 0; id < 256; ++id) for (unsigned row = 0; row < K; ++row) {
            unsigned exponent = id * (2*reversed(row,K) + 1);
            t.forward_phase[size_t(prime)*N + id*K + row] = uint64_t(R) * powers[exponent] % p;
            // The tensor inverse reads and scales the forward-ordered values
            // before reversing the output coefficients in shared memory.
            t.inverse_phase[size_t(prime)*N + row*256 + (256-id)%256] =
                uint64_t(scale) * powers[(2*N-exponent) % (2*N)] % p;
        }
        const auto fixed = [&](uint32_t v) { return Twiddle{v,uint32_t((uint64_t{v}<<32)/p)}; };
        Twiddle* roots = t.roots.data() + size_t(prime)*(2*K+256);
        for (unsigned groups = 1; groups < K; groups *= 2) {
            unsigned distance = K / (2*groups);
            for (unsigned group = 0; group < groups; ++group) {
                unsigned exponent = 256 * distance * (2*reversed(group, groups)+1);
                roots[groups+group] = fixed(powers[exponent]);
                roots[K+groups+group] = fixed(powers[2*N-exponent]);
            }
        }
        uint32_t local = 1, step = rns_detail::pow_mod(omega,K,p);
        for (unsigned i = 0; i < 256; ++i) {
            roots[2*K+i] = fixed(local);
            local = uint64_t(local)*step % p;
        }
    }
    return t;
}
} // namespace bytecrt::warp_ntt
