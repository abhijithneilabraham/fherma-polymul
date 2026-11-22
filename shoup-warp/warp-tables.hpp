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
    if(!((N==65536 && P==114)||(N==32768 && P==57)))throw std::runtime_error("warp NTT point");
    Tables t;t.mods.resize(P);t.roots.resize(size_t(P)*(K+256));
    t.forward_phase.resize(size_t(P)*N);t.inverse_phase.resize(size_t(P)*N);
    std::vector<uint32_t> powers(N);
    for(unsigned prime=0;prime<P;++prime){
        uint32_t p=plan.primes[prime],omega=plan.omega_roots[prime];
        auto m=make_mod(p,plan.n_inverse[prime]);t.mods[prime]=m;
        uint32_t R=uint32_t((uint64_t{1}<<32)%p);powers[0]=R;
        for(unsigned i=1;i<N;++i)powers[i]=uint64_t(powers[i-1])*omega%p;
        for(unsigned id=0;id<256;++id)for(unsigned row=0;row<K;++row){
            unsigned exponent=id*reversed(row,K);
            t.forward_phase[size_t(prime)*N+id*K+row]=powers[exponent];
            t.inverse_phase[size_t(prime)*N+row*256+id]=powers[(N-exponent)&(N-1)];
        }
        const auto fixed = [&](uint32_t value){ return Twiddle{value,uint32_t((uint64_t{value}<<32)/p)}; };
        uint32_t local=1,step=rns_detail::pow_mod(omega,256,p);
        for(unsigned i=0;i<K;++i){t.roots[size_t(prime)*(K+256)+i]=fixed(local);local=uint64_t(local)*step%p;}
        local=1;step=rns_detail::pow_mod(omega,K,p);
        for(unsigned i=0;i<256;++i){t.roots[size_t(prime)*(K+256)+K+i]=fixed(local);local=uint64_t(local)*step%p;}
    }
    return t;
}
} // namespace bytecrt::warp_ntt
