// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "warp-tables.hpp"

namespace bytecrt::warp_ntt {
inline std::vector<uint32_t> make_inverse_matrix(const RnsPlan& plan) {
    constexpr unsigned length=32, bytes=4;
    const unsigned count=plan.pcount/2;
    std::vector<uint32_t> packed(size_t(count)*length*length*bytes);
    for(unsigned prime=0;prime<count;++prime) {
        const uint32_t p=plan.primes[prime];
        const auto root=rns_detail::pow_mod(plan.psi_roots[prime],plan.n/length,p);
        const auto step=rns_detail::pow_mod(root,p-2,p);
        uint32_t powers[2*length]={1};
        for(unsigned i=1;i<2*length;++i)powers[i]=uint64_t(powers[i-1])*step%p;
        auto* weights=packed.data()+size_t(prime)*length*length*bytes;
        for(unsigned row=0;row<length;++row)for(unsigned freq=0;freq<length;++freq) {
            const unsigned exponent=(row*(2*reversed(freq,length)+1))%(2*length);
            uint32_t v=powers[exponent];
            for(unsigned k=0;k<bytes;++k) {
                for(unsigned out=0;out<bytes;++out)
                    weights[(4*row+out)*length+freq]|=((v>>(8*out))&255u)<<(8*k);
                v=uint64_t(v)*256%p;
            }
        }
    }
    return packed;
}
} // namespace bytecrt::warp_ntt
