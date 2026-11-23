// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "core/rns-plan.hpp"

namespace bytecrt::warp_ntt {
// Pack each radix-16 transform weight into four byte columns. Input byte
// positions carry their own powers of 256 before the value is reduced mod p.
inline std::vector<uint32_t> tensor_weights(const RnsPlan& plan) {
    std::vector<uint32_t> packed(size_t(plan.pcount)*1024);
    auto* bytes = reinterpret_cast<uint8_t*>(packed.data());
    for (unsigned prime=0; prime<plan.pcount; ++prime) {
        uint32_t p=plan.primes[prime];
        uint32_t root=rns_detail::pow_mod(plan.omega_roots[prime],plan.n/16,p);
        for (unsigned k=0;k<16;++k) for (unsigned j=0;j<16;++j) {
            uint32_t value=rns_detail::pow_mod(root,j*k,p);
            for (unsigned digit=0;digit<4;++digit) {
                for (unsigned byte=0;byte<4;++byte)
                    bytes[prime*4096+(4*k+byte)*64+4*j+digit]=value>>(8*byte);
                value=uint64_t(value)*256%p;
            }
        }
    }
    return packed;
}
}
