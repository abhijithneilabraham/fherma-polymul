// SPDX-License-Identifier: Apache-2.0
#include "../two-plan.hpp"
#include "../warp-tables.hpp"

#include <cassert>
#include <iostream>
#include <vector>

void check(unsigned n,unsigned w,uint32_t gap) {
    using namespace bytecrt::warp_ntt;
    std::vector<uint32_t> q((w+31)/32,UINT32_MAX);
    q.front()=uint32_t{0}-gap;
    q.back()=(1u<<(w%32))-1;
    const auto plan=two_digit::make(n,w,q);
    const auto tables=make_tables(plan.transform);
    const unsigned count=plan.primes.size(),words=forward_phase_words(n);
    assert(tables.forward_phase.size()==size_t(count)*words);
    for(unsigned prime=0;prime<count;++prime) {
        const uint32_t p=plan.primes[prime];
        const uint32_t radix=(uint64_t{1}<<32)%p;
        const auto* table=tables.forward_phase.data()+size_t(prime)*words;
        for(unsigned row=0;row<32;++row) {
            const unsigned odd=2*reversed(row,32)+1;
            assert(2*reverse_five(row)+1==odd);
            const uint32_t step=bytecrt::rns_detail::pow_mod(plan.transform.psi_roots[prime],odd,p);
            uint32_t wanted=radix;
            for(unsigned id=0;id<n/32;++id) {
                assert(forward_phase_at(table,id*odd,tables.mods[prime])==wanted);
                wanted=uint64_t(wanted)*step%p;
            }
        }
    }
    std::cout<<"PASS "<<size_t(n)*count<<" exact forward rotations at N="<<n
             <<"; "<<tables.forward_phase.size()*sizeof(uint32_t)<<" phase bytes\n";
}
int main() {
    check(32768,868,43646975u);
    check(65536,1747,9699327u);
}
