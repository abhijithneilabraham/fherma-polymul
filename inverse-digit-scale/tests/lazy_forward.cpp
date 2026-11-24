#include "../two-plan.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <random>

using namespace bytecrt::warp_ntt;
using bytecrt::rns_detail::pow_mod;
template<size_t N>
size_t check(const two_digit::PrimeRoot (&roots)[N]) {
    size_t checked=0;
    std::mt19937 rng(21983);
    for(const auto& root:roots) {
        const uint32_t p=root.p,radix=(uint64_t(1)<<32)%p;
        assert(p<(uint32_t(1)<<31));
        const Mod mod=make_mod(p,1);
        uint32_t rinv=pow_mod(radix,p-2,p);
        const std::array<uint32_t,7> edges={0,1,p-1,p,p+1,2*p-2,2*p-1};
        for(unsigned i=0;i<4096;++i) {
            uint32_t value=i<edges.size()?edges[i]%p:rng()%p;
            Twiddle w{value,uint32_t((uint64_t(value)<<32)/p)};
            for(auto x:edges) {
                uint32_t expected=uint64_t(x)*value%p;
                assert(multiply_twiddle(x,w,mod)==expected);
                assert(multiply(x,value,mod)==uint64_t(expected)*rinv%p);
                ++checked;
            }
        }
    }
    return checked;
}
int main() {
    std::cout<<"PASS "<<check(two_digit::large)+check(two_digit::small)
             <<" widened fixed and variable products\n";
}
