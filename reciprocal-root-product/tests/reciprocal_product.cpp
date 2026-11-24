#include "../two-primes.hpp"
#include "../warp-mod.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <random>

using namespace bytecrt::warp_ntt;
template <size_t N>
size_t check(const two_digit::PrimeRoot (&primes)[N]) {
    std::mt19937 rng(7319);
    size_t cases=0;
    for(const auto& root:primes) {
        const uint32_t p=root.p;
        const Mod mod=make_mod(p,1);
        const std::array<uint32_t,8> edges={0,1,p/2,p-1,p,p+1,2*p-1,UINT32_MAX};
        for(unsigned i=0;i<8192;++i) {
            const uint32_t y=i<edges.size()?edges[i]%p:rng()%p;
            const Twiddle w=make_twiddle(y,p);
            auto verify=[&](uint32_t a) {
                assert(multiply_twiddle(a,w,mod)==uint64_t(a)*y%p);
                ++cases;
            };
            for(auto a:edges) verify(a);
            verify(rng());
        }
    }
    return cases;
}

int main() {
    std::cout<<"PASS "<<check(two_digit::large)+check(two_digit::small)
             <<" fixed products across both prime sets\n";
}
