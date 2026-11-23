// SPDX-License-Identifier: Apache-2.0
#include "../two-primes.hpp"
#include "../warp-mod.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <random>

template <size_t Count>
unsigned check(const two_digit::PrimeRoot (&primes)[Count]) {
    using namespace bytecrt::warp_ntt;
    unsigned tested=0;
    std::mt19937 rng(356643);
    for(const auto prime:primes) {
        uint32_t p=prime.p;
        const auto mod=make_mod(p,1);
        const uint64_t radix=(uint64_t{1}<<32)%p;
        auto verify=[&](uint32_t a,uint32_t b) {
            assert(add(a,b,p)==(uint64_t(a)+b)%p);
            assert(sub(a,b,p)==(uint64_t(a)+p-b)%p);
            uint64_t product=uint64_t(a)*b%p;
            assert(uint64_t(multiply(a,b,mod))*radix%p==product);
            Twiddle fixed={b,uint32_t((uint64_t{b}<<32)/p)};
            assert(multiply_twiddle(a,fixed,mod)==product);
            ++tested;
        };
        std::array<uint32_t,9> edge={0,1,2,65535,65536,p/2,p-3,p-2,p-1};
        for(auto a:edge)for(auto b:edge)verify(a,b);
        const auto verify_product=[&](uint32_t a,uint32_t b) {
            assert(uint64_t(multiply(a,b,mod))*radix%p==uint64_t(a)*b%p);
            ++tested;
        };
        for(auto a:edge)for(auto b:edge) {
            verify_product(3*p+a,b);
            verify_product(p+a,p+b);
        }
        for(unsigned i=0;i<16384;++i) {
            uint32_t a=rng()%p, b=rng()%p;
            verify(a,b);
            verify(a,a);
            verify_product(rng()%(4*p),b);
            verify_product(rng()%(2*p),rng()%(2*p));
        }
    }
    return tested;
}
int main() {
    auto count=check(two_digit::large)+check(two_digit::small);
    std::cout<<"PASS "<<count<<" input pairs, including wide Montgomery products\n";
}
