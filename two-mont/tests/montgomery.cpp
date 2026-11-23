// SPDX-License-Identifier: Apache-2.0
#include "../two-primes.hpp"
#include "../warp-mod.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <random>

template<size_t Count>
void check(const two_digit::PrimeRoot (&primes)[Count]) {
    using namespace bytecrt::warp_ntt;
    unsigned tested = 0;
    std::mt19937 rng(75271);
    for (auto prime : primes) {
        const uint32_t p = prime.p;
        const uint64_t radix = (uint64_t{1} << 32) % p;
        const Mod mod = make_mod(p, 1);
        assert(mod.p * mod.word_inverse == 1u);
        auto verify = [&](uint32_t a, uint32_t b) {
            const uint32_t result = multiply(a, b, mod);
            assert(result < p && result * radix % p == uint64_t{a} * b % p);
            ++tested;
        };
        const std::array<uint32_t,11> edge = {0,1,2,65535,65536,65537,p/2,p/2+1,p-3,p-2,p-1};
        for (auto a : edge) for (auto b : edge) verify(a,b);
        for (unsigned i=0; i<16384; ++i) {
            uint32_t a=rng()%p, b=rng()%p;
            verify(a,b);
            verify(a,a);
        }
    }
    std::cout<<"PASS "<<tested<<" subtractive Montgomery products across "<<Count<<" primes\n";
}
int main() { check(two_digit::large); check(two_digit::small); }
