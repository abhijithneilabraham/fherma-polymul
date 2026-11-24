#include "../warp-mod.hpp"
#include "../two-primes.hpp"
#include <array>
#include <cassert>
#include <iostream>

using namespace bytecrt::warp_ntt;

template<size_t Count>
void check_basis(const two_digit::PrimeRoot (&basis)[Count]) {
    for (const auto& prime : basis) {
        uint32_t p = prime.p;
        const uint64_t radix = uint64_t{1} << 32;
        Mod mod = make_mod(p,1);
        const std::array<uint32_t,8> values = {0,1,p/2,p-1,p,p+1,2*p-2,2*p-1};
        for (uint32_t root : {1u,2u,p/2,p-1}) {
            Twiddle w{root,uint32_t(radix*root/p)};
            uint32_t phase = (radix%p)*root%p;
            for (uint32_t a : values) for (uint32_t b : values) {
                uint32_t left = canonical(a,p), right = multiply_twiddle(b,w,mod);
                assert(left == a%p && right == uint64_t(b)*root%p);
                uint32_t sum = left+right, difference = left+p-right;
                assert(sum < 2*p && difference < 2*p);
                for (uint32_t value : {sum,difference})
                    assert(multiply(value,phase,mod) == uint64_t(value)*root%p);
            }
        }
    }
}

int main() {
    check_basis(two_digit::large);
    check_basis(two_digit::small);
    std::cout << "PASS inverse range and final phase for both prime bases\n";
}
