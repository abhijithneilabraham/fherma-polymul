#include <array>
#include <cassert>
#include <cfenv>
#include <cstdint>
#include <iostream>
#include <random>
#include "float_intrinsics.hpp"
#include "../fp32-reduce.cuh"
#include "../core/rns-plan.hpp"

static uint32_t estimate_reduce(uint64_t x, uint32_t p, float inverse) {
    // Four byte-dot sums reconstruct x. Use an exact base-256 decomposition
    // with the upper sum containing the remaining bits, within s32 bounds.
    const int32_t sums[4] = {int32_t(x & 255), int32_t((x >> 8) & 255),
                            int32_t((x >> 16) & 255), int32_t(x >> 24)};
    return bytecrt::fast_encoding::reduce(sums, p, inverse);
}

int main() {
    assert(std::fegetround() == FE_TONEAREST);
    uint64_t checked = 0;
    std::mt19937_64 random(476901);
    for (auto shape : {std::array<unsigned,3>{65536,55,114}, {32768,28,57}}) {
        const auto N=shape[0], L=shape[1], P=shape[2];
        unsigned primes = 0;
        for (uint32_t p=(uint32_t{1}<<31)-2*N+1; primes<P; p-=2*N) {
            if (!bytecrt::rns_detail::prime32(p)) continue;
            ++primes;
            const float inverse = static_cast<float>(1.0 / static_cast<double>(p));
            const uint64_t limit = uint64_t{4} * L * 255 * p;
            for (unsigned q=0; q<4*L*255; ++q) {
                for (int64_t delta : {-int64_t{1}, int64_t{0}, int64_t{1}, int64_t{p}-1}) {
                    const int64_t x = static_cast<int64_t>(uint64_t{q}*p) + delta;
                    if (x < 0 || uint64_t(x) >= limit) continue;
                    assert(estimate_reduce(x,p,inverse) == uint64_t(x)%p);
                    ++checked;
                }
            }
            for (unsigned i=0; i<1000; ++i) {
                uint64_t x=random()%limit;
                assert(estimate_reduce(x,p,inverse) == x%p);
                ++checked;
            }
            assert(estimate_reduce(limit-1,p,inverse) == (limit-1)%p);
            ++checked;
        }
    }
    std::cout << "PASS " << checked << " quotient boundary/random checks; native GPU executions=0\n";
}
