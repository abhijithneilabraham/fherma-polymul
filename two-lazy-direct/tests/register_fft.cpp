// Compare the production warp code with direct evaluations and cyclic products.
#include <array>
#include <barrier>
#include <cassert>
#include <iostream>
#include <random>
#include <thread>
#include "../two-plan.hpp"

using U = uint32_t;
thread_local unsigned host_lane;
static U exchange[32];
static std::barrier rendezvous{32};
static U __shfl_xor_sync(U mask, U value, unsigned bit) {
    assert(mask == UINT32_MAX && bit < 32 && (bit & (bit-1)) == 0);
    exchange[host_lane] = value;
    rendezvous.arrive_and_wait();
    U peer = exchange[host_lane ^ bit];
    rendezvous.arrive_and_wait();
    return peer;
}
#define __device__
#define __forceinline__ inline
#include "../register-fft.cuh"
using namespace bytecrt::warp_ntt;

void check(unsigned n, unsigned w, U gap) {
    std::vector<U> modulus((w+31)/32, UINT32_MAX);
    modulus.front() = 0u-gap;
    modulus.back() = (1u << (w%32))-1;
    auto plan = two_digit::make(n, w, modulus);
    auto tables = make_tables(plan.transform);
    std::mt19937 rng(25914);
    for (unsigned prime = 0; prime < plan.primes.size(); ++prime) {
        const Mod mod = tables.mods[prime];
        U p = mod.p;
        auto roots = tables.roots.data()+prime*(2*(n/256)+256)+2*(n/256);
        std::array<U,256> a, b, observed, spectrum;
        for (unsigned i=0; i<256; ++i) {
            a[i] = i%7 == 0 ? p-1 : rng()%p;
            b[i] = i%5 == 0 ? 0 : rng()%p;
        }
        std::vector<std::thread> lanes;
        for (unsigned l=0; l<32; ++l) lanes.emplace_back([&,l] {
            host_lane = l;
            U x[8], y[8];
            for (unsigned r=0; r<8; ++r) { x[r]=a[32*r+l]; y[r]=b[32*r+l]; }
            middle_fft<false>(x, roots, mod, l);
            middle_fft<false>(y, roots, mod, l);
            for (unsigned r=0; r<8; ++r) {
                unsigned original = 32*(l%8)+(l & ~7u)+r;
                spectrum[original] = x[r];
                x[r] = multiply(x[r], y[r], mod);
            }
            middle_fft<true>(x, roots, mod, l);
            for (unsigned r=0; r<8; ++r) observed[32*r+l]=x[r];
        });
        for (auto& t : lanes) t.join();
        U mont_inverse = bytecrt::rns_detail::pow_mod((uint64_t{1}<<32)%p,p-2,p);
        for (unsigned i=0; i<256; ++i) {
            unsigned frequency = reversed(i,256);
            U point=roots[frequency].value, evaluation=0, coefficient=0;
            for (unsigned j=0; j<256; ++j) {
                evaluation=(uint64_t(evaluation)*point+a[255-j])%p;
                coefficient=(uint64_t(a[j])*b[(i+256-j)%256]+coefficient)%p;
            }
            assert(spectrum[i]<2*p && evaluation==spectrum[i]%p);
            assert(uint64_t(256)*coefficient%p*mont_inverse%p == observed[i]%p && observed[i]<2*p);
        }
    }
    std::cout << "PASS " << plan.primes.size() << " complete 256-point spectra and dense cyclic products\n";
}
int main() {
    check(32768,868,43646975u);
    check(65536,1747,9699327u);
}
