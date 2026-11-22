// CPU test of the production output helper. MMA execution remains a GPU test.
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
#include "../core/rns-plan.hpp"

static std::array<std::array<int32_t, 2>, 32> registers;
static unsigned current_lane, exchange;
static int32_t __shfl_xor_sync(unsigned mask, int32_t value, unsigned delta) {
    assert(mask == 0xffffffffu && delta == 1 && exchange < 2);
    assert(value == registers[current_lane][exchange]);
    return registers[current_lane ^ 1][exchange++];
}
static uint64_t __umul64hi(uint64_t a, uint64_t b) {
    return static_cast<uint64_t>((static_cast<unsigned __int128>(a) * b) >> 64);
}
#define __device__
#define __forceinline__ inline
// Exact production Barrett reducer, extracted by test_warp_encode.py.
#include "encode-reducer.inc"
#include "../warp-output.cuh"

static uint32_t modulo(const std::vector<uint32_t>& words, uint32_t p) {
    uint64_t value = 0;
    for (auto it = words.rbegin(); it != words.rend(); ++it)
        value = ((value << 32) | *it) % p;
    return uint32_t(value);
}

template<bool Transposed>
static size_t fixture(unsigned N, unsigned L, unsigned P, unsigned block, unsigned pattern) {
    using namespace bytecrt;
    std::vector<uint32_t> primes;
    for (uint32_t p = (uint32_t{1} << 31) - 2 * N + 1; primes.size() < P; p -= 2 * N)
        if (rns_detail::prime32(p)) primes.push_back(p);
    std::vector<uint64_t> mu;
    for (auto p : primes) mu.push_back(UINT64_MAX / p);
    const unsigned id = Transposed ? block % 256 : 0;
    const unsigned first_row = (Transposed ? block / 256 : block) * 32;
    std::mt19937 random(991 + block * 11 + pattern);
    std::vector<std::vector<uint32_t>> input(32, std::vector<uint32_t>(L));
    for (unsigned row = 0; row < 32; ++row)
        for (unsigned limb = 0; limb < L; ++limb)
            input[row][limb] = pattern == 0 ? 0 : pattern == 1 ? UINT32_MAX :
                pattern == 2 ? (uint32_t{1} << ((row + limb) % 32)) : random();
    constexpr uint32_t untouched = UINT32_MAX;
    std::vector<uint32_t> output(size_t{2} * P * N + 2, untouched);
    size_t stores = 0;
    for (unsigned operand = 0; operand < 2; ++operand) {
        for (unsigned group = 0; group < (P + 13) / 14; ++group) {
            for (unsigned warp = 0; warp < 7; ++warp) {
                if (group * 14 + 2 * warp >= P) continue;
                for (unsigned quarter = 0; quarter < 4; ++quarter) {
                    // Independent byte-dot oracle, then the documented C fragment mapping.
                    for (unsigned lane = 0; lane < 32; ++lane) {
                        const unsigned row = (lane >> 2) + 8 * quarter;
                        for (unsigned reg = 0; reg < 2; ++reg) {
                            const unsigned column = group * 56 + warp * 8 + 2 * (lane & 3) + reg;
                            const unsigned prime = column / 4, digit = column % 4;
                            uint64_t dot = 0, power = 1;
                            if (prime < P) {
                                for (unsigned k = 0; k < 4 * L; ++k) {
                                    dot += ((input[row][k / 4] >> (8 * (k % 4))) & 255) *
                                           ((power >> (8 * digit)) & 255);
                                    power = power * 256 % primes[prime];
                                }
                            }
                            assert(dot <= INT32_MAX);
                            registers[lane][reg] = int32_t(dot);
                        }
                    }
                    for (unsigned lane = 0; lane < 32; ++lane) {
                        current_lane = lane; exchange = 0;
                        warp_encoding::store_pair<Transposed>(registers[lane][0], registers[lane][1],
                            lane, warp, group, first_row, id, quarter, N, P, operand,
                            primes.data(), mu.data(), output.data() + 1);
                        assert(exchange == 2); // Includes inactive writers and the odd tail prime.
                    }
                }
            }
        }
        // Independent inverse output mapping. Also detects duplicate/missing owners.
        for (unsigned prime = 0; prime < P; ++prime) {
            for (unsigned row = 0; row < 32; ++row) {
                const unsigned natural = Transposed ? id + (first_row + row) * 256 : first_row + row;
                if (natural >= N) continue;
                const unsigned destination = Transposed ? (natural % 256) * (N / 256) + natural / 256 : natural;
                auto& cell = output[1 + (size_t{operand} * P + prime) * N + destination];
                assert(cell == modulo(input[row], primes[prime]));
                cell = untouched;
                ++stores;
            }
        }
    }
    for (auto value : output) assert(value == untouched); // Includes both buffer guards.
    return stores;
}

int main() {
    size_t checked = 0;
    for (unsigned pattern = 0; pattern < 4; ++pattern) {
        for (unsigned block : {0u, 1071u, 2047u, 2048u})
            checked += fixture<true>(65536, 55, 114, block, pattern);
        for (unsigned block : {0u, 599u, 1023u, 1024u})
            checked += fixture<true>(32768, 28, 57, block, pattern);
        for (unsigned block : {0u, 2047u, 2048u})
            checked += fixture<false>(65536, 55, 114, block, pattern);
        for (unsigned block : {0u, 1023u, 1024u})
            checked += fixture<false>(32768, 28, 57, block, pattern);
    }
    // Shared-residue slices use N=K, local prime pointers and partial groups.
    for (unsigned pattern = 0; pattern < 4; ++pattern) {
        for (unsigned block : {0u, 7u, 8u}) {
            checked += fixture<false>(256, 55, 8, block, pattern);
            checked += fixture<false>(256, 55, 2, block, pattern);
        }
        for (unsigned block : {0u, 3u, 4u}) {
            checked += fixture<false>(128, 28, 8, block, pattern);
            checked += fixture<false>(128, 28, 1, block, pattern);
        }
    }
    std::cout << "PASS " << checked << " independent modulo/output checks; native GPU executions=0\n";
}
