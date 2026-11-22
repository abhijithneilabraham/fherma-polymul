// CPU ownership/layout proof. No CUDA execution or performance inference.
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
#include "../fused-layout.hpp"

static void check(unsigned N, unsigned P, unsigned L) {
    const unsigned K = N / 256, groups = (P + 7) / 8;
    std::vector<uint8_t> loaded(size_t{N} * L, 0);
    std::vector<uint8_t> consumed(size_t{2} * P * 256, 0);
    for (unsigned id = 0; id < 256; ++id) {
        for (unsigned first = 0; first < K; first += 32) {
            for (unsigned row = first; row < first + 32; ++row) {
                const auto natural = bytecrt::fused_layout::coefficient(id, row);
                assert(natural < N && natural % 256 == id && natural / 256 == row);
                for (unsigned limb = 0; limb < L; ++limb) {
                    auto offset = bytecrt::fused_layout::input_word(id, row, limb, L);
                    assert(offset == size_t{natural} * L + limb);
                    assert(offset < loaded.size() && loaded[offset]++ == 0);
                }
            }
        }
        for (unsigned operand = 0; operand < 2; ++operand) {
            for (unsigned group = 0; group < groups; ++group) {
                const unsigned first_prime = group * 8;
                const unsigned count = P - first_prime < 8 ? P - first_prime : 8;
                std::vector<uint8_t> ready(8 * K, 0);
                for (unsigned first = 0; first < K; first += 32) {
                    for (unsigned warp = 0; warp < 4; ++warp) {
                        if (2 * warp >= count) continue;
                        for (unsigned lane = 0; lane < 32; ++lane) {
                            const unsigned local = 2 * warp + (lane & 3) / 2;
                            if ((lane & 1) || local >= count) continue;
                            for (unsigned quarter = 0; quarter < 4; ++quarter) {
                                const unsigned row = first + (lane >> 2) + 8 * quarter;
                                assert(ready[local * K + row]++ == 0);
                            }
                        }
                    }
                }
                for (unsigned local = 0; local < 8; ++local)
                    for (unsigned i = 0; i < K; ++i)
                        assert(ready[local * K + i] == (local < count ? 1 : 0));
                for (unsigned local = 0; local < count; ++local) {
                    const unsigned prime = first_prime + local;
                    const auto base = bytecrt::fused_layout::plane(operand, prime, P, N);
                    assert(base == (size_t{operand} * P + prime) * N);
                    const auto owner = base / K + id;
                    assert(owner < consumed.size() && consumed[owner]++ == 0);
                }
            }
        }
    }
    for (auto count : loaded) assert(count == 1);
    for (auto count : consumed) assert(count == 1);
    std::cout << "PASS N=" << N << ": " << loaded.size() << " input words; "
              << consumed.size() << " unique SDK stage consumers\n";
}

int main() { check(65536,114,55); check(32768,57,28); }
