// SPDX-License-Identifier: Apache-2.0
#include "../short-layout.hpp"

#include <cassert>
#include <iostream>
#include <set>
#include <vector>

using bytecrt::warp_ntt::short_partial_index;

namespace {
constexpr unsigned rows = 32;
constexpr unsigned words_per_sector = 8;

unsigned previous_index(unsigned column, unsigned row) {
    return (column / 4) * rows * 4 + row * 4 + column % 4;
}

template<unsigned N>
void check() {
    constexpr unsigned columns = N / rows;
    std::vector<bool> visited(N);
    for (unsigned column = 0; column < columns; ++column) {
        std::set<unsigned> before, after;
        for (unsigned row = 0; row < rows; ++row) {
            const unsigned index = short_partial_index(column, row);
            assert(index < N && !visited[index]);
            visited[index] = true;
            before.insert(previous_index(column, row) / words_per_sector);
            after.insert(index / words_per_sector);
        }
        assert(before.size() == 16 && after.size() == 8);
    }
    for (bool written : visited) assert(written);

    constexpr unsigned warp_size = 32;
    for (unsigned first = 0; first < columns; first += 2) {
        for (unsigned half = 0; half < 2; ++half) {
            std::set<unsigned> sectors;
            for (unsigned lane = 0; lane < warp_size; ++lane) {
                const unsigned address = short_partial_index(first + lane % 2, lane / 2 + 16 * half);
                assert(address == first * rows + half * warp_size + lane);
                sectors.insert(address / words_per_sector);
            }
            assert(sectors.size() == 4);
        }
    }
    constexpr unsigned middle_threads = 512;
    constexpr unsigned column_step = 256;
    for (unsigned first = 0; first < rows; first += 2)
        for (unsigned base = 0; base < columns; base += column_step)
            for (unsigned warp = 0; warp < middle_threads / warp_size; ++warp) {
                std::set<unsigned> before, after;
                for (unsigned lane = 0; lane < warp_size; ++lane) {
                    const unsigned thread = warp * warp_size + lane;
                    const unsigned column = base + 4 * (thread / 8) + thread % 4;
                    const unsigned row = first + (thread / 4) % 2;
                    before.insert(previous_index(column, row) / words_per_sector);
                    after.insert(short_partial_index(column, row) / words_per_sector);
                }
                assert(before.size() == 4 && after.size() == 8);
            }
    std::cout << "PASS all " << N << " partial addresses; both columns use eight full load sectors\n";
}
} // namespace

int main() {
    check<32768>();
    check<65536>();
}
