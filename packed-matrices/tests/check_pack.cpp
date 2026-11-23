#include "../matrix-pack.hpp"
#include <cassert>
#include <random>
int main() {
    std::mt19937 rng(128);
    for (auto shape : {std::pair{456u, 224u}, {220u, 480u}, {228u, 128u}, {112u, 256u}}) {
        auto [columns, depth] = shape;
        std::vector<uint8_t> source(size_t(columns) * depth);
        for (auto& byte : source) byte = static_cast<uint8_t>(rng());
        auto packed = bytecrt::pack_mma_matrix(source, columns, depth);
        size_t cursor = 0;
        for (unsigned group = 0; group < (columns + 55) / 56; ++group)
            for (unsigned warp = 0; warp < 7; ++warp)
                for (unsigned k = 0; k < depth; k += 32)
                    for (unsigned lane = 0; lane < 32; ++lane)
                        for (unsigned half : {0u, 16u})
                            for (unsigned i = 0; i < 4; ++i) {
                                unsigned column = 56 * group + 8 * warp + lane / 4;
                                unsigned row = k + 4 * (lane % 4) + half + i;
                                unsigned expected = column < columns ? source[size_t(column) * depth + row] : 0;
                                assert(packed.at(cursor++) == expected);
                            }
        assert(cursor == packed.size());
    }
}
