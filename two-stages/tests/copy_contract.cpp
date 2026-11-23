#include "../host_copy.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>

#ifdef HOST_COPY_PARALLEL_CHECK
static unsigned observed_parallel = 0;
extern "C" void bytecrt_vector_copy_observe(int, bool parallel, std::size_t) {
    if (parallel) ++observed_parallel;
}
#endif
int main() {
    std::mt19937 rng(20261122);
    unsigned cases = 0;
    for (size_t n : {size_t(0), size_t(1), size_t(17), size_t(1025), size_t(400001)}) {
        std::vector<uint32_t> source(n), staged(n + 2, 0x9e3779b9u);
        for (unsigned draw = 0; draw < 4; ++draw) {
            for (size_t i = 0; i < n; ++i)
                source[i] = draw == 0 ? 0 : draw == 1 ? UINT32_MAX : draw == 2 ? uint32_t(1) << (i % 32) : rng();
            host_copy::upload_bytes(staged.data() + 1, source);
            assert(staged.front() == 0x9e3779b9u && staged.back() == 0x9e3779b9u);
            assert(std::equal(source.begin(), source.end(), staged.begin() + 1));
            fherma::Outputs metadata;
            metadata.c.shape = {1, int64_t(n), 1};
            assert(metadata.c.data.empty());
            auto owned = host_copy::download_bytes(std::move(metadata), staged.data() + 1, n);
            assert((owned.c.shape == std::vector<int64_t>{1, int64_t(n), 1}));
            assert(owned.c.data == source);
            std::fill(staged.begin() + 1, staged.end() - 1, 0x12345678u);
            assert(owned.c.data == source); // independent ownership, not a staging alias
            ++cases;
        }
    }
    fherma::Outputs nonempty;
    nonempty.c.data = {7};
    bool rejected = false;
    try { (void)host_copy::download_bytes(std::move(nonempty), nullptr, 0); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
#ifdef HOST_COPY_PARALLEL_CHECK
    assert(observed_parallel > 0);
#endif
    std::cout << "PASS " << cases << " raw-copy/ownership fixtures; no CUDA or polynomial arithmetic\n";
}
