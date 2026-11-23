#include <array>
#include <cstdint>
#include <iostream>
#define __device__
#define __forceinline__ inline
#include "../short-fold.cuh"
template<unsigned W,unsigned L>
void run(uint32_t gap) {
    std::array<uint32_t,L+2> a;
    for (auto& x : a) std::cin >> x;
    bytecrt::short_fold::reduce<W,L>(a.data(),gap);
    for (auto x : a) std::cout << x << ' ';
    std::cout << '\n';
}
int main() {
    unsigned w;
    uint32_t gap;
    while (std::cin >> w >> gap) {
        if (w==1747) run<1747,55>(gap);
        else if (w==868) run<868,28>(gap);
        else return 1;
    }
}
