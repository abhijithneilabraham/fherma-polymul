#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
#define __device__
#define __forceinline__ inline
#include "../alpha-sum.cuh"
int main() {
    unsigned count;
    while(std::cin>>count) {
        std::vector<uint32_t> primes(count),terms(count);
        for(auto& p:primes)std::cin>>p;
        for(auto& t:terms)std::cin>>t;
        uint32_t sum=0;
        for(unsigned j=0;j<count;++j) {
            uint32_t reciprocal=(uint64_t{1}<<42)/primes[j];
            sum+=bytecrt::warp_alpha::fraction(terms[j],reciprocal);
        }
        std::cout<<bytecrt::warp_alpha::round_sum(sum)<<"\n";
    }
}
