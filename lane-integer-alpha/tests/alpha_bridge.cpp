#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
static std::array<uint32_t,32> partials;
static unsigned current_lane;
static uint32_t __reduce_add_sync(unsigned mask,uint32_t value) {
    assert(mask==UINT32_MAX && value==partials[current_lane]);
    uint32_t result=0;for(auto x:partials)result+=x;
    return result;
}
#define __device__
#define __forceinline__ inline
#include "../alpha-sum.cuh"
int main() {
    unsigned count;
    while(std::cin>>count) {
        assert(count>0 && count<=114);
        std::vector<uint32_t> primes(count),terms(count),recip(count);
        for(auto& p:primes)std::cin>>p;
        for(auto& t:terms)std::cin>>t;
        for(unsigned j=0;j<count;++j) {
            assert(primes[j]>(1u<<23) && terms[j]<primes[j]);
            recip[j]=(uint64_t{1}<<55)/primes[j];
        }
        for(unsigned i=0;i<32;++i)
            partials[i]=bytecrt::warp_alpha::partial(terms.data(),recip.data(),count,i);
        std::cout<<bytecrt::warp_alpha::sum(terms.data(),recip.data(),count,current_lane)<<"\n";
    }
}
