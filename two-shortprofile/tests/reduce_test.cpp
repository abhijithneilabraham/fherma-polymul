#include "../two-primes.hpp"
#include "../word-reduce.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <random>

template<unsigned Pad,size_t Count>
void check(const two_digit::PrimeRoot (&roots)[Count]) {
    constexpr uint32_t C=Pad*255u*255u;
    constexpr uint64_t high_bound=uint64_t(C)*257+(uint64_t(C)*257>>16);
    static_assert(high_bound<UINT32_MAX);
    std::mt19937 rng(Pad);
    unsigned cases=0;
    for(auto root:roots) {
        uint32_t p=root.p,mu=(uint64_t{1}<<48)/p;
        // Worst possible deficit of the two truncations is strictly below one.
        assert((uint64_t(65535)<<32)+high_bound*p<(uint64_t(p)<<32));
        auto verify=[&](std::array<int32_t,4> sums) {
            uint64_t x=0;
            for(unsigned j=0;j<4;++j)x+=uint64_t(sums[j])<<(8*j);
            assert(two_digit::reduce_dot(sums.data(),p,mu)==x%p);
            ++cases;
        };
        for(unsigned mask=0;mask<16;++mask){
            std::array<int32_t,4> s{};
            for(unsigned j=0;j<4;++j)s[j]=(mask&(1<<j))?C:0;
            verify(s);
        }
        for(unsigned i=0;i<2048;++i){
            std::array<int32_t,4> s{};
            for(auto& v:s)v=rng()%(C+1);
            verify(s);
        }
        // Construct exact multiples and their neighbors, including 16-bit carries.
        for(unsigned i=1;i<1024;++i)for(int delta:{-1,0,1}){
            uint64_t x=uint64_t(i)*p+delta;
            verify({int32_t(x&255),int32_t((x>>8)&255),int32_t((x>>16)&255),int32_t(x>>24)});
        }
    }
    std::cout<<"PASS "<<cases<<" byte-dot residues at "<<Count<<" primes\n";
}
int main(){check<128>(two_digit::large);check<64>(two_digit::small);}
