#include <array>
#include <cassert>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
static std::array<std::array<float,32>,6> states;
static unsigned current_lane, depth;
static float __fadd_rn(float a,float b) { volatile float r=a+b;return r; }
static float __fmul_rn(float a,float b) { volatile float r=a*b;return r; }
static float __uint2float_rn(uint32_t a) { volatile float r=static_cast<float>(a);return r; }
static float __shfl_down_sync(unsigned mask,float value,unsigned delta) {
    assert(mask==0xffffffffu && depth<5 && delta==(16u>>depth));
    assert(value==states[depth][current_lane]);
    const unsigned source=current_lane+delta<32 ? current_lane+delta : current_lane;
    return states[depth++][source];
}
#define __device__
#define __forceinline__ inline
#include "../alpha-sum.cuh"
int main() {
    assert(std::fegetround()==FE_TONEAREST);
    unsigned P;
    while(std::cin>>P) {
        assert(P>0 && P<=114);
        std::vector<uint32_t> primes(P),terms(P);std::vector<float> inverse(P);
        for(auto& p:primes)std::cin>>p;
        for(auto& t:terms)std::cin>>t;
        assert(std::cin.good());
        for(unsigned j=0;j<P;++j){assert(terms[j]<primes[j]);inverse[j]=static_cast<float>(1.0/double(primes[j]));}
        for(unsigned lane=0;lane<32;++lane)
            states[0][lane]=bytecrt::warp_alpha::partial(terms.data(),inverse.data(),P,lane);
        for(unsigned step=0;step<5;++step)
            for(unsigned lane=0;lane<32;++lane){
                unsigned neighbor=lane+(16u>>step);
                states[step+1][lane]=__fadd_rn(states[step][lane],states[step][neighbor<32?neighbor:lane]);
            }
        float result=0;
        for(unsigned lane=0;lane<32;++lane){
            current_lane=lane;depth=0;
            float value=bytecrt::warp_alpha::sum(terms.data(),inverse.data(),P,lane);
            assert(depth==5 && value==states[5][lane]);
            if(!lane)result=value;
        }
        std::cout<<uint32_t(std::nearbyint(result))<<'\n';
    }
}
