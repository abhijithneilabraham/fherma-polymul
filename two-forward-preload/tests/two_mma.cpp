#include <array>
#include <barrier>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
#include "../two-plan.hpp"
#include "host-split.hpp"
#include "../word-reduce.hpp"
#define __device__
#define __forceinline__ inline
#include "../tile-fragments.cuh"
using U = uint32_t;
using namespace bytecrt::resident_tile;

struct Warp {
    std::barrier<> barrier{32};
    U a[32][4], b[32][2];
};
struct EmulatedMma {
    unsigned lane;
    Warp* warp;
    void operator()(int32_t* c,U a0,U a1,U a2,U a3,U b0,U b1) const {
        auto& w=*warp;
        w.a[lane][0]=a0;w.a[lane][1]=a1;w.a[lane][2]=a2;w.a[lane][3]=a3;
        w.b[lane][0]=b0;w.b[lane][1]=b1;
        w.barrier.arrive_and_wait();
        for(unsigned out=0;out<4;++out) {
            unsigned row=lane/4+8*(out/2), col=2*(lane%4)+(out%2);
            for(unsigned k=0;k<32;++k) {
                U av=w.a[4*(row%8)+(k%16)/4][row/8+2*(k/16)];
                U bv=w.b[4*col+(k%16)/4][k/16];
                c[out]+=((av>>(8*(k%4)))&255)*((bv>>(8*(k%4)))&255);
            }
        }
        w.barrier.arrive_and_wait();
    }
};

template<unsigned W,unsigned L,unsigned Count,unsigned Pad,unsigned Split>
void check(unsigned n,uint32_t gap) {
    constexpr unsigned stride=Pad/4+4;
    std::vector<U> q(L,UINT32_MAX); q.front()=0u-gap;q.back()=(1u<<(W%32))-1;
    auto plan=two_digit::make(n,W,q);
    std::mt19937 rng(12753);
    U digits[32][stride]{};
    for(unsigned row=0;row<16;++row) {
        U raw[L];for(auto& x:raw)x=rng();
        raw[L-1]&=(1u<<(W%32))-1;
        auto loaded=split_row<W,L,Pad/4,Split>(raw);
        for(unsigned k=0;k<Pad/4;++k) {
            digits[row][k]=loaded[k].first;
            digits[16+row][k]=loaded[k].second;
        }
    }
    unsigned checked=0;
    for(unsigned warp_id=0;warp_id<7;++warp_id) for(unsigned pair=warp_id;pair<Count/2;pair+=7) {
        Warp warp;
        int32_t acc[32][1][8]{};
        std::vector<std::thread> lanes;
        for(unsigned lane=0;lane<32;++lane) lanes.emplace_back([&,lane] {
            accumulate<Count,Pad>(&digits[0][0],stride,plan.encode.data(),pair,lane,acc[lane],EmulatedMma{lane,&warp});
        });
        for(auto& lane:lanes)lane.join();
        for(unsigned lane=0;lane<32;lane+=2) for(unsigned h=0;h<2;++h) {
            unsigned j=2*pair+(lane%4)/2, row=lane/4+8*h;
            U p=plan.primes[j], actual[2],ref[2]{};
            for(unsigned side=0;side<2;++side) {
                uint64_t total=0;
                for(unsigned b=0;b<4;++b) total+=uint64_t(acc[lane+b/2][0][4*side+2*h+b%2])<<(8*b);
                int32_t sums[4];
                for(unsigned b=0;b<4;++b)sums[b]=acc[lane+b/2][0][4*side+2*h+b%2];
                actual[side]=two_digit::reduce_dot(sums,p,plan.reciprocal[j]);
                assert(actual[side]==total%p);
                for(unsigned k=stride;k-->0;) ref[side]=((uint64_t(ref[side])<<32)+digits[16*side+row][k])%p;
            }
            auto m=bytecrt::warp_ntt::make_mod(p,1);
            U hi=bytecrt::warp_ntt::multiply_twiddle(actual[1],plan.theta[j],m);
            uint64_t product=uint64_t(ref[1])*plan.theta[j].value%p;
            assert(bytecrt::warp_ntt::add(actual[0],hi,p)==(ref[0]+product)%p);
            assert(bytecrt::warp_ntt::sub(actual[0],hi,p)==(ref[0]+p-product)%p);
            checked+=2;
        }
    }
    assert(checked==32*Count);
    std::cout<<"PASS "<<checked<<" two-digit tensor evaluation planes\n";
}
int main(){check<1747,55,58,128,110>(65536,9699327); check<868,28,30,64,55>(32768,43646975);}
