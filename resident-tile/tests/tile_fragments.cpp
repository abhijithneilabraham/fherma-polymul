#include <array>
#include <barrier>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
#include "../core/rns-plan.hpp"
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
template<unsigned L,unsigned P,unsigned Kpad>
void check(unsigned N,unsigned W,U gap) {
    constexpr unsigned stride=Kpad/4+4;
    std::vector<U> q(L,UINT32_MAX); q.front()=0u-gap;q.back()=(1u<<(W%32))-1;
    auto plan=bytecrt::make_rns_plan(N,W,q);
    std::vector<U> packed(P*Kpad);
    auto* weights=reinterpret_cast<uint8_t*>(packed.data());
    for(unsigned p=0;p<P;++p) {
        U power=1;
        for(unsigned k=0;k<4*L;++k) {
            for(unsigned d=0;d<4;++d) weights[(4*p+d)*Kpad+k]=power>>(8*d);
            power=uint64_t(power)*256%plan.primes[p];
        }
    }
    std::mt19937 rng(3589);
    U input[32][stride]{};
    for(unsigned r=0;r<32;++r) for(unsigned l=0;l<L;++l)
        input[r][l]=r==0?0:r==1?UINT32_MAX:rng();
    std::array<std::array<bool,P>,32> seen{};
    for(unsigned physical=0;physical<4;++physical) {
        for(unsigned first=pairs_per_warp*physical;first<(P+1)/2;first+=4*pairs_per_warp) {
            Warp warp;
            int32_t acc[32][pairs_per_warp][8]{};
            std::vector<std::thread> lanes;
            for(unsigned lane=0;lane<32;++lane) lanes.emplace_back([&,lane] {
                accumulate<P,Kpad>(&input[0][0],stride,weights,first,lane,acc[lane],EmulatedMma{lane,&warp});
            });
            for(auto& thread:lanes) thread.join();
            for(unsigned j=0;j<pairs_per_warp;++j) for(unsigned lane=0;lane<32;lane+=2)
                for(unsigned quarter=0;quarter<4;++quarter) {
                    unsigned p=2*(first+j)+(lane%4)/2;
                    if(p>=P) continue;
                    unsigned row=lane/4+8*quarter;
                    assert(!seen[row][p]);seen[row][p]=true;
                    uint64_t sum=0;
                    for(unsigned d=0;d<4;++d)
                        sum+=uint64_t(acc[lane+d/2][j][2*quarter+d%2])<<(8*d);
                    U ref=0;
                    for(unsigned l=L;l-->0;) ref=((uint64_t(ref)<<32)+input[row][l])%plan.primes[p];
                    assert(sum%plan.primes[p]==ref);
                }
        }
    }
    for(auto& row:seen) for(bool visited:row) assert(visited);
    std::cout<<"PASS "<<32*P<<" prime residues from actual fragment loads\n";
}
int main() {check<55,114,224>(65536,1747,9699327); check<28,57,128>(32768,868,43646975);}
