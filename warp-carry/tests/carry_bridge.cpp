#include <array>
#include <barrier>
#include <cassert>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
#define __device__
#define __host__
#define __forceinline__ inline
#include "../warp-carry.cuh"

struct Warp {
    std::barrier<> barrier{32};
    std::array<uint32_t,32> values{};
};
struct Ops {
    Warp& w; unsigned lane;
    uint32_t ballot(bool value) const {
        w.values[lane]=value;
        w.barrier.arrive_and_wait();
        uint32_t mask=0;
        for(unsigned i=0;i<32;++i) mask |= w.values[i]<<i;
        w.barrier.arrive_and_wait();
        return mask;
    }
    uint32_t get(uint32_t value,unsigned src) const {
        w.values[lane]=value;
        w.barrier.arrive_and_wait();
        uint32_t result=w.values[src];
        w.barrier.arrive_and_wait();
        return result;
    }
};
template<unsigned W,unsigned L>
void run(unsigned count) {
    struct Case { std::array<uint32_t,L> low, high, out; uint32_t gap; };
    std::vector<Case> cases(count);
    for(auto& c:cases) {
        std::cin>>c.gap;
        for(auto& x:c.low) std::cin>>x;
        for(auto& x:c.high) std::cin>>x;
    }
    Warp w;std::vector<std::thread> lanes;
    for(unsigned lane=0;lane<32;++lane) lanes.emplace_back([&,lane]{
        for(auto& c:cases) bytecrt::warp_carry::finish<W,L>(c.low.data(),c.high.data(),c.gap,c.out.data(),lane,Ops{w,lane});
    });
    for(auto& t:lanes)t.join();
    for(auto& c:cases) {
        for(auto x:c.out) std::cout<<x<<' ';
        std::cout<<'\n';
    }
}
template<unsigned L> void check_scatter() {
    constexpr unsigned stride=(L+2)|1;
    uint32_t lo[32][stride],hi[32][stride];
    for(auto& row:lo) for(auto& v:row)v=0xdeadbeef;
    for(auto& row:hi) for(auto& v:row)v=0xdeadbeef;
    std::mt19937 rng(345);
    for(unsigned group=0;group<(L+13)/14;++group) {
        uint32_t dot[32][56];
        for(auto& row:dot) for(auto& d:row)d=rng()%30000000;
        for(unsigned warp=0;warp<7;++warp) for(unsigned lane=0;lane<32;++lane)
            for(unsigned part=0;part<4;++part) {
                unsigned row=lane/4+8*part,col=8*warp+2*(lane%4),other=col^2;
                bytecrt::warp_carry::store_dot<L>(lo,hi,lane,warp,group,part,dot[row][col],
                    dot[row][col+1],dot[row][other],dot[row][other+1]);
            }
        for(unsigned row=0;row<32;++row) for(unsigned j=0;j<14 && 14*group+j<L;++j) {
            uint64_t sum=0;
            for(unsigned byte=0;byte<4;++byte)sum+=uint64_t(dot[row][4*j+byte])<<(8*byte);
            assert(lo[row][14*group+j]==uint32_t(sum) && hi[row][14*group+j]==sum>>32);
        }
    }
    for(unsigned row=0;row<32;++row) for(unsigned j=L;j<stride;++j)
        assert(lo[row][j]==0xdeadbeef && hi[row][j]==0xdeadbeef);
}
int main() {
    check_scatter<55>();check_scatter<28>();
    std::mt19937 rng(13149);
    for(unsigned i=0;i<100000;++i) {
        uint32_t gen=rng(),prop=rng()&~gen,carry=i%2,mask=0;
        auto c=bytecrt::warp_carry::carries(gen,prop,carry);
        for(unsigned j=0;j<32;++j) {
            mask|=carry<<j;
            carry=(gen>>j&1)|((prop>>j&1)&carry);
        }
        assert(c.incoming==mask && c.outgoing==carry);
    }
    unsigned w,count;while(std::cin>>w>>count) {
        if(w==1747)run<1747,55>(count);
        else if(w==868)run<868,28>(count);
        else return 1;
    }
}
