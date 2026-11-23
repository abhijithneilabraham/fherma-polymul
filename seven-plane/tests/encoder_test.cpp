#include "../encode.hpp"
#include <barrier>
#include <cassert>
#include <iostream>
#include <thread>
#include <vector>

using namespace seven_encode;
struct Warp {
    std::barrier<> sync{32};
    U a[32][4], b[32][2];
    int32_t shuffle[32];
};
struct Emulated {
    unsigned lane;
    Warp* state;
    void mma(int32_t* acc,U a0,U a1,U a2,U a3,U b0,U b1) const {
        auto& w=*state;
        w.a[lane][0]=a0; w.a[lane][1]=a1; w.a[lane][2]=a2; w.a[lane][3]=a3;
        w.b[lane][0]=b0; w.b[lane][1]=b1;
        w.sync.arrive_and_wait();
        for(unsigned out=0;out<4;++out) {
            unsigned row=lane/4+8*(out/2), col=2*(lane%4)+(out%2);
            for(unsigned k=0;k<32;++k) {
                U av=w.a[4*(row%8)+(k%16)/4][row/8+2*(k/16)];
                U bv=w.b[4*col+(k%16)/4][k/16];
                acc[out]+=((av>>(8*(k%4)))&255)*((bv>>(8*(k%4)))&255);
            }
        }
        w.sync.arrive_and_wait();
    }
    int32_t neighbor(int32_t x) const {
        state->shuffle[lane]=x;
        state->sync.arrive_and_wait();
        auto other=state->shuffle[lane^1];
        state->sync.arrive_and_wait();
        return other;
    }
    U reduce(const int32_t* c,const Field& f) const {
        uint64_t x=0;
        for(unsigned b=0;b<4;++b)x+=uint64_t(c[b])<<(8*b);
        return x%f.mod.p;
    }
};

template<unsigned W,unsigned Count>
int check() {
    constexpr unsigned L=(W+31)/32;
    U raw[16][L+1]{};
    for(auto& row:raw)for(unsigned i=0;i<L;++i)std::cin>>row[i];
    Field fields[Count];
    U matrix[Count*4][8]{};
    for(unsigned j=0;j<Count;++j) {
        U p,w;
        std::cin>>p>>w;
        auto& f=fields[j];
        f.mod=make_mod(p,1); f.forward=prepare(w,f.mod);
        for(auto& t:f.digit_twist) {
            std::cin>>t.value;t.reciprocal=(uint64_t(t.value)<<32)/p;
        }
        U power=1;
        for(unsigned byte=0;byte<32;++byte) {
            for(unsigned b=0;b<4;++b)
                matrix[4*j+b][byte/4]|=((power>>(8*b))&255)<<(8*(byte%4));
            power=uint64_t(power)*256%p;
        }
    }
    if(!std::cin)return 2;
    for(unsigned pair=0;pair<Count/2;++pair) {
        Warp warp;
        U values[32][2][7]{};
        std::vector<std::thread> lanes;
        for(unsigned lane=0;lane<32;++lane)lanes.emplace_back([&,lane] {
            unsigned j=2*pair+(lane%4)/2;
            U* weights=matrix[8*pair+lane/4];
            evaluate<W,L>(raw,lane,fields[j],weights[lane%4],weights[4+lane%4],values[lane],Emulated{lane,&warp});
        });
        for(auto& thread:lanes)thread.join();
        for(unsigned lane=0;lane<32;lane+=2)for(unsigned h=0;h<2;++h) {
            std::cout<<2*pair+(lane%4)/2<<' '<<lane/4+8*h;
            for(auto y:values[lane][h])std::cout<<' '<<y;
            std::cout<<'\n';
        }
    }
    return 0;
}
int main() {
    unsigned w;std::cin>>w;
    return w==1747 ? check<1747,18>():check<868,10>();
}
