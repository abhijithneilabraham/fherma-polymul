#include "../tensor-weights.hpp"
#include <array>
#include <barrier>
#include <cassert>
#include <iostream>
#include <random>
#include <thread>
#define __device__
#define __forceinline__ inline
#include "../tensor-radix.cuh"
using namespace bytecrt::warp_ntt;
using U=uint32_t;
using Row=std::array<U,256>;

struct State {
    std::barrier<> barrier{32};
    U a[32][4],b[32][2];
    int32_t values[32];
};
struct CpuOps {
    State& w; unsigned lane;
    void sync() const { w.barrier.arrive_and_wait(); }
    int32_t neighbor(int32_t v) const {
        w.values[lane]=v; sync();
        auto x=w.values[lane^1]; sync(); return x;
    }
    U reduce(const int32_t* d,U p,uint64_t) const {
        uint64_t s=0;
        for (unsigned i=0;i<4;++i) s+=uint64_t(d[i])<<(8*i);
        return s%p;
    }
    void mma(int32_t* c,const U* a,const U* b) const {
        for (unsigned i=0;i<4;++i) w.a[lane][i]=a[i];
        w.b[lane][0]=b[0]; w.b[lane][1]=b[4]; sync();
        for(unsigned output=0;output<4;++output) {
            unsigned r=lane/4+8*(output/2),col=2*(lane%4)+output%2;
            for(unsigned k=0;k<32;++k) {
                U av=w.a[4*(r%8)+(k%16)/4][r/8+2*(k/16)];
                U bv=w.b[4*col+(k%16)/4][k/16];
                c[output]+=((av>>(8*(k%4)))&255)*((bv>>(8*(k%4)))&255);
            }
        }
        sync();
    }
};
Row transform(const Row& x,const U* weights,const Twiddle* roots,Mod m) {
    TensorRow s{}; State w;
    U padded[64*20]{};
    for (unsigned i=0;i<1024;++i) padded[(i/16)*20+i%16]=weights[i];
    for(unsigned i=0;i<256;++i) s.word[i%16][i/16]=x[i];
    std::vector<std::thread> threads;
    for(unsigned lane=0;lane<32;++lane) threads.emplace_back([&,lane] {
        CpuOps ops{w,lane};
        radix16<true>(s,padded,roots,m,UINT64_MAX/m.p,lane,ops);
        radix16<false>(s,padded,roots,m,UINT64_MAX/m.p,lane,ops);
    });
    for(auto& thread:threads) thread.join();
    Row result;
    for(unsigned i=0;i<256;++i) result[i]=s.word[i%16][i/16];
    return result;
}
int main() {
    std::mt19937 rng(13579);
    for(unsigned N : {32768u,65536u}) {
        unsigned W=N==65536?1747:868;
        std::vector<U> q((W+31)/32,UINT32_MAX);
        q.front()=0u-(N==65536?9699327u:43646975u);
        q.back()=(1u<<(W%32))-1;
        auto plan=bytecrt::make_rns_plan(N,W,q);
        auto weights=tensor_weights(plan);
        for(unsigned pidx=0;pidx<plan.pcount;++pidx) {
            U p=plan.primes[pidx];
            U omega=bytecrt::rns_detail::pow_mod(plan.omega_roots[pidx],N/256,p);
            Twiddle roots[256];
            U v=1;
            for(auto& t:roots) {
                t={v,U((uint64_t(v)<<32)/p)};
                v=uint64_t(v)*omega%p;
            }
            const U* packed=weights.data()+pidx*1024;
            const auto* bytes=reinterpret_cast<const uint8_t*>(packed);
            for(unsigned k=0;k<16;++k) for(unsigned j=0;j<16;++j) {
                U w=roots[(16*k*j)%256].value;
                for(unsigned d=0;d<4;++d) {
                    U rebuilt=0;
                    for(unsigned b=0;b<4;++b) rebuilt |= U(bytes[(4*k+b)*64+4*j+d]) << (8*b);
                    assert(rebuilt==w); w=uint64_t(w)*256%p;
                }
            }
            if(pidx!=0 && pidx!=plan.pcount/2 && pidx+1!=plan.pcount) continue;
            Mod m=make_mod(p,plan.n_inverse[pidx]);
            Row a,b,expected{};
            for(auto& x:a) x=rng()%p;
            for(auto& x:b) x=rng()%p;
            for(unsigned i=0;i<256;++i) for(unsigned j=0;j<256;++j)
                expected[(i+j)%256]=(expected[(i+j)%256]+uint64_t(a[i])*b[j])%p;
            Row fa=transform(a,packed,roots,m),fb=transform(b,packed,roots,m);
            for(unsigned k=0;k<256;++k) {
                U ref=0;
                for(unsigned j=0;j<256;++j) ref=(ref+uint64_t(a[j])*roots[(j*k)%256].value)%p;
                assert(ref==fa[k]);
                fa[k]=multiply(fa[k],fb[k],m);
            }
            Row out=transform(fa,packed,roots,m);
            U R=(uint64_t(1)<<32)%p;
            for(unsigned j=0;j<256;++j)
                assert(uint64_t(out[(256-j)%256])*R%p == uint64_t(expected[j])*256%p);
        }
        std::cout<<"PASS "<<plan.pcount<<" prime weight tables and three cyclic convolutions through both tensor passes\n";
    }
}
