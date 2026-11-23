// Run the submitted block on cooperative CPU fibers, including every barrier.
#define _XOPEN_SOURCE 600
#include <ucontext.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <iostream>
#include <random>
#include <vector>
#include "two-plan.hpp"

struct Index { unsigned x,y; };
Index threadIdx{},blockIdx{},blockDim{};
namespace sim {
unsigned threads, shuffles;
struct Lane { ucontext_t ctx{}; std::array<char,32768> stack; int waiting=-1; };
std::vector<Lane> lanes(1024);
ucontext_t scheduler;
unsigned arrivals[33]{};
std::function<void()> kernel;
void invoke() { kernel(); lanes[threadIdx.x].waiting=-2; }
void barrier(unsigned id) {
    unsigned i=threadIdx.x;
    lanes[i].waiting=id;
    if(++arrivals[id]==(id==32?threads:32)) {
        for(unsigned j=(id==32?0:32*id),end=(id==32?threads:j+32);j<end;++j) {
            assert(lanes[j].waiting==int(id));lanes[j].waiting=-1;
        }
        arrivals[id]=0;
    }
    swapcontext(&lanes[i].ctx,&scheduler);
}
void launch(unsigned count, std::function<void()> body) {
    threads=count; blockDim={count,1}; kernel=std::move(body);
    for(unsigned i=0;i<count;++i) {
        auto& l=lanes[i];
        getcontext(&l.ctx);
        l.ctx.uc_stack.ss_sp=l.stack.data();l.ctx.uc_stack.ss_size=l.stack.size();
        l.ctx.uc_stack.ss_flags=0;l.ctx.uc_link=&scheduler;l.waiting=-1;
        makecontext(&l.ctx,invoke,0);
    }
    for(;;) {
        unsigned live=0,ready=0;
        for(unsigned i=0;i<threads;++i) {
            live+=lanes[i].waiting!=-2;
            if(lanes[i].waiting==-1){++ready;threadIdx.x=i;swapcontext(&scheduler,&lanes[i].ctx);}
        }
        if(!live)break;
        assert(ready && "kernel barrier deadlock");
    }
}
struct Warp {uint32_t shuffle[32],a[32][4],b[32][2];};
Warp warps[32];
}
uint32_t __shfl_xor_sync(unsigned mask,uint32_t v,unsigned d) {
    unsigned i=threadIdx.x/32,lane=threadIdx.x%32;
    if(lane==0)++sim::shuffles;
    sim::warps[i].shuffle[lane]=v;sim::barrier(i);
    uint32_t peer=sim::warps[i].shuffle[lane^d];sim::barrier(i);return peer;
}

#define __launch_bounds__(...)
#define __global__
#define __device__
#define __forceinline__ inline
#define __shared__ static
#define __syncthreads() sim::barrier(32)
#include "warp-stages.cuh"

template<unsigned N,unsigned W,unsigned P>
void check() {
    using namespace bytecrt::warp_ntt;
    using bytecrt::rns_detail::pow_mod;
    constexpr unsigned K=N/256;
    std::vector<uint32_t> q((W+31)/32,UINT32_MAX);
    q.front()=0u-(W==1747?9699327u:43646975u);q.back()=(1u<<(W%32))-1;
    auto plan=two_digit::make(N,W,q);auto tables=make_tables(plan.transform);
    std::vector<uint32_t> in(size_t(2)*P*N),out(in.size(),UINT32_MAX);
#ifndef CONTROL_FORWARD
    // Check every warp's eight-column shared transpose, in both directions.
    for(unsigned base=0;base<K;base+=32)for(unsigned w=0;w<8;++w) {
        std::array<unsigned,32> producers{},consumers{};
        for(unsigned lane=0;lane<32;++lane) {
            unsigned row=natural_row<K>(lane,base/32);
            ++producers[(w*(K+4)+tile_row<K>(row))%32];
            unsigned word=8*base+32*w+lane;
            ++consumers[((word%8)*(K+4)+tile_row<K>(word/8))%32];
        }
        for(unsigned i=0;i<32;++i)assert(producers[i]==1 && consumers[i]==1);
    }
#endif
    constexpr unsigned expected_shuffles=
#ifdef CONTROL_FORWARD
        5*(K/32);
#else
        K==256?28:16;
#endif
    std::mt19937 rng(N);
    for(unsigned plane:{0u,P/2,P+P-1})for(unsigned first:{0u,248u}) {
        unsigned prime=plane%P;uint32_t p=plan.transform.primes[prime],psi=plan.transform.psi_roots[prime];
        for(unsigned col=first;col<first+8;++col)for(unsigned j=0;j<K;++j)
            in[size_t(plane)*N+col*K+j]=j%7==0?p-1:rng()%p;
        blockIdx={first/8,plane};sim::shuffles=0;
        sim::launch(256,[&]{forward<N,P>(in.data(),out.data(),tables.forward_phase.data(),tables.roots.data(),tables.mods.data());});
        assert(sim::shuffles==8*expected_shuffles);
        for(unsigned col=first;col<first+8;++col)for(unsigned row=0;row<K;++row) {
            unsigned exponent=2*reversed(row,K)+1;
            auto point=pow_mod(psi,256*exponent,p);
            uint32_t value=0;
            for(unsigned j=K;j-->0;)value=(uint64_t(value)*point+in[size_t(plane)*N+col*K+j])%p;
            auto& observed=out[size_t(plane)*N+row*256+col];
            assert(observed==uint64_t(value)*pow_mod(psi,col*exponent,p)%p);
            observed=UINT32_MAX;
        }
        for(auto word:out)assert(word==UINT32_MAX);
    }
    std::cout<<"PASS "<<6*8*K<<" forward outputs and "<<expected_shuffles<<" exchanges per warp at N="<<N<<"\n";
}
int main(){check<32768,868,60>();check<65536,1747,116>();}
