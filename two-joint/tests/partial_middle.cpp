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
#include "../two-plan.hpp"

struct Index { unsigned x,y; };
Index threadIdx{},blockIdx{},blockDim{};
namespace sim {
unsigned threads;
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
    sim::warps[i].shuffle[lane]=v;sim::barrier(i);
    uint32_t peer=sim::warps[i].shuffle[lane^d];sim::barrier(i);return peer;
}

#define __launch_bounds__(...)
#define __device__
#define __forceinline__ inline
#define __global__
#define __shared__ static
#define __syncthreads() sim::barrier(32)
#include "../warp-stages.cuh"

template<unsigned N,unsigned W,unsigned Count>
void check(){
    using namespace bytecrt::warp_ntt;
    using bytecrt::rns_detail::pow_mod;
    constexpr unsigned K=N/256,low_count=K/32,first=K-32,P=2*Count,prime=P-1;
    std::vector<uint32_t> q((W+31)/32,UINT32_MAX);
    q.front()=0u-(W==1747?9699327u:43646975u);q.back()=(1u<<(W%32))-1;
    auto plan=two_digit::make(N,W,q);auto t=make_tables(plan.transform);
    std::vector<uint32_t> input(size_t{2}*P*N),out(size_t{P}*N,UINT32_MAX);
    std::array<std::array<uint32_t,256>,32> expected{};
    uint32_t p=plan.transform.primes[prime],psi=plan.transform.psi_roots[prime];
    uint32_t rinv=pow_mod((uint64_t{1}<<32)%p,p-2,p);
    std::mt19937 rng(W);
    for(unsigned r=0;r<32;++r){
        auto* a=input.data()+size_t(prime)*N+(first+r)*256;
        auto* b=a+size_t(P)*N;
        for(unsigned id=0;id<256;++id){a[id]=rng()%p;b[id]=rng()%p;}
        for(unsigned i=0;i<256;++i){
            uint32_t v=0;
            for(unsigned j=0;j<256;++j)v=(v+uint64_t(a[j])*b[(i+256-j)%256])%p;
            expected[r][i]=uint64_t(v)*256%p*rinv%p*rinv%p*t.inverse_phase[(size_t(prime)*K+first+r)*256+i]%p;
        }
    }
    for(unsigned distance=1;distance<low_count;distance*=2)for(unsigned i=0;i<32;++i){
        if(i&distance)continue;
        unsigned groups=K/(2*distance),index=(first+i)/(2*distance);
        unsigned exponent=256*distance*(2*reversed(index,groups)+1);
        uint32_t root=pow_mod(psi,2*N-exponent,p);
        for(unsigned id=0;id<256;++id){
            uint32_t a=expected[i][id],b=expected[i+distance][id];
            expected[i][id]=(uint64_t(a)+b)%p;
            expected[i+distance][id]=(uint64_t(a)+p-b)%p*root%p;
        }
    }
    blockIdx={first/32,prime};
    sim::launch(1024,[&]{middle<N,P>(input.data(),out.data(),t.inverse_phase.data(),t.roots.data(),t.mods.data());});
    for(unsigned r=0;r<32;++r)for(unsigned id=0;id<256;++id){
        unsigned low=(first+r)%low_count,coarse=(first+r)/low_count;
        size_t pos=(low*128+id/2)*64+(coarse/4)*8+4*(id%2)+coarse%4;
        auto& value=out[size_t(prime)*N+pos];
        assert(value==expected[r][id]);value=UINT32_MAX;
    }
    for(auto v:out)assert(v==UINT32_MAX);
    std::vector<bool> visited(N,false);
    for(unsigned f=0;f<K;f+=32)for(unsigned start=0;start<8192;start+=32) {
        std::array<unsigned,32> sectors{};
        for(unsigned lane=0;lane<32;++lane){
            unsigned i=start+lane,id=2*((i/8)%128)+(i%8)/4;
            unsigned low=(i/1024)%low_count,coarse=4*(i/(1024*low_count))+i%4;
            unsigned addr=partial_index(id,low,f/low_count+coarse);
            assert(addr<N && !visited[addr]);visited[addr]=true;
            sectors[lane]=addr/8;
        }
        std::sort(sectors.begin(),sectors.end());
        assert(std::unique(sectors.begin(),sectors.end())-sectors.begin()==4);
    }
    for(bool seen:visited)assert(seen);
    for(unsigned id=0;id<256;id+=2)for(unsigned low=0;low<low_count;++low)for(unsigned j=0;j<2;++j){
        unsigned begin=partial_index(id,low,16*j);
        assert(begin%32==0);
        for(unsigned l=0;l<32;++l)assert(partial_index(id+(l%8)/4,low,16*j+4*(l/8)+l%4)==begin+l);
    }
    std::cout<<"PASS 8,192 partial inverse outputs at N="<<N<<"\n";
}
int main(){check<32768,868,30>();check<65536,1747,58>();}
