// Execute a whole inverse block with CPU barriers and independent input spectra.
#include <array>
#include <barrier>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

struct Index { unsigned x=0,y=0; };
thread_local Index threadIdx;
Index blockIdx;
struct Shuffle { std::barrier<> sync{32}; uint32_t value[32]; };
std::array<Shuffle,8> warps;
std::barrier<> block_sync{256};
uint32_t __shfl_xor_sync(unsigned mask,uint32_t value,unsigned distance) {
    assert(mask==0xffffffffu);
    auto& warp=warps[threadIdx.x/32];
    unsigned lane=threadIdx.x%32;
    warp.value[lane]=value;
    warp.sync.arrive_and_wait();
    auto other=warp.value[lane^distance];
    warp.sync.arrive_and_wait();
    return other;
}
#define __global__
#define __device__
#define __shared__ static
#define __forceinline__ inline
#define __syncthreads() block_sync.arrive_and_wait()
const Index blockDim{256,1};
unsigned __brev(unsigned x) { unsigned y=0; for(unsigned i=0;i<32;++i,x>>=1)y=2*y+(x&1);return y; }
#include "../warp-stages.cuh"
#include "../seven-plan.hpp"

using namespace bytecrt::warp_ntt;
using bytecrt::rns_detail::pow_mod;
using U=uint32_t;
template<class Func> void launch(Func f) {
    std::vector<std::thread> threads;
    for(unsigned i=0;i<256;++i)threads.emplace_back([&,i]{threadIdx.x=i;f();});
    for(auto& t:threads)t.join();
}
template<unsigned N,unsigned W,unsigned Count>
void check(){
    constexpr unsigned K=32,J=N/K,L=(W+31)/32,P=7*Count;
    unsigned gap=W==1747?9699327:43646975;
    std::vector<U> q(L,UINT32_MAX);q.front()=0u-gap;q.back()=(1u<<(W%32))-1;
    auto plan=seven_plan::make(N,W,q);auto t=make_tables(plan.transform);
    U prime=Count-1,p=plan.transform.primes[prime],psi=plan.transform.psi_roots[prime];
    U rinv=pow_mod((uint64_t(1)<<32)%p,p-2,p);
    std::mt19937 rng(N);
    std::vector<U> input(size_t{2}*P*N),output(input.size(),UINT32_MAX);
    unsigned lastplane=2*P-1,lastrow=K-1;
    for(unsigned id=J-8;id<J;++id)
        for(unsigned i=0;i<K;++i)input[size_t(lastplane)*N+id*K+i]=rng()%p;
    blockIdx={14*(J/8-1)+13,prime};
    launch([&]{forward<N,P>(input.data(),output.data(),t.forward_phase.data(),t.roots.data(),t.mods.data());});
    for(unsigned id=J-8;id<J;++id)for(unsigned row=0;row<K;++row){
        unsigned freq=2*reversed(row,K)+1;
        U x=pow_mod(psi,J*freq,p),v=0;
        for(unsigned i=K;i-->0;)v=(uint64_t(v)*x+input[size_t(lastplane)*N+id*K+i])%p;
        auto& actual=output[size_t(lastplane)*N+row*J+id];
        assert(actual==uint64_t(v)*pow_mod(psi,id*freq,p)%p);
        actual=UINT32_MAX;
    }
    for(auto x:output)assert(x==UINT32_MAX);
    std::vector<U> a(J),b(J),expected(J);
    for(unsigned i=0;i<J;++i) {
        a[i]=rng()%p;b[i]=rng()%p;
        input[(size_t(P-1)*K+lastrow)*J+i]=a[i];
        input[(size_t(2*P-1)*K+lastrow)*J+i]=b[i];
    }
    for(unsigned j=0;j<J;++j)for(unsigned i=0;i<J;++i) {
        auto& x=expected[(i+j)%J];x=(x+uint64_t(a[i])*b[j])%p;
    }
    blockIdx={lastrow,P-1};
    launch([&]{middle<N,P>(input.data(),output.data(),t.inverse_phase.data(),t.roots.data(),t.mods.data());});
    for(unsigned id=0;id<J;++id){
        unsigned dest=id/8*K*8+lastrow*8+id%8;
        auto& actual=output[size_t(P-1)*N+dest];
        U desired=uint64_t(expected[id])*J%p*rinv%p*rinv%p*t.inverse_phase[(size_t(prime)*K+lastrow)*J+id]%p;
        assert(actual==desired);
        actual=UINT32_MAX;
    }
    for(auto x:output)assert(x==UINT32_MAX);
    std::cout<<"PASS "<<J<<" dense cyclic outputs and "<<8*K<<" outer evaluations at N="<<N<<"\n";
}
int main(){check<32768,868,10>();check<65536,1747,18>();}
