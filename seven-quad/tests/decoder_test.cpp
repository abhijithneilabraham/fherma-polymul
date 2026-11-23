#define _XOPEN_SOURCE 600
#include <ucontext.h>
#include <functional>
// CUDA warp operations for testing the production decoder on one complete block.
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
#include "seven-plan.hpp"

struct Index { unsigned x,y; };
Index threadIdx{},blockIdx{},blockDim{};
namespace sim {
unsigned threads;
struct Lane { ucontext_t ctx{}; std::array<char,32768> stack; int waiting=-1; };
std::vector<Lane> lanes(512);
ucontext_t scheduler;
unsigned arrivals[17]{};
std::function<void()> kernel;
void invoke() { kernel(); lanes[threadIdx.x].waiting=-2; }
void barrier(unsigned id) {
    unsigned i=threadIdx.x;
    lanes[i].waiting=id;
    if(++arrivals[id]==(id==16?threads:32)) {
        for(unsigned j=(id==16?0:32*id),end=(id==16?threads:j+32);j<end;++j) {
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
struct Warp {uint64_t shuffle[32]; uint32_t a[32][4],b[32][2];};
Warp warps[16];
}
template<class T> T exchange(T x,unsigned from) {
    auto& w=sim::warps[threadIdx.x/32];
    w.shuffle[threadIdx.x%32]=x;sim::barrier(threadIdx.x/32);
    T result=w.shuffle[from];sim::barrier(threadIdx.x/32);
    return result;
}
uint32_t __shfl_xor_sync(unsigned mask,uint32_t x,unsigned distance){
    return exchange(x,(threadIdx.x%32)^distance);
}
uint64_t __shfl_down_sync(unsigned mask,uint64_t x,unsigned distance,unsigned width){
    unsigned lane=threadIdx.x%32;
    return exchange(x,lane%width+distance>=width?lane:lane+distance);
}
uint32_t __shfl_sync(unsigned mask,uint32_t x,unsigned from){return exchange(x,from);}
uint32_t __ballot_sync(unsigned mask,bool pred){
    auto& w=sim::warps[threadIdx.x/32];
    w.shuffle[threadIdx.x%32]=pred;sim::barrier(threadIdx.x/32);
    uint32_t bits=0;
    for(unsigned i=0;i<32;++i)bits|=uint32_t(w.shuffle[i])<<i;
    sim::barrier(threadIdx.x/32);return bits;
}
void emulate_mma(int32_t& c0,int32_t& c1,int32_t& c2,int32_t& c3,
                 uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1) {
    unsigned lane=threadIdx.x%32;auto& w=sim::warps[threadIdx.x/32];
    w.a[lane][0]=a0;w.a[lane][1]=a1;w.a[lane][2]=a2;w.a[lane][3]=a3;
    w.b[lane][0]=b0;w.b[lane][1]=b1;sim::barrier(threadIdx.x/32);
    int32_t* c[]={&c0,&c1,&c2,&c3};
    for(unsigned out=0;out<4;++out) {
        unsigned row=lane/4+8*(out/2), col=2*(lane%4)+(out%2);
        for(unsigned k=0;k<32;++k) {
            uint32_t a=w.a[4*(row%8)+(k%16)/4][row/8+2*(k/16)];
            uint32_t b=w.b[4*col+(k%16)/4][k/16];
            *c[out]+=((a>>(8*(k%4)))&255)*((b>>(8*(k%4)))&255);
        }
    }
    sim::barrier(threadIdx.x/32);
}
#define __device__
#define __host__
#define __shared__ static
#define __global__
#define __align__(n) __attribute__((aligned(n)))
#define __forceinline__ inline
#define __syncthreads() sim::barrier(16)
#include "../sparse-crt.hpp"
struct alignas(16) uint4 {uint32_t x,y,z,w;};
#include "decoder-emulated.hpp"

template<unsigned N,unsigned W,unsigned Count>
void check(){
    constexpr uint32_t L=(W+31)/32,K=32,J=N/K,first=J-4;
    constexpr uint32_t gap=W==1747?9699327:43646975;
    std::vector<uint32_t> q(L,UINT32_MAX);q.front()=0u-gap;q.back()=(1u<<(W%32))-1;
    auto plan=seven_plan::make(N,W,q);auto tables=bytecrt::warp_ntt::make_tables(plan.transform);
    std::vector<uint32_t> input(static_cast<size_t>(N)*7*Count,0),out(static_cast<size_t>(N)*L,UINT32_MAX);
    uint32_t expected[4*K][7*Count];
    for(auto& row:expected)for(auto& x:row)std::cin>>x;
    assert(std::cin);
    using bytecrt::rns_detail::pow_mod;
    for(unsigned column=0;column<4;++column) {
        unsigned id=first+column;
        for(unsigned pidx=0;pidx<Count;++pidx) {
            uint32_t p=plan.transform.primes[pidx],psi=plan.transform.psi_roots[pidx];
            uint32_t omega=(W==1747?seven_plan::roots1747[pidx]:seven_plan::roots868[pidx]).digit_root;
            for(unsigned freq=0;freq<7;++freq){
                uint32_t x[K]{};
                for(unsigned i=0;i<K;++i)for(unsigned d=0;d<7;++d){
                    uint64_t twist=uint64_t(plan.encode_fields[pidx].digit_twist[d].value)*pow_mod(omega,freq*d,p)%p;
                    x[i]=(x[i]+twist*expected[column*K+i][d*Count+pidx])%p;
                }
                for(unsigned row=0;row<K;++row){
                    uint32_t g=pow_mod(psi,J*(2*bytecrt::warp_ntt::reversed(row,K)+1),p),v=0;
                    for(unsigned i=K;i-->0;)v=(uint64_t(v)*g+x[i])%p;
                    unsigned address=(id/4)*K*4+row*4+id%4;
                    input[size_t(freq*Count+pidx)*N+address]=uint64_t(v)*pow_mod(K,p-2,p)%p;
                }
            }
        }
    }
    auto packed=sparse_crt::pack(plan.crt,L,Count);
    blockIdx.x=first/4;
    sim::launch(512,[&]{
        seven_decode::inverse_decode<N,W,Count>(input.data(),out.data(),packed.data(),plan.reciprocal48.data(),gap,tables.roots.data(),plan.inverse_fields.data());
    });
    for(unsigned index=0;index<4*K;++index){
        size_t base=size_t(first+index/K+J*(index%K))*L;
        for(unsigned j=0;j<L;++j){std::cout<<out[base+j]<<' ';out[base+j]=UINT32_MAX;}
        std::cout<<'\n';
    }
    for(auto x:out)assert(x==UINT32_MAX);
}
int main(){
    unsigned w;std::cin>>w;
    if(w==868)check<32768,868,10>();else check<65536,1747,18>();
}
