// Run the submitted block on cooperative CPU fibers, including every barrier.
#define _XOPEN_SOURCE 600
#include <ucontext.h>
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
void emulate_mma(int32_t* acc,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1) {
    unsigned i=threadIdx.x/32,lane=threadIdx.x%32;auto& w=sim::warps[i];
    w.a[lane][0]=a0;w.a[lane][1]=a1;w.a[lane][2]=a2;w.a[lane][3]=a3;
    w.b[lane][0]=b0;w.b[lane][1]=b1;sim::barrier(i);
    for(unsigned out=0;out<4;++out) {
        unsigned row=lane/4+8*(out/2),col=2*(lane%4)+(out%2);
        for(unsigned k=0;k<32;++k) {
            uint32_t a=w.a[4*(row%8)+(k%16)/4][row/8+2*(k/16)];
            uint32_t b=w.b[4*col+(k%16)/4][k/16];
            acc[out]+=((a>>(8*(k%4)))&255)*((b>>(8*(k%4)))&255);
        }
    }
    sim::barrier(i);
}
uint32_t __shfl_sync(unsigned mask,uint32_t value,unsigned from,unsigned width) {
    assert(mask==UINT32_MAX && from<width);
    unsigned warp=threadIdx.x/32,lane=threadIdx.x%32;
    auto& shared=sim::warps[warp].shuffle;
    shared[lane]=value;sim::barrier(warp);
    uint32_t peer=shared[(lane/width)*width+from];sim::barrier(warp);
    return peer;
}
#define __device__
#define __forceinline__ inline
#define __global__
#define __launch_bounds__(...)
#define __syncthreads() sim::barrier(32)
#include "emulated-tile.hpp"
#include "../tile-layout.cuh"

template<unsigned N,unsigned W,unsigned L,unsigned Count,unsigned Pad,unsigned Split>
void check(){
    using namespace bytecrt::warp_ntt;
    using bytecrt::rns_detail::pow_mod;
    constexpr unsigned K=N/256,id=255,shift=Split*8;
    // Count distinct addresses in each bank for every real warp store.
    for(unsigned first=0;first<Count;first+=12) {
        for(unsigned warp=0;warp<8;++warp) for(unsigned step=0;step<12;step+=4) {
            unsigned pair=step/2+warp%2;
            if(first+2*pair>=Count)continue;
            for(unsigned half=0;half<2;++half) {
                std::array<unsigned,32> banks{};
                for(unsigned lane=0;lane<32;++lane) {
                    unsigned plane=2*pair+(lane%4)/2+12*(lane&1);
                    unsigned row=16*(warp/2)+lane/4+8*half;
                    unsigned offset=two_digit::local_row(plane,row);
                    assert(offset<K && two_digit::local_row(plane,offset)==row);
                    ++banks[(plane*K+offset)%32];
                }
                for(unsigned hits:banks)assert(hits==1);
            }
        }
    }
    const unsigned gap=W==1747 ? 9699327 : 43646975;
    std::vector<uint32_t> q(L,UINT32_MAX);q.front()=0u-gap;q.back()=(1u<<(W%32))-1;
    auto plan=two_digit::make(N,W,q);auto tables=make_tables(plan.transform);
    std::vector<uint32_t> left(static_cast<size_t>(N)*L),right(left.size());
    std::vector<uint32_t> out(size_t(4)*Count*N,UINT32_MAX);
    std::vector<uint32_t> window(two_window::spectrum_words<N>,UINT32_MAX);
    std::mt19937 rng(1729);
    for(unsigned operand=0;operand<2;++operand){
        auto& input=operand ? right : left;
        for(unsigned row=0;row<K;++row){
            auto* raw=input.data()+size_t(id+256*row)*L;
            for(unsigned limb=0;limb<L;++limb)raw[limb]=row==0 ? UINT32_MAX : rng();
            raw[L-1]&=q.back();
        }
        for(unsigned group=0;group<two_window::group_count<Count>;++group) {
            constexpr unsigned capacity=two_window::capacity;
            unsigned first=group*capacity,count=two_window::live_primes(Count,first);
            blockIdx={id,operand};
            sim::launch(256,[&]{two_digit::tile_forward<N,W,L,Count,Pad,Split>(
                left.data(),right.data(),window.data(),plan.encode.data(),plan.primes.data(),
                plan.reciprocal.data(),tables.forward_phase.data(),tables.roots.data(),tables.mods.data(),first);});
            for(unsigned sign=0;sign<2;++sign)for(unsigned local=0;local<count;++local) {
                unsigned src=operand*2*capacity+sign*capacity+local;
                unsigned dst=operand*2*Count+sign*Count+first+local;
                for(unsigned row=0;row<K;++row) {
                    auto& value=window[size_t(src)*N+id*K+row];
                    assert(value!=UINT32_MAX);
                    out[size_t(dst)*N+id*K+row]=value;
                    value=UINT32_MAX;
                }
            }
            for(auto value:window)assert(value==UINT32_MAX);
        }
        for(unsigned prime=0;prime<Count;++prime){
            uint32_t p=plan.primes[prime],psi=plan.transform.psi_roots[prime];
            uint32_t theta=(W==1747?two_digit::large[prime]:two_digit::small[prime]).theta;
            std::array<std::array<uint32_t,K>,2> input_planes{};
            for(unsigned row=0;row<K;++row){
                auto* raw=input.data()+size_t(id+256*row)*L;
                uint32_t low=0,high=0;
                for(unsigned i=W;i-->0;){
                    auto& digit=i>=shift ? high : low;
                    digit=(2ull*digit+((raw[i/32]>>(i%32))&1))%p;
                }
                uint32_t twist=uint64_t(theta)*high%p;
                input_planes[0][row]=(uint64_t(low)+twist)%p;
                input_planes[1][row]=(uint64_t(low)+p-twist)%p;
            }
            for(unsigned freq=0;freq<K;++freq){
                unsigned exponent=2*reversed(freq,K)+1;
                uint32_t x=pow_mod(psi,256*exponent,p),phase=pow_mod(psi,id*exponent,p);
                for(unsigned digit=0;digit<2;++digit){
                    uint32_t y=0;
                    for(unsigned j=K;j-->0;)y=(uint64_t(y)*x+input_planes[digit][j])%p;
                    auto& value=out[(size_t(2*operand*Count+digit*Count+prime)*N)+id*K+freq];
                    assert(value==uint64_t(y)*phase%p);
                    value=UINT32_MAX;
                }
            }
        }
        for(auto v:out)assert(v==UINT32_MAX);
    }
    // Check every gather address linking the fused output to the middle FFT.
    std::vector<uint32_t> encoded(N);
    for(unsigned i=0;i<N;++i)encoded[i]=i;
    uint32_t tile[32][257];
    for(unsigned first=0;first<K;first+=32){
        for(unsigned i=0;i<1024;++i)gather_spectrum<K>(encoded.data(),tile,first,i/32,i%32);
        for(unsigned row=0;row<32;++row)for(unsigned col=0;col<256;++col)
            assert(tile[row][col]==col*K+first+row);
    }
    std::cout<<"PASS "<<4*Count*K<<" fused output words and "<<N<<" middle gather addresses at N="<<N<<"\n";
}
int main(){check<32768,868,28,30,64,55>();check<65536,1747,55,58,128,110>();}
