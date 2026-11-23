// Run the submitted block on 512 cooperative CPU fibers, including every barrier.
#define _XOPEN_SOURCE 600
#include <ucontext.h>
#include <array>
#include <cassert>
#include <functional>
#include <iostream>
#include <random>
#include <vector>
#include "../seven-plan.hpp"

struct Index { unsigned x,y; };
Index threadIdx{},blockIdx{};
namespace sim {
constexpr int threads=512;
struct Lane { ucontext_t ctx{}; std::array<char,32768> stack; int waiting=-1; };
std::vector<Lane> lanes(threads);
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
void launch(std::function<void()> body) {
    kernel=std::move(body);
    for(auto& l:lanes) {
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
Warp warps[16];
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
namespace bytecrt::detail {
uint32_t encode_mma_reduce(const int32_t* c,uint32_t p,uint64_t){
    uint64_t v=0;for(unsigned i=0;i<4;++i)v+=uint64_t(c[i])<<(8*i);return v%p;
}
}
#define __device__
#define __forceinline__ inline
#define __shared__ static
#define __global__
#define __syncthreads() sim::barrier(16)
#include "emulated-tile.hpp"
#include "../tile-layout.cuh"

template<unsigned N,unsigned W,unsigned Count>
void check() {
    using namespace bytecrt::rns_detail;
    constexpr unsigned K=N/256,L=(W+31)/32,id=255,pairs=N==65536?3:5;
    unsigned gap=W==1747?9699327:43646975;
    Words q(L,UINT32_MAX);q[0]=0u-gap;q.back()=(1u<<(W%32))-1;
    auto plan=seven_plan::make(N,W,q);auto tables=bytecrt::warp_ntt::make_tables(plan.transform);
    std::vector<uint32_t> input(static_cast<size_t>(N)*L),out(size_t(7)*Count*N,UINT32_MAX);
    std::mt19937 rng(N);
    for(unsigned i=0;i<K;++i){
        size_t start=size_t(id+256*i)*L;
        for(unsigned j=0;j<L;++j)input[start+j]=rng();
        input[start+L-1]&=q.back();
    }
    for(unsigned group=0;group<Count/(2*pairs);++group){
        blockIdx={id,group};
        sim::launch([&]{seven_encode::tile_forward<N,W,Count>(input.data(),out.data(),plan.matrix_words.data(),
             plan.encode_fields.data(),tables.forward_phase.data(),tables.roots.data());});
    }
    // Independent bit extraction, direct digit evaluations and direct odd-root DFT.
    for(unsigned j=0;j<Count;++j) {
        uint32_t p=plan.transform.primes[j],psi=plan.transform.psi_roots[j];
        uint32_t omega=(W==1747?seven_plan::roots1747[j]:seven_plan::roots868[j]).digit_root;
        uint32_t Rinv=pow_mod((uint64_t(1)<<32)%p,p-2,p);
        for(unsigned d=0;d<7;++d){
            uint32_t values[K]{};
            for(unsigned i=0;i<K;++i){
                const auto* raw=input.data()+size_t(id+256*i)*L;
                for(unsigned k=0;k<7;++k){
                    uint32_t v=0;
                    for(unsigned bit=((k+1)*W+6)/7;bit-->(k*W+6)/7;)
                        v=(uint64_t(v)*2+((raw[bit/32]>>(bit%32))&1))%p;
                    uint64_t t=uint64_t(plan.encode_fields[j].digit_twist[k].value)*pow_mod(omega,k*d,p)%p;
                    values[i]=(values[i]+t*v)%p;
                }
            }
            for(unsigned row=0;row<K;++row){
                uint32_t x=pow_mod(psi,256*(2*bytecrt::warp_ntt::reversed(row,K)+1),p),v=0;
                for(unsigned i=K;i-->0;)v=(uint64_t(v)*x+values[i])%p;
                v=uint64_t(v)*tables.forward_phase[size_t(j)*N+id*K+row]%p*Rinv%p;
                size_t plane=(size_t(d)*Count+j)*N;
                uint32_t tile[8][260];
                bytecrt::warp_ntt::gather_middle_word<K>(out.data()+plane,tile,row/8*8,8*id+row%8);
                assert(tile[row%8][id]==v);
                out[plane+id*K+row]=UINT32_MAX;
            }
        }
    }
    for(auto x:out)assert(x==UINT32_MAX);
    std::cout<<"PASS "<<7*Count*K<<" fused spectrum words and middle reads at N="<<N<<"\n";
}
int main(){check<32768,868,10>();check<65536,1747,18>();}
