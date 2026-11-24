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
std::vector<Lane> lanes(768);
ucontext_t scheduler;
unsigned arrivals[25]{};
std::function<void()> kernel;
void invoke() { kernel(); lanes[threadIdx.x].waiting=-2; }
void barrier(unsigned id) {
    unsigned i=threadIdx.x;
    lanes[i].waiting=id;
    if(++arrivals[id]==(id==24?threads:32)) {
        for(unsigned j=(id==24?0:32*id),end=(id==24?threads:j+32);j<end;++j) {
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
Warp warps[24];
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
uint32_t __shfl_sync(unsigned mask,uint32_t value,unsigned source,unsigned width) {
    assert(mask==UINT32_MAX && (width==16 || width==32));
    unsigned warp=threadIdx.x/32,lane=threadIdx.x%32;
    auto& values=sim::warps[warp].shuffle;
    values[lane]=value;sim::barrier(warp);
    uint32_t peer=values[(lane/width)*width+source];sim::barrier(warp);
    return peer;
}
#define __device__
#define __forceinline__ inline
#define __shared__ static
#define __launch_bounds__(...)
#define __global__
#define __syncwarp() sim::barrier(threadIdx.x/32)
#define __syncthreads() sim::barrier(24)
#include "emulated-tile.hpp"
unsigned __brev(unsigned x) { unsigned y=0; for(unsigned i=0;i<32;++i,x>>=1)y=2*y+(x&1);return y; }
#include "emulated-stage.hpp"

template<unsigned N,unsigned W,unsigned Count,unsigned Pad,unsigned Split>
void check() {
    using namespace bytecrt::rns_detail;
    using namespace bytecrt::warp_ntt;
    using U=uint32_t;
    constexpr unsigned K=32,J=N/K,L=(W+31)/32,P=2*Count,base=J-4;
    unsigned gap=W==1747?9699327:43646975;
    Words q(L,UINT32_MAX);q[0]=0u-gap;q.back()=(1u<<(W%32))-1;
    auto plan=two_digit::make(N,W,q);auto tables=make_tables(plan.transform);
    std::vector<U> a(size_t{N}*L),b(a.size()),spectrum(size_t{2}*P*N,UINT32_MAX);
    std::mt19937 rng(N);
    for(unsigned operand=0;operand<2;++operand) {
        auto& input=operand?b:a;
        for(unsigned col=0;col<4;++col)for(unsigned i=0;i<K;++i) {
            U* raw=input.data()+size_t(base+col+J*i)*L;
            for(unsigned limb=0;limb<L;++limb)raw[limb]=rng();
            raw[L-1]&=q.back();
        }
        blockIdx={base/4,operand};
        sim::launch(32*two_digit::input_warps,[&]{two_digit::tile_forward<N,W,L,Count,Pad,Split>(
            a.data(),b.data(),spectrum.data(),plan.encode.data(),plan.primes.data(),plan.reciprocal.data(),
            tables.forward_phase.data(),tables.roots.data(),tables.mods.data());});
        for(unsigned prime=0;prime<Count;++prime) {
            U p=plan.transform.primes[prime],psi=plan.transform.psi_roots[prime];
            U theta=(W==1747?two_digit::large[prime]:two_digit::small[prime]).theta;
            for(unsigned digit=0;digit<2;++digit)for(unsigned col=0;col<4;++col) {
                unsigned id=base+col;
                U values[K]{};
                for(unsigned i=0;i<K;++i){
                    const U* raw=input.data()+size_t(id+J*i)*L;
                    U low=0,high=0;
                    for(unsigned bit=W;bit-->0;) {
                        U& value=bit>=8*Split?high:low;
                        value=(uint64_t(value)*2+((raw[bit/32]>>(bit%32))&1))%p;
                    }
                    U rotated=uint64_t(theta)*high%p;
                    values[i]=digit?(low+p-rotated)%p:(low+rotated)%p;
                }
                for(unsigned row=0;row<K;++row){
                    unsigned freq=2*reversed(row,K)+1;
                    U x=pow_mod(psi,J*freq,p),v=0;
                    for(unsigned i=K;i-->0;)v=(uint64_t(v)*x+values[i])%p;
                    U expected=uint64_t(v)*pow_mod(psi,id*freq,p)%p*((uint64_t(1)<<32)%p)%p;
                    size_t plane=operand*P+digit*Count+prime;
                    auto& word=spectrum[plane*N+(row/2)*(2*J)+2*base+(row%2)*4+col];
                    assert(word==expected);
                    word=UINT32_MAX;
                }
            }
        }
        for(auto word:spectrum)assert(word==UINT32_MAX);
    }

    for(unsigned which=0;which<3;++which) {
        const unsigned prime=which==0?0:which==1?Count/2:Count-1;
        const unsigned plane=(which==0?0:1)*Count+prime;
        const unsigned first=which==0?0:which==1?K/2:K-2;
        U p=plan.transform.primes[prime],rinv=pow_mod((uint64_t(1)<<32)%p,p-2,p);
        std::vector<U> output(size_t{P}*N,UINT32_MAX);
        std::array<std::vector<U>,2> conv{std::vector<U>(J),std::vector<U>(J)};
        for(unsigned side=0;side<2;++side) {
            std::vector<U> x(J),y(J);
            for(unsigned id=0;id<J;++id) {
                size_t offset=size_t(first)*J+id/4*8+4*side+id%4;
                spectrum[size_t(plane)*N+offset]=x[id]=rng()%p;
                spectrum[size_t(plane+P)*N+offset]=y[id]=rng()%p;
            }
            for(unsigned i=0;i<J;++i)for(unsigned j=0;j<J;++j) {
                auto& v=conv[side][(i+j)%J];v=(v+uint64_t(x[i])*y[j])%p;
            }
        }
        std::vector<U> twiddles(2*Count*J);
        for(unsigned direction=0;direction<2;++direction) {
            blockIdx={direction,prime};
            sim::launch(1,[&]{make_sdk_table<J>(twiddles.data(),tables.sdk.data());});
            sim::launch(256,[&]{convert_sdk_table<J>(twiddles.data(),tables.sdk.data());});
        }
        for(unsigned side=0;side<2;++side) {
            blockIdx={first+side,plane};
            sim::launch(256,[&]{middle<N,P>(spectrum.data(),output.data(),tables.inverse_phase.data(),
                twiddles.data(),tables.sdk.data(),tables.mods.data());});
        }
        for(unsigned side=0;side<2;++side)for(unsigned id=0;id<J;++id) {
            unsigned row=first+side;
            auto& actual=output[size_t(plane)*N+id/4*K*4+row*4+id%4];
            U expected=uint64_t(conv[side][id])*rinv%p*rinv%p*tables.inverse_phase[(size_t(prime)*K+row)*J+id]%p;
            assert(actual==expected);actual=UINT32_MAX;
        }
        for(auto word:output)assert(word==UINT32_MAX);
    }
    std::cout<<"PASS "<<2*P*4*K<<" forward and "<<6*J<<" middle outputs at N="<<N<<"\n";
}
int main(){check<32768,868,30,64,55>();check<65536,1747,58,128,110>();}
