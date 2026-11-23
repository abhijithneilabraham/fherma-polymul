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
#include "../warp-stages.cuh"

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
        sim::launch(32*two_digit::input_warps<N>,[&]{two_digit::tile_forward<N,W,L,Count,Pad,Split>(
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
                    U expected=uint64_t(v)*pow_mod(psi,id*freq,p)%p;
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
        blockIdx={first/2,plane};
        sim::launch(512,[&]{middle<N,P>(spectrum.data(),output.data(),tables.inverse_phase.data(),tables.roots.data(),tables.mods.data());});
        for(unsigned side=0;side<2;++side)for(unsigned id=0;id<J;++id) {
            unsigned row=first+side;
            auto& actual=output[size_t(plane)*N+id/4*K*4+row*4+id%4];
            U expected=uint64_t(conv[side][id])*J%p*rinv%p*rinv%p*tables.inverse_phase[(size_t(prime)*K+row)*J+id]%p;
            assert(actual==expected);actual=UINT32_MAX;
        }
        for(auto word:output)assert(word==UINT32_MAX);
    }
    constexpr unsigned pad=W==1747?480:256;
    std::vector<U> inverse_input(static_cast<size_t>(P)*N,UINT32_MAX);
    std::vector<U> weighted(static_cast<size_t>(N)*pad/4,UINT32_MAX);
    for(unsigned column_group : {0u,J/4-1}) {
        unsigned group=(P-1)/8;
        for(unsigned plane=8*group;plane<P;++plane) {
            U p=plan.transform.primes[plane%Count];
            for(unsigned i=0;i<4*K;++i)
                inverse_input[size_t(plane)*N+column_group*K*4+i]=rng()%p;
        }
        blockIdx={column_group,group};
        sim::launch(256,[&]{inverse<N,P,pad>(inverse_input.data(),weighted.data(),tables.roots.data(),tables.mods.data());});
        for(unsigned plane=8*group;plane<P;++plane) {
            U p=plan.transform.primes[plane%Count],psi=plan.transform.psi_roots[plane%Count];
            for(unsigned col=0;col<4;++col)for(unsigned row=0;row<K;++row) {
                U expected=0;
                unsigned id=4*column_group+col;
                for(unsigned k=0;k<K;++k) {
                    unsigned exponent=(J*row*(2*reversed(k,K)+1))%(2*N);
                    U value=inverse_input[size_t(plane)*N+column_group*K*4+4*k+col];
                    expected=(expected+uint64_t(value)*pow_mod(psi,(2*N-exponent)%(2*N),p))%p;
                }
                auto& result=weighted[(size_t(id)*(pad/4)+plane)*K+row];
                assert(result==expected);result=UINT32_MAX;
            }
        }
        for(auto value:weighted)assert(value==UINT32_MAX);
    }
    // Validate sector counts and the coefficient order read by the CRT block.
    for(unsigned group=0;group<J/4;++group)for(unsigned part=0;part<4;++part) {
        std::array<unsigned,4> reads{},writes{};
        for(unsigned lane=0;lane<32;++lane) {
            unsigned col=lane%4,row=lane/4+8*part;
            unsigned read=(group*K*4+4*row+col)/8;
            unsigned write=((4*group+col)*(pad/4)*K+row)/8;
            if(!col)reads[lane/8]=read;
            if(lane/4==0)writes[col]=write;
            assert(read==reads[lane/8] && write==writes[col]);
        }
        for(unsigned i=1;i<4;++i)assert(reads[i]==reads[i-1]+1 && writes[i]>writes[i-1]);
    }
    std::cout<<"PASS "<<2*P*4*K<<" forward, "<<6*J<<" middle and "
             <<2*(P%8?P%8:8)*4*K<<" inverse outputs at N="<<N<<"\n";
}
int main(){check<32768,868,30,64,55>();check<65536,1747,58,128,110>();}
