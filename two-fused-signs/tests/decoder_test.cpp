// CUDA warp operations for testing the production decoder on one complete block.
#include <array>
#include <bit>
#include <cmath>
#include <barrier>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>
#include "two-plan.hpp"

struct Index { unsigned x=0,y=0; };
thread_local Index threadIdx;
Index blockIdx;
struct Warp {
    std::barrier<> sync{32};
    uint64_t v[32];
    uint32_t a[32][4],b[32][2];
};
std::array<Warp,8> warps;
std::barrier<> block_sync{224};
template<class T> T exchange(T x,unsigned from) {
    auto& w=warps[threadIdx.x/32];
    w.v[threadIdx.x%32]=x;w.sync.arrive_and_wait();
    T result=w.v[from];w.sync.arrive_and_wait();
    return result;
}
uint32_t __shfl_xor_sync(unsigned mask,uint32_t x,unsigned distance){
    return exchange(x,(threadIdx.x%32)^distance);
}
float __shfl_down_sync(unsigned mask,float value,unsigned delta) {
    unsigned l=threadIdx.x%32;
    return std::bit_cast<float>(exchange(std::bit_cast<uint32_t>(value),l+delta<32?l+delta:l));
}
float __fadd_rn(float a,float b){volatile float v=a+b;return v;}
float __fmul_rn(float a,float b){volatile float v=a*b;return v;}
float __uint2float_rn(uint32_t x){return static_cast<float>(x);}
uint32_t __float2uint_rn(float x){return static_cast<uint32_t>(std::nearbyint(x));}
void emulate_mma(int32_t& c0,int32_t& c1,int32_t& c2,int32_t& c3,
                 uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1) {
    unsigned lane=threadIdx.x%32;auto& w=warps[threadIdx.x/32];
    w.a[lane][0]=a0;w.a[lane][1]=a1;w.a[lane][2]=a2;w.a[lane][3]=a3;
    w.b[lane][0]=b0;w.b[lane][1]=b1;w.sync.arrive_and_wait();
    int32_t* c[]={&c0,&c1,&c2,&c3};
    for(unsigned out=0;out<4;++out) {
        unsigned row=lane/4+8*(out/2), col=2*(lane%4)+(out%2);
        for(unsigned k=0;k<32;++k) {
            uint32_t a=w.a[4*(row%8)+(k%16)/4][row/8+2*(k/16)];
            uint32_t b=w.b[4*col+(k%16)/4][k/16];
            *c[out]+=((a>>(8*(k%4)))&255)*((b>>(8*(k%4)))&255);
        }
    }
    w.sync.arrive_and_wait();
}
#define __device__
#define __host__
#define __shared__ static
#define __global__
#define __align__(n) __attribute__((aligned(n)))
#define __forceinline__ inline
#define __syncthreads() block_sync.arrive_and_wait()
#define __launch_bounds__(...)
#include "decoder-emulated.hpp"

template<unsigned N,unsigned W,unsigned Count,unsigned Kpad>
void check(){
    constexpr unsigned L=(W+31)/32,K=N/256,gap=W==1747?9699327:43646975;
    std::vector<uint32_t> q(L,UINT32_MAX);q.front()=0u-gap;q.back()=(1u<<(W%32))-1;
    auto plan=two_digit::make(N,W,q);
    auto tables=bytecrt::warp_ntt::make_tables(plan.transform);
    std::vector<uint32_t> input(static_cast<size_t>(N)*2*Count,0xdeadbeef),out(static_cast<size_t>(N)*L,UINT32_MAX);
    constexpr unsigned group=K/32,low=group-1;
    uint32_t residues[64][2*Count];
    for(auto& row:residues)for(auto& v:row)std::cin>>v;
    assert(std::cin);
    using bytecrt::rns_detail::pow_mod;
    for(unsigned side=0;side<2;++side) for(unsigned prime=0;prime<2*Count;++prime) {
        unsigned id=254+side;
        uint32_t p=plan.transform.primes[prime],psi=plan.transform.psi_roots[prime];
        for(unsigned coarse=0;coarse<32;++coarse) {
            uint32_t x=pow_mod(psi,256*group*(2*bytecrt::warp_ntt::reversed(coarse,32)+1),p);
            uint32_t v=0;
            for(unsigned j=32;j-->0;)v=(uint64_t(v)*x+residues[32*side+j][prime])%p;
            size_t index=(low*128+id/2)*64+(coarse/4)*8+4*(id%2)+coarse%4;
            input[size_t(prime)*N+index]=uint64_t(v)*pow_mod(32,p-2,p)%p;
        }
    }
    blockIdx.x=N/64-1;
    std::vector<std::thread> lanes;
    for(unsigned i=0;i<224;++i)lanes.emplace_back([&,i]{
        threadIdx.x=i;
        bytecrt::inverse_crt<W,L,2*Count,Kpad>(input.data(),tables.roots.data(),tables.mods.data(),
            plan.decode.data(),plan.inverse.data(),plan.primes.data(),plan.inverse_theta.data(),
            q.data(),gap,out.data());
    });
    for(auto& l:lanes)l.join();
    for(unsigned side=0;side<2;++side)for(unsigned row=0;row<32;++row){
        size_t pos=size_t(254+side+(low+group*row)*256)*L;
        for(unsigned j=0;j<L;++j){std::cout<<out[pos+j]<<' ';out[pos+j]=UINT32_MAX;}
        std::cout<<'\n';
    }
    for(auto x:out)assert(x==UINT32_MAX);
}
int main(){unsigned w;std::cin>>w;if(w==868)check<32768,868,30,256>();else check<65536,1747,58,480>();}
