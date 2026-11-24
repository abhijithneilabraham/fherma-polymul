// CUDA warp operations for testing the production decoder on one complete block.
#include <array>
#include <barrier>
#include <bit>
#include <cmath>
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
uint32_t __shfl_xor_sync(unsigned mask,uint32_t value,unsigned distance) {
    assert(mask==UINT32_MAX);
    return exchange(value,(threadIdx.x%32)^distance);
}
float __shfl_down_sync(unsigned mask,float value,unsigned distance) {
    assert(mask==UINT32_MAX);
    unsigned lane=threadIdx.x%32;
    return std::bit_cast<float>(exchange(std::bit_cast<uint32_t>(value),lane+distance<32?lane+distance:lane));
}
float __fadd_rn(float a,float b) {return a+b;}
float __fmul_rn(float a,float b) {return a*b;}
float __uint2float_rn(uint32_t a) {return float(a);}
uint32_t __float2uint_rn(float a) {return uint32_t(std::nearbyint(a));}
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
    constexpr unsigned L=(W+31)/32,K=32,J=N/K,gap=W==1747?9699327:43646975;
    std::vector<uint32_t> q(L,UINT32_MAX);q.front()=0u-gap;q.back()=(1u<<(W%32))-1;
    auto plan=two_digit::make(N,W,q);
    constexpr unsigned P=2*Count,id=J-1,mask=N==65536?1:0;
    constexpr size_t part_base=size_t(mask)*N*P,out_base=size_t(mask)*N*L;
    std::array<std::array<uint32_t,P>,K> values{};
    for(auto& row:values)for(auto& value:row)std::cin>>value;
    assert(std::cin);

    // The reference supplies the 32 inverse outputs for each plane. Build
    // the corresponding spectrum by direct evaluation, with the 1/32 factor.
    using bytecrt::rns_detail::pow_mod;
    auto tables=bytecrt::warp_ntt::make_tables(plan.transform);
    std::vector<uint32_t> input(part_base+static_cast<size_t>(P)*N,UINT32_MAX);
    std::vector<uint32_t> out(out_base+static_cast<size_t>(N)*L,UINT32_MAX);
    for(unsigned plane=0;plane<P;++plane) {
        uint32_t p=plan.primes[plane%Count],psi=plan.transform.psi_roots[plane%Count];
        uint32_t scale=pow_mod(32,p-2,p);
        for(unsigned index=0;index<K;++index) {
            unsigned odd=2*bytecrt::warp_ntt::reversed(index,K)+1;
            uint32_t root=pow_mod(psi,J*odd,p),value=0;
            for(unsigned row=K;row-->0;)
                value=(uint64_t(value)*root+values[row][plane])%p;
            input[part_base+size_t(plane)*N+(id/4)*K*4+4*index+id%4]=uint64_t(value)*scale%p;
        }
    }
    blockIdx.x=(id<<mask)+mask;
    std::vector<std::thread> lanes;
    for(unsigned i=0;i<224;++i)lanes.emplace_back([&,i]{
        threadIdx.x=i;
        bytecrt::g1_crt32_warp_alpha<W,L,2*Count,Kpad>(
            input.data(),plan.decode.data(),plan.inverse.data(),
            plan.primes.data(),plan.inverse_theta.data(),tables.roots.data(),tables.mods.data(),gap,out.data(),0,mask);
    });
    for(auto& l:lanes)l.join();
    for(unsigned row=0;row<32;++row){
        size_t pos=out_base+size_t(id+row*J)*L;
        for(unsigned j=0;j<L;++j){std::cout<<out[pos+j]<<' ';out[pos+j]=UINT32_MAX;}
        std::cout<<'\n';
    }
    for(auto x:out)assert(x==UINT32_MAX);
}
int main(){unsigned w;std::cin>>w;if(w==868)check<32768,868,30,256>();else check<65536,1747,58,480>();}
