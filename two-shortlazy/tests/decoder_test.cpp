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
float __shfl_down_sync(unsigned mask,float value,unsigned distance) {
    assert(mask==UINT32_MAX);
    unsigned lane=threadIdx.x%32;
    return std::bit_cast<float>(exchange(std::bit_cast<uint32_t>(value),lane+distance<32?lane+distance:lane));
}
uint32_t __shfl_sync(unsigned mask,uint32_t value,unsigned from) {
    assert(mask==UINT32_MAX && from<32);
    return exchange(value,from);
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
    std::vector<uint32_t> input(static_cast<size_t>(N)*Kpad/4,0xdeadbeefu),out(static_cast<size_t>(N)*L,UINT32_MAX);
    constexpr unsigned id=J-1;
    for(unsigned row=0;row<32;++row)for(unsigned plane=0;plane<2*Count;++plane)
        std::cin>>input[(size_t(id)*Kpad/4+plane)*K+row];
    assert(std::cin);
    blockIdx.x=N/32-1;
    std::vector<std::thread> lanes;
    for(unsigned i=0;i<224;++i)lanes.emplace_back([&,i]{
        threadIdx.x=i;
        bytecrt::g1_crt32_warp_alpha<W,L,2*Count,Kpad,true>(
            reinterpret_cast<const uint8_t*>(input.data()),plan.decode.data(),plan.inverse.data(),
            plan.primes.data(),plan.inverse_theta.data(),plan.crt_bias.data(),gap,out.data());
    });
    for(auto& l:lanes)l.join();
    for(unsigned row=0;row<32;++row){
        size_t pos=size_t(id+row*J)*L;
        for(unsigned j=0;j<L;++j){std::cout<<out[pos+j]<<' ';out[pos+j]=UINT32_MAX;}
        std::cout<<'\n';
    }
    for(auto x:out)assert(x==UINT32_MAX);
}
int main(){unsigned w;std::cin>>w;if(w==868)check<32768,868,32,256>();else check<65536,1747,60,480>();}
