// CUDA warp operations for testing the production decoder on one complete block.
#include <array>
#include <barrier>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>
#include "four-plan.hpp"

struct Index { unsigned x=0,y=0; };
thread_local Index threadIdx;
Index blockIdx;
struct Warp {
    std::barrier<> sync{32};
    uint64_t v[32];
    uint32_t a[32][4],b[32][2];
};
std::array<Warp,8> warps;
std::barrier<> block_sync{256};
template<class T> T exchange(T x,unsigned from) {
    auto& w=warps[threadIdx.x/32];
    w.v[threadIdx.x%32]=x;w.sync.arrive_and_wait();
    T result=w.v[from];w.sync.arrive_and_wait();
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
    auto& w=warps[threadIdx.x/32];
    w.v[threadIdx.x%32]=pred;w.sync.arrive_and_wait();
    uint32_t bits=0;
    for(unsigned i=0;i<32;++i)bits|=uint32_t(w.v[i])<<i;
    w.sync.arrive_and_wait();return bits;
}
uint32_t __reduce_add_sync(unsigned mask,uint32_t value) {
    assert(mask==UINT32_MAX);
    auto& w=warps[threadIdx.x/32];
    w.v[threadIdx.x%32]=value;
    w.sync.arrive_and_wait();
    uint32_t total=0;
    for(auto v:w.v) total+=uint32_t(v);
    w.sync.arrive_and_wait();
    return total;
}
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
#include "../sparse-crt.hpp"
#include "decoder-emulated.hpp"

template<unsigned N,unsigned W,unsigned Count>
void check(){
    constexpr uint32_t L=(W+31)/32,K=32,J=N/K,id=J-1;
    constexpr uint32_t gap=W==1747?9699327:43646975;
    std::vector<uint32_t> q(L,UINT32_MAX);q.front()=0u-gap;q.back()=(1u<<(W%32))-1;
    auto plan=four_plan::make(N,W,q);auto tables=bytecrt::warp_ntt::make_tables(plan.transform);
    std::vector<uint32_t> input(static_cast<size_t>(N)*4*Count,0),out(static_cast<size_t>(N)*L,UINT32_MAX);
    uint32_t expected[K][4*Count];
    for(auto& row:expected)for(auto& x:row)std::cin>>x;
    assert(std::cin);
    using bytecrt::rns_detail::pow_mod;
    for(unsigned pidx=0;pidx<Count;++pidx) {
        uint32_t p=plan.transform.primes[pidx],psi=plan.transform.psi_roots[pidx];
        uint32_t omega=(W==1747?four_plan::roots1747[pidx]:four_plan::roots868[pidx]).digit_root;
        for(unsigned freq=0;freq<4;++freq){
            uint32_t x[K]{};
            for(unsigned i=0;i<K;++i)for(unsigned d=0;d<4;++d){
                uint64_t twist=uint64_t(pow_mod(2,(d*W+3)/4,p))*pow_mod((W==1747?four_plan::roots1747[pidx]:four_plan::roots868[pidx]).digit_scale,d,p)%p*pow_mod(omega,freq*d,p)%p;
                x[i]=(x[i]+twist*expected[i][d*Count+pidx])%p;
            }
            for(unsigned row=0;row<K;++row){
                uint32_t g=pow_mod(psi,J*(2*bytecrt::warp_ntt::reversed(row,K)+1),p),v=0;
                for(unsigned i=K;i-->0;)v=(uint64_t(v)*g+x[i])%p;
                unsigned address=(id/4)*K*4+row*4+id%4;
                input[size_t(freq*Count+pidx)*N+address]=uint64_t(v)*pow_mod(K,p-2,p)%p;
            }
        }
    }
    auto packed=sparse_crt::pack(plan.crt,L,Count);
    blockIdx.x=id;
    std::vector<std::thread> threads;
    for(unsigned i=0;i<256;++i)threads.emplace_back([&,i]{
        threadIdx.x=i;four_decode::inverse_decode<N,W,Count>(input.data(),out.data(),packed.data(),plan.alpha_reciprocal.data(),gap,tables.roots.data(),plan.inverse_fields.data());
    });
    for(auto& t:threads)t.join();
    for(unsigned row=0;row<32;++row){
        size_t base=size_t(id+J*row)*L;
        for(unsigned j=0;j<L;++j){std::cout<<out[base+j]<<' ';out[base+j]=UINT32_MAX;}
        std::cout<<'\n';
    }
    for(auto x:out)assert(x==UINT32_MAX);
}
int main(){
    unsigned w;std::cin>>w;
    if(w==868)check<32768,868,16>();else check<65536,1747,30>();
}
