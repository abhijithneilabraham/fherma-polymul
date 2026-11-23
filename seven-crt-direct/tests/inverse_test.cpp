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
#include "../seven-inverse.cuh"

using namespace seven_inverse;
U power(U x,U exponent,U p){
    U y=1;
    for(;exponent;exponent>>=1,x=uint64_t(x)*x%p) if(exponent&1)y=uint64_t(x)*y%p;
    return y;
}
unsigned reverse(unsigned x,unsigned n){
    unsigned y=0;
    for(n/=2;n;n/=2,x/=2)y=2*y+(x%2);
    return y;
}
template<unsigned N,unsigned Count>
void check(U p,U omega,U generator) {
    constexpr unsigned K=N/256, prime=Count-1, id=255;
    auto fixed=[&](U x){return Twiddle{x,U((uint64_t(x)<<32)/p)};};
    U psi=power(generator,(p-1)/(2*K),p);
    assert(power(psi,K,p)==p-1);
    std::vector<Field> fields(Count);
    auto& f=fields[prime]; f.mod=make_mod(p,1);
    f.inverse=prepare(power(omega,p-2,p),f.mod);
    std::mt19937 rng(N);
    for(auto& fct:f.recovery)fct=fixed(1+rng()%(p-1));
    std::vector<Twiddle> roots(Count*(2*K+256));
    auto* inverse_roots=roots.data()+prime*(2*K+256)+K;
    for(unsigned groups=1;groups<K;groups*=2) {
        unsigned distance=K/(2*groups);
        for(unsigned j=0;j<groups;++j) {
            unsigned e=distance*(2*reverse(j,groups)+1);
            inverse_roots[groups+j]=fixed(power(psi,2*K-e,p));
        }
    }
    U digits[7][K],eval[7][K];
    for(auto& row:digits)for(auto& x:row)x=rng()%p;
    for(unsigned s=0;s<7;++s)for(unsigned i=0;i<K;++i){
        uint64_t acc=0;
        for(unsigned d=0;d<7;++d)acc+=uint64_t(digits[d][i])*power(omega,s*d,p)%p;
        eval[s][i]=acc%p;
    }
    std::vector<U> input(7*Count*N),output(input.size(),UINT32_MAX);
    for(unsigned j=0;j<K;++j){
        U x=power(psi,2*reverse(j,K)+1,p);
        for(unsigned s=0;s<7;++s){
            U v=0;
            for(unsigned i=K;i-->0;) v=(uint64_t(v)*x+eval[s][i])%p;
            input[(size_t(s)*Count+prime)*N+id*K+j]=v;
        }
    }
    blockIdx={id,prime};
    std::vector<std::thread> threads;
    for(unsigned i=0;i<256;++i)threads.emplace_back([&,i]{
        threadIdx.x=i;inverse<N,Count>(input.data(),output.data(),roots.data(),fields.data());
    });
    for(auto& t:threads)t.join();
    for(unsigned d=0;d<7;++d)for(unsigned i=0;i<K;++i){
        auto& actual=output[(size_t(id)*7*Count+d*Count+prime)*K+i];
        assert(actual == uint64_t(digits[d][i])*K*7%p*f.recovery[d].value%p);
        actual=UINT32_MAX;
    }
    for(auto x:output)assert(x==UINT32_MAX);
    std::cout<<"PASS "<<7*K<<" fused inverse outputs at N="<<N<<"\n";
}
int main(){ check<32768,10>(1883176961,1674877682,3); check<65536,18>(919339009,82969376,11); }
