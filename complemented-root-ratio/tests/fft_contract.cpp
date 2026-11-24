// Execute the actual warp butterfly code with a synchronous CPU shuffle model.
#include <array>
#include <cassert>
#include <iostream>
#include <random>
#include "../two-plan.hpp"

static std::array<uint32_t,256> sent;
static unsigned lane, slot, active_distance;
static uint32_t __shfl_xor_sync(unsigned mask, uint32_t value, unsigned distance) {
    assert(mask == 0xffffffff && distance == active_distance);
    assert(sent[32*slot+lane] == value);
    return sent[32*slot++ + (lane^distance)];
}
#define __device__
#define __forceinline__ inline
#include "../warp-fft.cuh"
#include "../odd-fft.cuh"
using namespace bytecrt::warp_ntt;
using bytecrt::rns_detail::pow_mod;
using U = uint32_t;
static U ordinary_root(const bytecrt::warp_ntt::Twiddle& root,U p) {
    uint64_t fraction=(uint64_t(root.high)<<32)-root.low;
    return U((__uint128_t(fraction)*p)>>64);
}

static U mul(U a,U b,U p) { return uint64_t(a)*b%p; }

template<unsigned Len, bool Odd, bool Inverse, unsigned D=Inverse?1:Len/2>
void transform(U* v, const Twiddle* roots, Mod m) {
    if constexpr (D<32) for (unsigned i=0;i<Len;++i) {
        sent[i]=v[i];
        if (i&D) {
            if constexpr (Odd && !Inverse)
                sent[i]=mul(v[i],ordinary_root(roots[Len/(2*D)+i/(2*D)],m.p),m.p);
            if constexpr (!Odd && Inverse)
                sent[i]=mul(v[i],ordinary_root(roots[(Len-(i&(D-1))*(Len/(2*D)))&(Len-1)],m.p),m.p);
        }
    }
    for (lane=0;lane<32;++lane) {
        U regs[Len/32];
        for (unsigned r=0;r<Len/32;++r) regs[r]=v[lane+32*r];
        slot=0;
        active_distance=D;
        if constexpr (Odd) odd_step<Len,D,Inverse>(regs,roots,m,lane);
        else step<Len,D,Inverse>(regs,roots,m,lane);
        assert(slot==(D<32?Len/32:0));
        for (unsigned r=0;r<Len/32;++r) v[lane+32*r]=regs[r];
    }
    if constexpr (Inverse && D<Len/2) transform<Len,Odd,Inverse,2*D>(v,roots,m);
    if constexpr (!Inverse && D>1) transform<Len,Odd,Inverse,D/2>(v,roots,m);
}

template<unsigned K>
void check(const bytecrt::RnsPlan& plan,const Tables& tables,unsigned prime,std::mt19937& rng) {
    constexpr unsigned N=256*K;
    Mod m=tables.mods[prime];
    const Twiddle* roots=tables.roots.data()+size_t(prime)*(2*K+256);
    std::array<U,K> v,input;
    for(auto& x:input) x=rng()%m.p;
    v=input;
    transform<K,true,false>(v.data(),roots,m);
    // Direct evaluations at every odd root independently fix the frequency order.
    for(unsigned r=0;r<K;++r) {
        U point=pow_mod(plan.psi_roots[prime],256*(2*reversed(r,K)+1),m.p);
        U acc=0;
        for(unsigned i=K;i-->0;) acc=(uint64_t(acc)*point+input[i])%m.p;
        assert(acc==v[r]);
    }
    transform<K,true,true>(v.data(),roots+K,m);
    for(unsigned i=0;i<K;++i) assert(v[i]==mul(input[i],K,m.p));

    // Validate every cross-stage constant at every prime against a second
    // recurrence, then run a complete dense by sparse product at three primes.
    U R=(uint64_t(1)<<32)%m.p;
    for(unsigned row=0;row<K;++row) {
        U ratio=pow_mod(plan.psi_roots[prime],2*reversed(row,K)+1,m.p);
        U power=1;
        for(unsigned id=0;id<256;++id) {
            U forward=tables.forward_phase[size_t(prime)*N+id*K+row];
            U inverse=tables.inverse_phase[size_t(prime)*N+row*256+id];
            assert(forward==mul(R,power,m.p));
            assert(mul(mul(inverse,power,m.p),N,m.p)==mul(m.r2,plan.crt_inverse[prime],m.p));
            power=mul(power,ratio,m.p);
        }
    }
    if(prime!=0 && prime!=plan.pcount/2 && prime+1!=plan.pcount) return;
    std::vector<U> a(N),b(N),fa(N),fb(N),stage(N),expected(N);
    for(auto& x:a) x=rng()%m.p;
    for(unsigned j:{0u,1u,255u,256u,N-1}) {
        b[j]=1+rng()%(m.p-1);
        for(unsigned i=0;i<N;++i) {
            U term=mul(a[i],b[j],m.p);
            unsigned n=i+j;
            expected[n%N]=n<N?add(expected[n%N],term,m.p):sub(expected[n%N],term,m.p);
        }
    }
    auto forward=[&](const auto& in, auto& out) {
        for(unsigned id=0;id<256;++id) {
            for(unsigned i=0;i<K;++i) v[i]=in[id+256*i];
            transform<K,true,false>(v.data(),roots,m);
            for(unsigned row=0;row<K;++row)
                out[row*256+id]=multiply(v[row],tables.forward_phase[size_t(prime)*N+id*K+row],m);
        }
    };
    forward(a,fa);
    forward(b,fb);
    for(unsigned row=0;row<K;++row) {
        U* x=&fa[row*256];
        U* y=&fb[row*256];
        transform<256,false,false>(x,roots+2*K,m);
        transform<256,false,false>(y,roots+2*K,m);
        for(unsigned i=0;i<256;++i) x[i]=multiply(x[i],y[i],m);
        transform<256,false,true>(x,roots+2*K,m);
        for(unsigned id=0;id<256;++id)
            stage[id*K+row]=multiply(x[id],tables.inverse_phase[size_t(prime)*N+row*256+id],m);
    }
    for(unsigned id=0;id<256;++id) {
        U* x=&stage[id*K];
        transform<K,true,true>(x,roots+K,m);
        for(unsigned i=0;i<K;++i) assert(x[i]==mul(expected[id+256*i],plan.crt_inverse[prime],m.p));
    }
}
int main() {
    std::mt19937 rng(61329);
    for(unsigned N:{32768u,65536u}) {
        unsigned W=N==65536?1747:868;
        std::vector<U> q((W+31)/32,UINT32_MAX);
        q.front()=0u-(N==65536?9699327u:43646975u);
        q.back()=(1u<<(W%32))-1;
        auto plan=two_digit::make(N,W,q).transform;
        auto tables=make_tables(plan);
        for(unsigned j=0;j<plan.pcount;++j) {
            if(N==65536) check<256>(plan,tables,j,rng);
            else check<128>(plan,tables,j,rng);
        }
        std::cout<<"PASS N="<<N<<" "<<plan.pcount<<" prime frequency maps and phases, 3 full negacyclic products\n";
    }
}
