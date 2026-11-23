// Check all coefficients through the 32 x J decomposition and packaged tables.
#include "../four-plan.hpp"
#include <cassert>
#include <iostream>
#include <random>
using U=uint32_t;
using namespace bytecrt::warp_ntt;
using bytecrt::rns_detail::pow_mod;
U mul(U a,U b,U p){return uint64_t(a)*b%p;}
void cyclic(U* x,unsigned J,U root,U p) {
    for(unsigned i=0;i<J;++i){unsigned j=reversed(i,J);if(i<j)std::swap(x[i],x[j]);}
    for(unsigned len=2;len<=J;len*=2) {
        U step=pow_mod(root,J/len,p);
        for(unsigned i=0;i<J;i+=len){
            U w=1;
            for(unsigned t=0;t<len/2;++t){
                U a=x[i+t], b=mul(w,x[i+t+len/2],p);
                x[i+t]=add(a,b,p);x[i+t+len/2]=sub(a,b,p);w=mul(w,step,p);
            }
        }
    }
}
void check(unsigned n,unsigned width,unsigned gap) {
    constexpr unsigned K=32;
    unsigned J=n/K;
    std::vector<U> q((width+31)/32,UINT32_MAX);q.front()=0u-gap;q.back()=(1u<<(width%32))-1;
    auto plan=four_plan::make(n,width,q).transform;
    auto t=make_tables(plan);
    unsigned i=plan.pcount-1;
    U p=plan.primes[i],psi=plan.psi_roots[i],omega=pow_mod(psi,2*K,p);
    U rinv=pow_mod((uint64_t{1}<<32)%p,p-2,p);
    U gf[K][K],gi[K][K];
    for(unsigned row=0;row<K;++row) {
        U g=pow_mod(psi,J*(2*reversed(row,K)+1),p);
        for(unsigned k=0;k<K;++k){gf[row][k]=pow_mod(g,k,p);gi[row][k]=pow_mod(gf[row][k],p-2,p);}
    }
    std::mt19937 rng(n);
    std::vector<U> a(n),b(n),expected(n),first(n),second(n),result(n);
    for(auto& x:a)x=rng()%p;
    for(unsigned j:{0u,1u,J-1,J,n-1}){
        b[j]=rng()%p;
        for(unsigned k=0;k<n;++k){
            U v=mul(a[k],b[j],p);unsigned idx=(k+j)%n;
            expected[idx]=k+j<n?add(expected[idx],v,p):sub(expected[idx],v,p);
        }
    }
    auto forward=[&](const auto& in,auto& out){
        for(unsigned id=0;id<J;++id)for(unsigned row=0;row<K;++row){
            U v=0;for(unsigned k=0;k<K;++k)v=add(v,mul(in[id+J*k],gf[row][k],p),p);
            out[row*J+id]=mul(mul(v,t.forward_phase[size_t(i)*n+id*K+row],p),rinv,p);
        }
    };
    forward(a,first);forward(b,second);
    for(unsigned row=0;row<K;++row){
        U* x=first.data()+row*J;U* y=second.data()+row*J;
        cyclic(x,J,omega,p);cyclic(y,J,omega,p);
        for(unsigned id=0;id<J;++id)x[id]=mul(mul(x[id],y[id],p),rinv,p);
        cyclic(x,J,pow_mod(omega,p-2,p),p);
        for(unsigned id=0;id<J;++id)
            result[id/4*K*4+row*4+id%4]=mul(mul(x[id],t.inverse_phase[(size_t(i)*K+row)*J+id],p),rinv,p);
    }
    for(unsigned id=0;id<J;++id)for(unsigned k=0;k<K;++k){
        U v=0;
        for(unsigned row=0;row<K;++row)v=add(v,mul(result[id/4*K*4+row*4+id%4],gi[row][k],p),p);
        assert(v==mul(expected[id+J*k],plan.crt_inverse[i],p));
    }
    std::cout<<"PASS "<<n<<" coefficients through 32 x "<<J<<" phases and negacyclic wrap\n";
}
int main(){check(32768,868,43646975);check(65536,1747,9699327);}
