// Mathematical contract for a warp-oriented RNS NTT schedule. No CUDA execution.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
#include "../core/rns-plan.hpp"
using U=uint32_t;
static U mul(U a,U b,U p){return uint64_t(a)*b%p;}
static U add(U a,U b,U p){U x=a+b;return x>=p?x-p:x;}
static U sub(U a,U b,U p){return a>=b?a-b:a+p-b;}
static U power(U a,U n,U p){return bytecrt::rns_detail::pow_mod(a,n,p);}
static unsigned reverse(unsigned i,unsigned n){unsigned r=0;for(n>>=1;n;n>>=1){r=(r<<1)|(i&1);i>>=1;}return r;}

// DIF emits bit-reversed frequencies. DIT inverse consumes exactly that order.
static void dif(std::vector<U>& x,U root,U p){
    unsigned n=x.size();
    for(unsigned d=n/2;d;d>>=1){
        U step=power(root,n/(2*d),p);
        for(unsigned first=0;first<n;first+=2*d){
            U w=1;
            for(unsigned j=0;j<d;++j){
                U a=x[first+j],b=x[first+j+d];
                x[first+j]=add(a,b,p);x[first+j+d]=mul(sub(a,b,p),w,p);w=mul(w,step,p);
            }
        }
    }
}
static void inverse_dit(std::vector<U>& x,U inverse_root,U p){
    unsigned n=x.size();
    for(unsigned d=1;d<n;d<<=1){
        U step=power(inverse_root,n/(2*d),p);
        for(unsigned first=0;first<n;first+=2*d){
            U w=1;
            for(unsigned j=0;j<d;++j){
                U a=x[first+j],b=mul(x[first+j+d],w,p);
                x[first+j]=add(a,b,p);x[first+j+d]=sub(a,b,p);w=mul(w,step,p);
            }
        }
    }
}
// Independent natural-order radix-2 reference, with input bit permutation.
static std::vector<U> reference(std::vector<U> x,U root,U p,bool normalize=false){
    unsigned n=x.size();
    for(unsigned i=0;i<n;++i){unsigned j=reverse(i,n);if(i<j)std::swap(x[i],x[j]);}
    for(unsigned length=2;length<=n;length<<=1){
        U step=power(root,n/length,p);
        for(unsigned base=0;base<n;base+=length){
            U w=1;
            for(unsigned j=0;j<length/2;++j){
                U left=x[base+j],right=mul(w,x[base+j+length/2],p);
                x[base+j]=add(left,right,p);x[base+j+length/2]=sub(left,right,p);w=mul(w,step,p);
            }
        }
    }
    if(normalize){U inverse_n=power(n,p-2,p);for(auto& a:x)a=mul(a,inverse_n,p);}
    return x;
}

static void check(unsigned N,unsigned M,U p,unsigned pattern){
    assert(N%M==0 && p<(U{1}<<31));unsigned K=N/M;
    U psi=bytecrt::rns_detail::primitive_2n_root(p,N),omega=mul(psi,psi,p);
    U inverse_omega=power(omega,p-2,p),inverse_psi=power(psi,p-2,p),inverse_n=power(N,p-2,p);
    std::vector<U> phase(N),twist(N),untwist(N);phase[0]=twist[0]=untwist[0]=1;
    for(unsigned i=1;i<N;++i){phase[i]=mul(phase[i-1],omega,p);twist[i]=mul(twist[i-1],psi,p);untwist[i]=mul(untwist[i-1],inverse_psi,p);}
    std::mt19937 rng(79139+N+pattern);std::vector<U> a(N),b(N),ta(N),tb(N);
    for(unsigned i=0;i<N;++i){
        a[i]=pattern==0?0:pattern==1?p-1:rng()%p;b[i]=pattern==0?0:pattern==1?p-1:rng()%p;
        ta[i]=mul(a[i],twist[i],p);tb[i]=mul(b[i],twist[i],p);
    }
    auto fa=reference(ta,omega,p),fb=reference(tb,omega,p);
    auto expected=fa;for(unsigned i=0;i<N;++i)expected[i]=mul(fa[i],fb[i],p);
    expected=reference(expected,inverse_omega,p,true);
    for(unsigned i=0;i<N;++i)expected[i]=mul(expected[i],untwist[i],p);
    if(N<=64){
        std::vector<U> school(N,0);
        for(unsigned i=0;i<N;++i)for(unsigned j=0;j<N;++j){U v=mul(a[i],b[j],p);unsigned t=i+j;school[t%N]=t<N?add(school[t],v,p):sub(school[t%N],v,p);}
        assert(school==expected);
    }
    std::vector<U> stage_a(N),stage_b(N),middle(N),actual(N);
    for(unsigned id=0;id<M;++id){
        std::vector<U> row_a(K),row_b(K);
        for(unsigned i=0;i<K;++i){row_a[i]=ta[id+i*M];row_b[i]=tb[id+i*M];}
        dif(row_a,power(omega,M,p),p);dif(row_b,power(omega,M,p),p);
        for(unsigned r=0;r<K;++r){U factor=phase[id*reverse(r,K)];stage_a[r*M+id]=mul(row_a[r],factor,p);stage_b[r*M+id]=mul(row_b[r],factor,p);}
    }
    for(unsigned r=0;r<K;++r){
        std::vector<U> row_a(stage_a.begin()+r*M,stage_a.begin()+(r+1)*M);
        std::vector<U> row_b(stage_b.begin()+r*M,stage_b.begin()+(r+1)*M);
        dif(row_a,power(omega,K,p),p);dif(row_b,power(omega,K,p),p);
        for(unsigned s=0;s<M;++s){
            unsigned frequency=reverse(r,K)+K*reverse(s,M);
            assert(row_a[s]==fa[frequency] && row_b[s]==fb[frequency]);
            row_a[s]=mul(row_a[s],row_b[s],p);
        }
        inverse_dit(row_a,power(inverse_omega,K,p),p);
        for(unsigned id=0;id<M;++id){unsigned exponent=id*reverse(r,K);middle[id*K+r]=mul(row_a[id],phase[(N-exponent)%N],p);}
    }
    for(unsigned id=0;id<M;++id){
        std::vector<U> row(middle.begin()+id*K,middle.begin()+(id+1)*K);
        inverse_dit(row,power(inverse_omega,M,p),p);
        for(unsigned i=0;i<K;++i){unsigned coefficient=id+i*M;actual[coefficient]=mul(mul(row[i],inverse_n,p),untwist[coefficient],p);}
    }
    assert(actual==expected);
    std::cout<<"PASS N="<<N<<" M="<<M<<" p="<<p<<" pattern="<<pattern<<": full frequency map and negacyclic output\n";
}
int main(){
    for(unsigned pattern=0;pattern<3;++pattern){check(16,4,2013265921,pattern);check(64,8,2013265921,pattern);}
    for(unsigned N:{32768u,65536u}){
        U p=(U{1}<<31)-2*N+1;while(!bytecrt::rns_detail::prime32(p))p-=2*N;
        check(N,256,p,1);check(N,256,p,2);
    }
}
