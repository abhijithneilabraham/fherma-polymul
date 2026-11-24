// Compose the production phase table with an independent dense-by-sparse convolution.
#include "../two-plan.hpp"
#include <array>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using U=uint32_t;
using bytecrt::rns_detail::pow_mod;
using namespace bytecrt::warp_ntt;
static U product(U a,U b,U p) {return uint64_t(a)*b%p;}

#include "cyclic-reference.hpp"

static void check(unsigned n,unsigned width,unsigned gap) {
    const unsigned inner=n/32;
    std::vector<U> q((width+31)/32,UINT32_MAX);
    q.front()=0u-gap;q.back()=(1u<<(width%32))-1;
    auto plan=two_digit::make(n,width,q).transform;
    auto tables=make_tables(plan);
    assert(tables.sdk.size()==plan.pcount/2);
    std::mt19937 rng(n);
    for(unsigned prime=0; prime<tables.sdk.size(); ++prime) {
        const auto field=tables.sdk[prime];
        assert(field.p==plan.primes[prime]);
        assert(pow_mod(field.root,inner/2,field.p)==field.p-1);
        assert(product(field.root,field.inverse_root,field.p)==1);
        assert(product(inner,field.inverse_length,field.p)==1);
        for(unsigned row=0;row<32;++row) {
            U step=pow_mod(plan.psi_roots[prime],2*reversed(row,32)+1,field.p);
            U power=1;
            for(unsigned id=0;id<inner;++id) {
                assert(tables.forward_phase[prime*n+32*id+row]==product(tables.mods[prime].r2,power,field.p));
                U normalized=product(tables.inverse_phase[prime*n+row*inner+id],power,field.p);
                assert(product(normalized,32,field.p)==plan.crt_inverse[prime]);
                power=product(power,step,field.p);
            }
        }
    }
    // Test all output coefficients with negative wraparound at top degree.
    for(unsigned prime:std::array<unsigned,3>{0,plan.pcount/4,plan.pcount/2-1}) {
        U p=plan.primes[prime],psi=plan.psi_roots[prime];
        const Mod mod=tables.mods[prime];
        U root=pow_mod(psi,64,p);
        std::vector<U> a(n),b(n),expected(n),actual(n);
        for(auto& x:a)x=rng()%p;
        for(unsigned j:std::array<unsigned,3>{0,inner+1,n-1}) {
            b[j]=1+rng()%(p-1);
            for(unsigned i=0;i<n;++i) {
                unsigned k=(i+j)%n;U v=product(a[i],b[j],p);
                expected[k]=(uint64_t(expected[k])+(i+j>=n?p-v:v))%p;
            }
        }
        for(unsigned row=0;row<32;++row) {
            U outer=pow_mod(psi,inner*(2*reversed(row,32)+1),p);
            std::vector<U> x(inner),y(inner);
            for(unsigned id=0;id<inner;++id) {
                for(unsigned j=32;j-->0;) {
                    x[id]=(uint64_t(x[id])*outer+a[id+j*inner])%p;
                    y[id]=(uint64_t(y[id])*outer+b[id+j*inner])%p;
                }
                U phase=tables.forward_phase[prime*n+32*id+row];
                x[id]=multiply(x[id],phase,mod);
                y[id]=multiply(y[id],phase,mod);
            }
            cyclic(x,root,p);cyclic(y,root,p);
            for(unsigned j=0;j<inner;++j)x[j]=multiply(x[j],y[j],mod);
            cyclic(x,pow_mod(root,p-2,p),p);
            U inverse=pow_mod(outer,p-2,p);
            for(unsigned id=0;id<inner;++id) {
                U term=multiply(product(x[id],tables.sdk[prime].inverse_length,p),tables.inverse_phase[prime*n+row*inner+id],mod);
                for(unsigned j=0;j<32;++j) {
                    U& dest=actual[id+j*inner];
                    dest=(uint64_t(dest)+term)%p;
                    term=product(term,inverse,p);
                }
            }
        }
        for(unsigned i=0;i<n;++i) assert(actual[i]==product(expected[i],plan.crt_inverse[prime],p));
    }
    std::cout<<"PASS all SDK phases and three full products at N="<<n<<"\n";
}
int main(){check(32768,868,43646975);check(65536,1747,9699327);}
