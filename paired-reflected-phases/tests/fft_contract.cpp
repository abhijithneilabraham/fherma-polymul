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

static void cyclic(std::vector<U>& a,U root,U p) {
    const unsigned n=a.size();
    for(unsigned i=1,j=0;i<n;++i) {
        unsigned bit=n/2;
        while(j&bit) {j^=bit;bit/=2;}
        j^=bit;
        if(i<j) std::swap(a[i],a[j]);
    }
    for(unsigned width=2;width<=n;width*=2) {
        U step=pow_mod(root,n/width,p);
        for(unsigned begin=0;begin<n;begin+=width) {
            U factor=1;
            for(unsigned j=0;j<width/2;++j) {
                U x=a[begin+j],y=product(a[begin+j+width/2],factor,p);
                a[begin+j]=(uint64_t(x)+y)%p;
                a[begin+j+width/2]=(uint64_t(x)+p-y)%p;
                factor=product(factor,step,p);
            }
        }
    }
}

static void check(unsigned n,unsigned width,unsigned gap) {
    const unsigned inner=n/32;
    std::vector<U> q((width+31)/32,UINT32_MAX);
    q.front()=0u-gap;q.back()=(1u<<(width%32))-1;
    auto plan=two_digit::make(n,width,q).transform;
    auto tables=make_tables(plan);
    assert(tables.phase.size()==size_t(plan.pcount/2)*16*(inner+phase_overhang(inner)));
    std::mt19937 rng(n);

    // Every phase appears on the output side, at both extremes of the
    // cyclic carry and for a varying second input index.
    for(unsigned prime=0;prime<plan.pcount/2;++prime) {
        U p=plan.primes[prime],psi=plan.psi_roots[prime];
        const Mod mod=tables.mods[prime];
        U radix=(uint64_t(1)<<32)%p;
        for(unsigned row=0;row<32;++row) {
            U wrap=pow_mod(psi,inner*(2*reversed(row,32)+1),p);
            for(unsigned i=0;i<inner;++i) {
                auto address=input_phase(prime,inner,i,row);
                assert(address<tables.phase.size());
                assert((address&3)==((row&1)?3-i%4:i%4));
                assert((address&~3u)==(input_phase(prime,inner,i&~3u,row)&~3u));
                assert(output_phase(prime,inner,i,row)<tables.phase.size());
                const std::array<unsigned,3> indices={0u,inner-1,(73*i+row)%inner};
                for(unsigned j:indices) {
                    auto phase=[&](unsigned index){return tables.phase[input_phase(prime,inner,index,row)];};
                    unsigned k=(i+j)%inner;
                    U pair=multiply(phase(i),phase(j),mod);
                    U actual=multiply(pair,tables.phase[output_phase(prime,inner,k,row)],mod);
                    assert(actual==(i+j>=inner?product(radix,wrap,p):radix));
                }
            }
        }
    }
    // Test all output coefficients with negative wraparound at top degree.
    for(unsigned prime:std::array<unsigned,3>{0,plan.pcount/4,plan.pcount/2-1}) {
        U p=plan.primes[prime],psi=plan.psi_roots[prime];
        const Mod mod=tables.mods[prime];
        U radix=(uint64_t(1)<<32)%p;
        U scale=product(product(radix,plan.n_inverse[prime],p),plan.crt_inverse[prime],p);
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
                U phase=tables.phase[input_phase(prime,inner,id,row)];
                x[id]=multiply(x[id],phase,mod);
                y[id]=multiply(product(y[id],scale,p),phase,mod);
            }
            cyclic(x,root,p);cyclic(y,root,p);
            for(unsigned j=0;j<inner;++j)x[j]=multiply(x[j],y[j],mod);
            cyclic(x,pow_mod(root,p-2,p),p);
            U inverse=pow_mod(outer,p-2,p);
            for(unsigned id=0;id<inner;++id) {
                U term=multiply(x[id],tables.phase[output_phase(prime,inner,id,row)],mod);
                for(unsigned j=0;j<32;++j) {
                    U& dest=actual[id+j*inner];
                    dest=(uint64_t(dest)+term)%p;
                    term=product(term,inverse,p);
                }
            }
        }
        for(unsigned i=0;i<n;++i) assert(actual[i]==product(expected[i],plan.crt_inverse[prime],p));
    }
    std::cout<<"PASS all reflected phases and three full products at N="<<n<<"\n";
}
int main(){check(32768,868,43646975);check(65536,1747,9699327);}
