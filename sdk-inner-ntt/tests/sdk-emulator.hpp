// Emulate the documented SDK transform. Native code is linked separately in CI.
#pragma once
namespace cupqc { enum class nttDirection {FORWARD,INVERSE}; }
namespace bytecrt::warp_ntt {
using U=uint32_t;
using bytecrt::rns_detail::pow_mod;
static U product(U a,U b,U p){return uint64_t(a)*b%p;}
#include "cyclic-reference.hpp"
template<unsigned J,cupqc::nttDirection Direction> struct InnerNtt {
    void make_twiddles(U* table,U p,U root) {
        assert(blockDim.x==1 && threadIdx.x==0);
        table[0]=1;
        for(unsigned i=1;i<J;++i)table[i]=product(table[i-1],root,p);
    }
    void transform_twiddles_to_mont(U* table,U p) {
        assert(blockDim.x==256);
        for(unsigned i=threadIdx.x;i<J;i+=256)table[i]=product(table[i],(uint64_t(1)<<32)%p,p);
    }
    void execute(U* a,const U* table,U p,U scale=1) {
        if(threadIdx.x)return;
        assert(table[0]==(uint64_t(1)<<32)%p);
        U root=product(table[1],pow_mod(table[0],p-2,p),p);
        std::vector<U> values(a,a+J);
        cyclic(values,root,p);
        for(unsigned i=0;i<J;++i)a[i]=product(values[i],scale,p);
    }
};
}
