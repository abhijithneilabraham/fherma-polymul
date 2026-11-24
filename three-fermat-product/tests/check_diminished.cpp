#include "../diminished.hpp"
#include <algorithm>
using namespace fermat_native;

template<unsigned W>
void compute(const uint32_t* x,const uint32_t* y,unsigned ex,unsigned ey,unsigned sub,uint32_t* z) {
    constexpr unsigned G=8, Local=W/G;
    Stored<W> a{},b{};
    std::copy(x,x+W,a.limb);std::copy(y,y+W,b.limb);
    a.zero=x[W];b.zero=y[W];
    if(a.zero || b.zero) {
        for(unsigned i=0;i<W;++i)
            z[i]=a.zero?(sub?~rotated_word(b,i,ey):rotated_word(b,i,ey)):rotated_word(a,i,ex);
        z[W]=a.zero&&b.zero;
        return;
    }
    uint32_t sums[G][Local];unsigned generates=0, propagates=0;
    for(unsigned lane=0;lane<G;++lane) {
        uint32_t left[Local],right[Local];unsigned g,p;
        for(unsigned j=0;j<Local;++j) {
            left[j]=rotated_word(a,lane*Local+j,ex);
            right[j]=rotated_word(b,lane*Local+j,ey);
        }
        local_sum(left,right,bool(sub),sums[lane],g,p);
        generates|=g<<lane;propagates|=p<<lane;
    }
    const unsigned initial=1-carry_into(generates,propagates,255,0);
    z[W]=propagates==255;
    for(unsigned lane=0;lane<G;++lane) {
        apply_carry(sums[lane],carry_into(generates,propagates,(1u<<lane)-1,initial));
        std::copy(sums[lane],sums[lane]+Local,z+lane*Local);
    }
}

extern "C" void operation(unsigned words,const uint32_t* x,const uint32_t* y,unsigned ex,unsigned ey,unsigned sub,uint32_t* z) {
    switch(words) {
        case 32:compute<32>(x,y,ex,ey,sub,z);break;
        case 40:compute<40>(x,y,ex,ey,sub,z);break;
        case 48:compute<48>(x,y,ex,ey,sub,z);break;
    }
}

#include "../product.hpp"
template<unsigned W>
void product_checked(const uint32_t* x,const uint32_t* y,uint32_t* z) {
    Stored<W> a{},b{},out{};
    std::copy(x,x+W,a.limb);std::copy(y,y+W,b.limb);a.zero=x[W];b.zero=y[W];
    host_product(a,b,out);
    std::copy(out.limb,out.limb+W,z);z[W]=out.zero;
}
extern "C" void product(unsigned words,const uint32_t* x,const uint32_t* y,uint32_t* z) {
    switch(words) {
        case 32:product_checked<32>(x,y,z);break;
        case 40:product_checked<40>(x,y,z);break;
        case 48:product_checked<48>(x,y,z);break;
    }
}
