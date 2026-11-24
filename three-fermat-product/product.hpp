// Reference pointwise product in diminished-one representation.
#pragma once
#include "diminished.hpp"
#ifdef __CUDACC__
#define FERMAT_PRODUCT_HD __host__ __device__ __forceinline__
#else
#define FERMAT_PRODUCT_HD inline
#endif

namespace fermat_native {
template<unsigned W>
FERMAT_PRODUCT_HD int64_t raw_column(const Stored<W>& a,const Stored<W>& b,unsigned column) {
    // Each word contains two radix-65536 digits. The two output diagonals
    // use the same pair of A digits and three adjacent B digits.
    int64_t even=0, odd=0;
    for(unsigned i=0;i<W;++i) {
        const unsigned current=(column+W-i)%W;
        const unsigned prior=current==0?W-1:current-1;
        uint32_t x=a.limb[i], y=b.limb[current], previous=b.limb[prior];
        const int64_t lo=x&65535u, hi=x>>16;
        const int64_t same=lo*(y&65535u);
        const int64_t cross=lo*(y>>16)+hi*(y&65535u);
        const int64_t carry_diagonal=hi*(previous>>16);
        even += i<=column ? same : -same;
        odd += i<=column ? cross : -cross;
        even += i<column ? carry_diagonal : -carry_diagonal;
    }
    return even + odd*65536 + a.limb[column] + b.limb[column];
}

template<unsigned W>
inline void host_product(const Stored<W>& a,const Stored<W>& b,Stored<W>& out) {
    if(a.zero || b.zero) {out={};out.zero=1;return;}
    int64_t columns[W];
    for(unsigned i=0;i<W;++i) columns[i]=raw_column(a,b,i);
    int64_t carry=0;
    for(unsigned i=0;i<W;++i) {
        int64_t previous=i==0 ? -(columns[W-1]>>32) : columns[i-1]>>32;
        int64_t total=int64_t(uint32_t(columns[i]))+previous+carry;
        out.limb[i]=uint32_t(total);
        carry=total>>32;
    }
    int64_t correction=-carry;
    for(unsigned i=0;i<W;++i) {
        int64_t sum=int64_t(out.limb[i])+correction;
        out.limb[i]=uint32_t(sum);correction=sum>>32;
    }
    out.zero=correction!=0;
}
} // namespace fermat_native
#undef FERMAT_PRODUCT_HD
