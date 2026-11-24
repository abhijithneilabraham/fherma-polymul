// Arithmetic modulo 2^(32*Words)+1. A nonzero residue x stores x-1.
#pragma once
#include <cstdint>

#ifdef __CUDACC__
#define D1_HD __host__ __device__ __forceinline__
#else
#define D1_HD inline
#endif

namespace fermat_native {
template<unsigned Words>
struct Stored {
    uint32_t limb[Words];
    uint32_t zero;
};

// The bit stream repeats with inverted polarity after each Words-word interval.
// Reading the shifted stream combines rotation and the sign from wraparound.
template<unsigned Words>
D1_HD uint32_t rotated_word(const Stored<Words>& v, unsigned out_word, unsigned exponent) {
    const unsigned e = exponent % (64*Words);
    unsigned start = out_word + 2*Words - (e+31)/32;
    if(start >= 2*Words) start-=2*Words;
    const unsigned bit = (32-e%32)%32;
    auto extended = [&](unsigned at) {
        if(at>=2*Words) at-=2*Words;
        return (at>=Words) ? ~v.limb[at-Words] : v.limb[at];
    };
    uint32_t low=extended(start);
    if (!bit) return low;
    return (low>>bit) | (extended(start+1)<<(32-bit));
}

D1_HD unsigned carry_into(unsigned generate, unsigned propagate, unsigned mask, unsigned initial) {
    const unsigned decisive=(~propagate)&mask;
    if (!decisive) return initial;
#ifdef __CUDA_ARCH__
    unsigned last=31u-__clz(decisive);
#else
    unsigned last=31u-__builtin_clz(decisive);
#endif
    return (generate>>last)&1u;
}

template<unsigned Local>
D1_HD void local_sum(const uint32_t (&a)[Local],const uint32_t (&b)[Local],bool subtract,
                     uint32_t (&result)[Local],unsigned& generate,unsigned& propagate) {
    uint64_t carry=0;
    uint32_t combined=~uint32_t{0};
    for(unsigned j=0;j<Local;++j) {
        const uint32_t term = subtract ? ~b[j] : b[j];
        uint64_t sum=uint64_t(a[j])+term+carry;
        result[j]=uint32_t(sum);
        carry=sum>>32;
        combined &= result[j];
    }
    generate=unsigned(carry);
    propagate=(combined==~uint32_t{0});
}

template<unsigned Local>
D1_HD void apply_carry(uint32_t (&result)[Local],unsigned carry) {
    for(unsigned j=0;j<Local;++j) {
        uint32_t word=result[j]+carry;
        carry=(word<result[j]);
        result[j]=word;
    }
}
} // namespace fermat_native
#undef D1_HD
