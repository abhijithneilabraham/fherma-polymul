// SPDX-License-Identifier: MIT
#pragma once
#include "folded-plan.hpp"
namespace folded {
struct CrtWeights {
    uint32_t words[8][2][32][2];
    uint32_t reciprocal48[16];
};
inline CrtWeights make_crt_weights(const Plan& plan) {
    CrtWeights result{};
    for (unsigned p = 0; p < 16; ++p) {
        result.reciprocal48[p] = (uint64_t{1}<<48) / plan.primes[p].p;
        for (unsigned b = 0; b < 4; ++b) {
            unsigned k = 4*p+b;
            for (unsigned out = b; out < 64; ++out) {
                unsigned source = out-b;
                uint32_t cofactor = plan.primes[p].crt[source/4];
                uint32_t byte = (cofactor >> (8*(source%4))) & 255;
                unsigned lane = (out%8)*4 + (k%16)/4;
                result.words[out/8][k/32][lane][(k%32)/16] |= byte << (8*(k%4));
            }
        }
    }
    return result;
}
#ifdef __CUDACC__
__host__ __device__
#endif
inline void finish_crt_words(const uint64_t* raw, int alpha, const Plan& plan, uint32_t* output) {
    // Each raw word is below 2^47, and the signed correction is below 2^36.
    int64_t carry = 0;
    for (unsigned i = 0; i < 16; ++i) {
        int64_t sum = int64_t(raw[i]) + (int64_t(1)-alpha)*plan.product[i] + carry;
        output[i] = uint32_t(sum);
        carry = sum >> 32;
    }
}
}
