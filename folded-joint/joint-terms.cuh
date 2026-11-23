// SPDX-License-Identifier: MIT
#pragma once
#include "digit-step.cuh"
#include "joint-weights.hpp"
namespace folded {
__device__ __forceinline__ void joint_terms_for_digit(uint32_t* terms,unsigned digit,
        const Plan* plan,const Mod* mods,const JointWeights* weights) {
    uint64_t scaled_sum=0;
    for(unsigned p=0;p<16;++p) {
        const Prime& prime=plan->primes[p];
        Mod mod=mods[p];
        unsigned plane=8*p+digit;
        uint32_t x=terms[plane];
        x=digit_step<true,1>(x,digit,prime,mod);
        x=digit_step<true,2>(x,digit,prime,mod);
        x=digit_step<true,4>(x,digit,prime,mod);
        x=multiply(x,prime.recover[digit],mod);
        terms[plane]=x;
        scaled_sum+=uint64_t(x)*weights->reciprocal48[p];
    }
    terms[128+digit]=uint32_t((scaled_sum+(uint64_t(1)<<47))>>48);
}
}
