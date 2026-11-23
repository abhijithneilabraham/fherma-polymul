// SPDX-License-Identifier: MIT
#pragma once
#include "folded-math.hpp"
#include "core/rns-plan.hpp"
namespace folded {
constexpr unsigned joint_terms=136,joint_pad=4*joint_terms;
struct JointWeights {
    uint8_t matrix[4*L][joint_pad];
    uint32_t reciprocal48[16];
};
inline JointWeights make_joint_weights(const Plan& plan) {
    using namespace bytecrt::rns_detail;
    JointWeights result{};
    Words q(L,UINT32_MAX);
    q.front()=0u-gap;q.back()=(1u<<(W%32))-1;
    Words modulus(plan.product,plan.product+16);trim(modulus);
    for(unsigned p=0;p<16;++p) result.reciprocal48[p]=(uint64_t(1)<<48)/plan.primes[p].p;
    for(unsigned term=0;term<joint_terms;++term) {
        unsigned digit=term%8;
        Words weight;
        if(term<128) {
            const uint32_t* base=plan.primes[term/8].crt;
            weight.assign(base,base+16);trim(weight);
        } else {
            weight=q;subtract(weight,mod_q(modulus,q));
        }
        unsigned shift=position(digit);
        weight.insert(weight.begin(),shift/32,0);mul_small(weight,1u<<(shift%32));
        weight=mod_q(weight,q);
        for(unsigned b=0;b<4;++b) {
            for(unsigned limb=0;limb<L;++limb) {
                uint32_t word=limb<weight.size()?weight[limb]:0;
                for(unsigned out=0;out<4;++out) result.matrix[4*limb+out][4*term+b]=word>>(8*out);
            }
            if(b!=3) {mul_small(weight,256);weight=mod_q(weight,q);}
        }
    }
    return result;
}
}
