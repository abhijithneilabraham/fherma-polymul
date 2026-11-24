// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "two-primes.hpp"
#include "warp-tables.hpp"
#include <stdexcept>

namespace two_digit {
using bytecrt::warp_ntt::Twiddle;
struct Plan {
    bytecrt::RnsPlan transform;
    std::vector<uint8_t> encode,decode;
    std::vector<uint32_t> primes;
    std::vector<uint32_t> reciprocal;
    std::vector<float> inverse;
    std::vector<Twiddle> theta,inverse_theta;
};

// One warp consumes eight columns and 32 bytes of K per matrix instruction.
// Each lane reads the low and high halves of its column together.
inline std::vector<uint64_t> pack_crt_weights(const std::vector<uint8_t>& matrix,
                                              unsigned limbs, unsigned depth) {
    const unsigned columns = 4 * limbs;
    if (depth % 32 || matrix.size() != size_t(columns) * depth)
        throw std::invalid_argument("CRT matrix shape");
    const unsigned tiles = ((columns + 55) / 56) * 7;
    std::vector<uint64_t> fragments(size_t(tiles) * depth);
    for (unsigned column = 0; column < columns; ++column) {
        for (unsigned k = 0; k < depth; ++k) {
            const unsigned lane = (column % 8) * 4 + (k % 16) / 4;
            const size_t index = size_t(column / 8) * depth + (k / 32) * 32 + lane;
            const unsigned byte = ((k % 32) / 16) * 4 + k % 4;
            fragments[index] |= uint64_t(matrix[size_t(column) * depth + k]) << (8 * byte);
        }
    }
    return fragments;
}
inline Plan make(unsigned n,unsigned w,const std::vector<uint32_t>& q) {
    using namespace bytecrt::rns_detail;
    const bool ranked = n==65536 && w==1747;
    if (!ranked && !(n==32768 && w==868)) throw std::invalid_argument("two-digit width");
    unsigned shift=ranked?880:440, count=ranked?58:30;
    unsigned enc_pad=ranked?128:64, crt_pad=ranked?480:256, limbs=(w+31)/32;
    uint32_t gap=ranked?9699327:43646975;
    const PrimeRoot* constants=ranked?large:small;
    Words product{1};
    for(unsigned j=0;j<count;++j) mul_small(product,constants[j].p);
    // Exact centered bound for either coefficient of the two-digit product.
    auto ones=[](unsigned bits) {
        Words x((bits+31)/32,UINT32_MAX);
        if(bits%32) x.back()=(1u<<(bits%32))-1;
        return x;
    };
    Words low=ones(shift),high=ones(w-shift);
    Words bound=multiply(high,high);
    mul_small(bound,gap); mul_small(bound,1u<<(2*shift-w));
    Words square=multiply(low,low);
    bound.resize(std::max(bound.size(),square.size()),0);
    uint64_t carry=0;
    for(unsigned j=0;j<bound.size();++j) {
        uint64_t sum=uint64_t(bound[j])+(j<square.size()?square[j]:0)+carry;
        bound[j]=uint32_t(sum); carry=sum>>32;
    }
    if(carry) bound.push_back(uint32_t(carry));
    Words cross=multiply(low,high); mul_small(cross,2);
    if(compare(cross,bound)>0) bound=cross;
    mul_small(bound,4*n);
    if(compare(product,bound)<=0) throw std::runtime_error("centered two-digit range");
    Plan s;auto& t=s.transform;t.n=n;t.w=w;t.l=limbs;t.pcount=2*count;
    s.encode.assign(8*count*enc_pad,0);
    s.decode.assign(4*limbs*crt_pad,0);
    auto column=[&](unsigned term,Words v) {
        for(unsigned byte=0;byte<4;++byte) {
            for(unsigned limb=0;limb<limbs;++limb) {
                uint32_t x=limb<v.size()?v[limb]:0;
                for(unsigned digit=0;digit<4;++digit)
                    s.decode[(4*limb+digit)*crt_pad+4*term+byte]=x>>(8*digit);
            }
            if(byte<3) {mul_small(v,256);v=mod_q(v,q);}
        }
    };
    auto shifted=[&](Words v) {
        v.insert(v.begin(),shift/32,0);mul_small(v,1u<<(shift%32));
        return mod_q(v,q);
    };
    for(unsigned j=0;j<count;++j) {
        uint32_t p=constants[j].p,theta=constants[j].theta;
        auto fixed=[&](uint32_t value){return Twiddle{value,uint32_t((uint64_t(value)<<32)/p)};};
        if(uint64_t(theta)*theta%p!=(uint64_t(gap)<<(2*shift-w))%p)
            throw std::runtime_error("invalid digit twist");
        s.primes.push_back(p);s.reciprocal.push_back((uint64_t{1}<<48)/p);
        s.inverse.push_back(float(1.0/double(p)));
        unsigned padding=shift%32;
        uint32_t adjusted=uint64_t(theta)*pow_mod(1u<<padding,p-2,p)%p;
        s.theta.push_back(fixed(adjusted));s.inverse_theta.push_back(fixed(pow_mod(theta,p-2,p)));
        for(unsigned half=0;half<2;++half) {
            uint32_t weight=half?adjusted:1u;
            unsigned length=half?(w+7)/8-4*(shift/32):shift/8;
            for(unsigned k=0;k<length;++k) {
                unsigned tile=(k/32)*(count/2)+j/2;
                for(unsigned out=0;out<4;++out) {
                    unsigned lane=4*(4*(j%2)+out)+(k%16)/4;
                    unsigned word=128*tile+64*half+32*((k%32)/16)+lane;
                    s.encode[4*word+k%4]=weight>>(8*out);
                }
                weight=uint64_t(weight)*256%p;
            }
        }
        uint32_t rem;Words cofactor=div_small(product,p,rem);
        if(rem) throw std::runtime_error("invalid cofactor");
        column(j,mod_q(cofactor,q));column(j+count,shifted(cofactor));
        uint32_t root=primitive_2n_root(p,n);
        t.primes.push_back(p);t.psi_roots.push_back(root);
        t.omega_roots.push_back(uint64_t(root)*root%p);t.n_inverse.push_back(pow_mod(n,p-2,p));
        // Both planes contribute half of the weighted evaluation.
        t.crt_inverse.push_back(pow_mod(uint64_t(2)*mod_small(cofactor,p)%p,p-2,p));
    }
    auto duplicate=[](auto& v) { auto original=v;v.insert(v.end(),original.begin(),original.end()); };
    duplicate(t.primes);duplicate(t.psi_roots);duplicate(t.omega_roots);
    duplicate(t.n_inverse);duplicate(t.crt_inverse);
    Words negative=q;subtract(negative,mod_q(product,q));
    column(2*count,negative);column(2*count+1,shifted(negative));
    return s;
}
}
