// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "encode.hpp"
#include "inverse-field.hpp"
#include "prime-roots.hpp"
#include "warp-tables.hpp"
#include <iterator>

namespace seven_plan {
struct Plan {
    bytecrt::RnsPlan transform;
    std::vector<seven_encode::Field> encode_fields;
    std::vector<seven_inverse::Field> inverse_fields;
    std::vector<uint32_t> matrix_words,reciprocal48;
    std::vector<uint8_t> crt;
};
inline Plan make(unsigned n,unsigned w,const std::vector<uint32_t>& q) {
    using namespace bytecrt::rns_detail;
    bool ranked=n==65536 && w==1747;
    if(!ranked && !(n==32768 && w==868))throw std::invalid_argument("seven-digit point");
    const auto* roots=ranked?roots1747:roots868;
    const unsigned count=ranked?18:10,L=(w+31)/32,planes=count*7,Kpad=32*((planes+14)/8);
    unsigned gap=ranked?9699327:43646975;
    Words M{1};
    for(unsigned j=0;j<count;++j)mul_small(M,roots[j].prime);
    Words guard=ranked?Words(std::begin(guard1747),std::end(guard1747))
                      :Words(std::begin(guard868),std::end(guard868));
    if(compare(M,guard)<=0)throw std::runtime_error("signed reconstruction bound");
    Plan s;auto& t=s.transform;t.n=n;t.w=w;t.l=L;t.pcount=count;
    s.matrix_words.resize(7*count*32);s.crt.resize(4*L*Kpad);
    auto put=[&](unsigned term,Words value,unsigned shift) {
        value.insert(value.begin(),shift/32,0);mul_small(value,1u<<(shift%32));
        value=mod_q(value,q);
        for(unsigned b=0;b<4;++b) {
            for(unsigned j=0;j<value.size();++j)
                for(unsigned k=0;k<4;++k)s.crt[(4*j+k)*Kpad+4*term+b]=value[j]>>(8*k);
            mul_small(value,256);value=mod_q(value,q);
        }
    };
    for(unsigned j=0;j<count;++j) {
        uint32_t p=roots[j].prime, r=roots[j].digit_scale, omega=roots[j].digit_root;
        uint32_t root=primitive_2n_root(p,n), invn=pow_mod(n,p-2,p);
        if(uint64_t(pow_mod(r,7,p))*pow_mod(2,w,p)%p!=gap%p)
            throw std::runtime_error("invalid digit twist");
        t.primes.push_back(p);t.psi_roots.push_back(root);
        t.omega_roots.push_back(uint64_t(root)*root%p);t.n_inverse.push_back(invn);
        uint32_t rem;Words cofactor=div_small(M,p,rem);if(rem)throw std::runtime_error("cofactor");
        t.crt_inverse.push_back(pow_mod(mod_small(cofactor,p),p-2,p));
        seven_encode::Field enc{};seven_inverse::Field inv{};
        enc.mod=inv.mod=bytecrt::warp_ntt::make_mod(p,invn);
        enc.reciprocal=UINT64_MAX/p;
        enc.forward=rader7::prepare(omega,enc.mod);
        inv.inverse=rader7::prepare(pow_mod(omega,p-2,p),inv.mod);
        auto fixed=[&](uint32_t v){return bytecrt::warp_ntt::Twiddle{v,uint32_t((uint64_t(v)<<32)/p)};};
        for(unsigned d=0;d<7;++d) {
            unsigned shift=(d*w+6)/7;
            uint32_t scale=uint64_t(pow_mod(2,shift,p))*pow_mod(r,d,p)%p;
            enc.digit_twist[d]=fixed(scale);
            if(ranked)seven_encode::set_weights<18>(s.matrix_words.data(),d,j,scale,p);
            else seven_encode::set_weights<10>(s.matrix_words.data(),d,j,scale,p);
            inv.recovery[d]=fixed(pow_mod(uint64_t(7)*scale%p,p-2,p));
            put(d*count+j,cofactor,shift);
        }
        s.encode_fields.push_back(enc);s.inverse_fields.push_back(inv);
        s.reciprocal48.push_back((uint64_t{1}<<48)/p);

    }
    Words negative=q;subtract(negative,mod_q(M,q));
    for(unsigned d=0;d<7;++d)put(planes+d,negative,(d*w+6)/7);
    return s;
}
}
