#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
#include "../warp-tables.hpp"
static std::array<uint32_t,256> exchange_values;
static unsigned active_lane,active_register,active_distance;
static uint32_t __shfl_xor_sync(unsigned mask,uint32_t value,unsigned distance){
    assert(mask==0xffffffffu && distance==active_distance);
    unsigned i=active_lane+32*active_register++;
    assert(value==exchange_values[i]);return exchange_values[(active_lane^distance)+32*(active_register-1)];
}
#define __device__
#define __forceinline__ inline
#include "../warp-fft.cuh"
using namespace bytecrt::warp_ntt;
static uint32_t modular(uint32_t a,uint32_t b,uint32_t p){return uint64_t(a)*b%p;}

template<unsigned Len,unsigned Distance,bool Inverse>
void check_step(std::vector<uint32_t>& values,const Twiddle* roots,const Mod& mod,uint32_t R,uint32_t inverse_R){
    auto before=values,expected=values,after=values;
    for(auto& v:expected)v=modular(v,inverse_R,mod.p);
    for(unsigned first=0;first<Len;first+=2*Distance)
        for(unsigned j=0;j<Distance;++j){
            uint32_t twiddle=roots[(Inverse?(Len-j*(Len/(2*Distance))):j*(Len/(2*Distance)))&(Len-1)].value;
            uint32_t a=expected[first+j],b=expected[first+j+Distance];
            if constexpr(Inverse){b=modular(b,twiddle,mod.p);expected[first+j]=(uint64_t(a)+b)%mod.p;expected[first+j+Distance]=(uint64_t(a)+mod.p-b)%mod.p;}
            else {expected[first+j]=(uint64_t(a)+b)%mod.p;expected[first+j+Distance]=modular((uint64_t(a)+mod.p-b)%mod.p,twiddle,mod.p);}
        }
    for(auto& v:expected)v=modular(v,R,mod.p);
    if constexpr(Distance<32){
        for(unsigned i=0;i<Len;++i){
            uint32_t v=before[i];
            if constexpr(Inverse){if(i&Distance){unsigned index=(i&(Distance-1))*(Len/(2*Distance));v=multiply_twiddle(v,roots[(Len-index)&(Len-1)],mod);}}
            exchange_values[i]=v;
        }
    }
    for(unsigned lane=0;lane<32;++lane){
        uint32_t v[Len/32];for(unsigned r=0;r<Len/32;++r)v[r]=before[lane+32*r];
        active_lane=lane;active_register=0;active_distance=Distance;
        step<Len,Distance,Inverse>(v,roots,mod,lane);
        assert(active_register==(Distance<32?Len/32:0));
        for(unsigned r=0;r<Len/32;++r)after[lane+32*r]=v[r];
    }
    assert(after==expected);values=after;
}
template<unsigned Len,bool Inverse,unsigned Distance=Inverse?1:Len/2>
void check_fft(std::vector<uint32_t>& values,const Twiddle* roots,const Mod& mod,uint32_t R,uint32_t inverse_R){
    check_step<Len,Distance,Inverse>(values,roots,mod,R,inverse_R);
    if constexpr(Inverse){if constexpr(Distance<Len/2)check_fft<Len,Inverse,Distance*2>(values,roots,mod,R,inverse_R);}
    else {if constexpr(Distance>1)check_fft<Len,Inverse,Distance/2>(values,roots,mod,R,inverse_R);}
}
template<unsigned Len>
void check_plan_fft(const Twiddle* roots,const Mod& mod,std::mt19937& random){
    uint32_t R=(uint64_t{1}<<32)%mod.p,inverse_R=bytecrt::rns_detail::pow_mod(R,mod.p-2,mod.p);
    for(unsigned pattern=0;pattern<3;++pattern){
        std::vector<uint32_t> values(Len);
        for(auto& v:values)v=modular(pattern==0?0:pattern==1?mod.p-1:random()%mod.p,R,mod.p);
        auto original=values;check_fft<Len,false>(values,roots,mod,R,inverse_R);
        check_fft<Len,true>(values,roots,mod,R,inverse_R);
        uint32_t inverse_len=bytecrt::rns_detail::pow_mod(Len,mod.p-2,mod.p);
        for(unsigned i=0;i<Len;++i)assert(modular(values[i],inverse_len,mod.p)==original[i]);
    }
}
int main(){
    std::mt19937 random(494178);unsigned prime_cases=0;uint64_t fixed_checks=0;
    for(unsigned N:{32768u,65536u}){
        unsigned W=N==65536?1747:868,L=(W+31)/32,K=N/256;
        uint32_t gap=N==65536?9699327:43646975;
        std::vector<uint32_t> q(L,UINT32_MAX);q[0]=uint32_t{0}-gap;q.back()=(uint32_t{1}<<(W-32*(L-1)))-1;
        auto plan=bytecrt::make_rns_plan(N,W,q);auto tables=make_tables(plan);
        for(unsigned prime=0;prime<plan.pcount;++prime){
            const Mod m=tables.mods[prime];uint32_t R=(uint64_t{1}<<32)%m.p;
            assert(uint32_t(uint64_t(m.p)*m.negative_inverse)==UINT32_MAX);
            const Twiddle* root=tables.roots.data()+size_t(prime)*(K+256);
            for(unsigned index=0;index<K+256;++index)
                for(uint32_t a:{0u,1u,m.p/2,m.p-1,uint32_t(random()%m.p)}){
                    assert(multiply_twiddle(a,root[index],m)==modular(a,root[index].value,m.p));++fixed_checks;
                }
            if(N==65536)check_plan_fft<256>(root,m,random);else check_plan_fft<128>(root,m,random);
            check_plan_fft<256>(root+K,m,random);++prime_cases;
            std::vector<uint32_t> powers(N);powers[0]=R;
            for(unsigned i=1;i<N;++i)powers[i]=modular(powers[i-1],plan.omega_roots[prime],m.p);
            for(unsigned id=0;id<256;++id)for(unsigned row=0;row<K;++row){
                unsigned exponent=id*reversed(row,K);
                assert(tables.forward_phase[size_t(prime)*N+id*K+row]==powers[exponent]);
                assert(tables.inverse_phase[size_t(prime)*N+row*256+id]==powers[(N-exponent)&(N-1)]);
            }
        }
        // Four-word padding makes the eight-row transpose bank-bijective.
        for(unsigned first=0;first<32;first+=4){
            std::array<bool,32> banks{};
            for(unsigned lane=0;lane<32;++lane){unsigned bank=((lane%8)*(K+4)+first+lane/8)%32;assert(!banks[bank]);banks[bank]=true;}
        }
    }
    std::cout<<"PASS "<<prime_cases<<" prime plans; actual warp-step source, both directions, all phase indexes and transpose banks; GPU executions=0; fixed-twiddle checks="<<fixed_checks<<"\n";
}
