#include <array>
#include <barrier>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>
#define __device__
#define __host__
#define __forceinline__ inline
static std::barrier<> fence(32);
static std::array<uint32_t,32> slots;
static thread_local unsigned thread_lane;
static uint32_t __shfl_xor_sync(uint32_t mask,uint32_t value,unsigned dist,unsigned width) {
    if(mask!=UINT32_MAX || width!=8 || dist>=width)std::abort();
    slots[thread_lane]=value;fence.arrive_and_wait();auto x=slots[thread_lane^dist];fence.arrive_and_wait();return x;
}
#include "../joint-terms.cuh"
#include "../warp-carry.cuh"
struct CarryOps {
    uint32_t ballot(bool b)const {
        slots[thread_lane]=b;fence.arrive_and_wait();uint32_t m=0;
        for(unsigned i=0;i<32;++i)m |= slots[i]<<i;
        fence.arrive_and_wait();return m;
    }
    uint32_t get(uint32_t x,unsigned lane)const {
        slots[thread_lane]=x;fence.arrive_and_wait();uint32_t v=slots[lane];fence.arrive_and_wait();return v;
    }
};
int main() {
    using namespace folded;
    static const auto w=make_joint_weights(host_plan);
    Mod mods[16];
    for(unsigned j=0;j<16;++j)mods[j]=bytecrt::warp_ntt::make_mod(host_plan.primes[j].p,1);
    unsigned count;std::cin>>count;
    if(count%4)std::abort();
    struct Case {std::array<uint32_t,136> terms{};std::array<uint32_t,55> lo{},hi{},out{};};
    std::vector<Case> cases(count);
    for(auto& c:cases)for(unsigned j=0;j<128;++j)std::cin>>c.terms[j];
    std::vector<std::thread> lanes;
    for(unsigned l=0;l<32;++l)lanes.emplace_back([&,l] {
        thread_lane=l;
        for(unsigned start=0;start<count;start+=4) {
            joint_terms_for_digit(cases[start+l/8].terms.data(),l%8,&host_plan,mods,&w);
            fence.arrive_and_wait();
            for(unsigned r=start;r<start+4;++r) {
                auto& c=cases[r];
                for(unsigned limb=l;limb<55;limb+=32) {
                    uint64_t word=0;
                    for(unsigned d=0;d<4;++d) {
                        uint32_t acc=0;
                        for(unsigned k=0;k<joint_pad;++k)acc+=((c.terms[k/4]>>(8*(k%4)))&255)*w.matrix[4*limb+d][k];
                        word+=uint64_t(acc)<<(8*d);
                    }
                    c.lo[limb]=uint32_t(word);c.hi[limb]=uint32_t(word>>32);
                }
                fence.arrive_and_wait();
                bytecrt::warp_carry::finish<W,L>(c.lo.data(),c.hi.data(),gap,c.out.data(),l,CarryOps{});
            }
        }
    });
    for(auto& l:lanes)l.join();
    for(auto& c:cases) {
        for(auto x:c.terms)std::cout<<x<<' ';
        for(auto x:c.out)std::cout<<x<<' ';
        std::cout<<'\n';
    }
}
