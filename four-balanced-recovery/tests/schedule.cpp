#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include "../four-plan.hpp"
#include "../sparse-crt.hpp"
#include "../decode-schedule.hpp"

void check(unsigned w, unsigned n, unsigned gap) {
    unsigned limbs=(w+31)/32, rows=7*((limbs+13)/14);
    std::vector<uint32_t> q(limbs,UINT32_MAX);
    q.front()=0u-gap;
    q.back()=(1u<<(w%32))-1;
    auto plan=four_plan::make(n,w,q);
    auto packed=sparse_crt::pack(plan.crt,limbs,plan.transform.pcount);
    std::vector<uint32_t> offsets(rows+1), uses(rows);
    std::memcpy(offsets.data(),packed.data(),offsets.size()*sizeof(uint32_t));
    std::array<unsigned,8> original{}, updated{};
    for(unsigned i=0;i<rows;++i) original[i%7]+=offsets[i+1]-offsets[i];
    for(unsigned warp=0;warp<8;++warp) {
        uint32_t sequence=w==1747 ? four_decode::ranked_slots[warp] : four_decode::auxiliary_slots[warp];
        for(;sequence;sequence>>=5) {
            unsigned index=(sequence&31u)-1;
            assert(index<rows);
            ++uses[index];
            updated[warp]+=offsets[index+1]-offsets[index];
        }
    }
    for(auto times:uses) assert(times==1);
    if(w==1747) for(auto pairs:updated) assert(pairs*8==offsets.back());
    unsigned before=*std::max_element(original.begin(),original.end());
    unsigned after=*std::max_element(updated.begin(),updated.end());
    assert(before==(w==1747?48:14) && after==(w==1747?32:12));
    std::cout<<"PASS W="<<w<<", "<<offsets.back()<<" matrix pairs, longest warp "<<before<<" -> "<<after<<"\n";
}
int main(){check(868,32768,43646975);check(1747,65536,9699327);}
