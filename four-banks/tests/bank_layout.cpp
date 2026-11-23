#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <set>
#include "../tile-layout.hpp"

using Counts = std::array<unsigned,5>;

// Count distinct 32-bit words requested at any one bank by a warp instruction.
template<class Address>
unsigned depth(Address address) {
    std::array<std::set<int>,32> banks;
    for (unsigned lane=0; lane<32; ++lane) {
        int word=address(lane);
        if (word>=0) banks[word%32].insert(word);
    }
    size_t maximum=0;
    for (const auto& bank : banks) maximum=std::max(maximum,bank.size());
    return unsigned(maximum);
}

template<unsigned W,bool Swizzled>
Counts audit() {
    constexpr unsigned words=(W+31)/32;
    constexpr unsigned raw_pitch=Swizzled ? four_encode::raw_stride<W> : words+1;
    constexpr unsigned tile_pitch=Swizzled ? 32 : 33;
    constexpr unsigned planes=(W==1747?3:2)*8;
    Counts counts{};
    auto raw=[](unsigned entry,unsigned limb){
        return entry*raw_pitch+(Swizzled ? four_encode::raw_limb<W>(entry%32,limb) : limb);
    };
    auto tile=[](unsigned plane,unsigned col,unsigned row){
        return (plane*4+col)*tile_pitch+(Swizzled ? four_encode::tile_row(plane,col,row) : row);
    };
    for(unsigned first=0;first<128*raw_pitch;first+=32)
        counts[0]=std::max(counts[0],depth([&](unsigned lane){unsigned i=first+lane;return raw(i/raw_pitch,i%raw_pitch);}));
    for(unsigned half=0;half<2;++half) for(unsigned highrow:{0u,8u})
        for(unsigned digit=0;digit<4;++digit) for(unsigned offset=0;offset<(W==1747?16:8);offset+=4){
            unsigned lo=(digit*W+3)/4,hi=((digit+1)*W+3)/4;
            counts[1]=std::max(counts[1],depth([&](unsigned lane){
                unsigned limb=lo/32+offset+lane%4;
                return 32*limb<hi ? int(raw(16*half+highrow+lane/4,limb)) : -1;
            }));
        }
    for(unsigned pair=0;pair<planes;pair+=2) for(unsigned col=0;col<4;++col) {
        for(unsigned row=0;row<32;row+=8)
            counts[2]=std::max(counts[2],depth([&](unsigned l){return l%2 ? -1 : int(tile(pair+l%4/2,col,row+l/4));}));
        for(unsigned p=pair;p<pair+2;++p)
            counts[3]=std::max(counts[3],depth([&](unsigned l){return tile(p,col,l);}));
    }
    for(unsigned p=0;p<planes;++p) for(unsigned row=0;row<32;row+=8)
        counts[4]=std::max(counts[4],depth([&](unsigned l){return tile(p,l%4,row+l/4);}));
    return counts;
}
template<unsigned W>
void check() {
    auto parent=audit<W,false>(), candidate=audit<W,true>();
    std::cout<<"W="<<W<<" raw store, fragment load, digit store, FFT access, output gather:";
    for(unsigned i=0;i<parent.size();++i) {
        std::cout<<" "<<parent[i]<<"->"<<candidate[i];
        assert(candidate[i]==1);
    }
    assert(parent[1]>1 && parent[2]>1 && parent[4]==4);
    std::cout<<"\n";
}
int main() {check<868>();check<1747>();}
