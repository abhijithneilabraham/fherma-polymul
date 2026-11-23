#include "../cluster-layout.hpp"
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
using namespace bytecrt::warp_ntt;

static void check(unsigned K) {
    const unsigned N = 256*K;
    std::array<std::vector<unsigned>,cluster_blocks> tiles;
    for(auto& tile:tiles) tile.assign(2*K*32,UINT32_MAX);
    for(unsigned rank=0;rank<8;++rank) for(unsigned warp=0;warp<32;++warp)
        for(unsigned r=0;r<K/32;++r) {
            uint32_t banks=0;
            for(unsigned lane=0;lane<32;++lane) {
                unsigned row=32*r+lane, id=32*rank+warp;
                unsigned pos=cluster_slot(row,warp);
                banks |= 1u<<(pos%32);
                for(unsigned operand=0;operand<2;++operand) {
                    unsigned& x=tiles[rank].at(operand*K*32+pos);
                    assert(x==UINT32_MAX);
                    x=operand*N+id*K+row;
                }
            }
            assert(banks==UINT32_MAX);
        }
    for(unsigned rank=0;rank<8;++rank) for(unsigned warp=0;warp<K/8;++warp) {
        unsigned row=rank*(K/8)+warp;
        for(unsigned peer=0;peer<8;++peer) {
            uint32_t banks=0;
            for(unsigned lane=0;lane<32;++lane) {
                unsigned id=32*peer+lane, pos=cluster_slot(row,lane);
                assert(pos/32==row);
                banks |= 1u<<(pos%32);
                assert(tiles[peer].at(pos)==id*K+row);
                assert(tiles[peer].at(K*32+pos)==N+id*K+row);
                tiles[peer][pos]=2*N+row*256+id;
            }
            assert(banks==UINT32_MAX);
        }
    }
    for(unsigned rank=0;rank<8;++rank) for(unsigned warp=0;warp<32;++warp)
        for(unsigned row=0;row<K;++row)
            assert(tiles[rank].at(cluster_slot(row,warp))==2*N+row*256+rank*32+warp);
    std::cout<<"PASS complete cluster tile N="<<N<<"\n";
}
int main() { check(128); check(256); }
