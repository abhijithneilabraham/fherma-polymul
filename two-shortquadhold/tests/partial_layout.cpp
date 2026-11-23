// SPDX-License-Identifier: Apache-2.0
#include <cassert>
#include <iostream>
#include <set>
#include <vector>

constexpr unsigned rows=32, sector_words=8, warp_size=32;

unsigned address(unsigned column,unsigned row) {
    return (column/4)*rows*4+4*row+column%4;
}

template<unsigned N>
void check() {
    constexpr unsigned columns=N/rows;
    std::vector<bool> written(N);
    for(unsigned first=0;first<rows;first+=2) {
        for(unsigned begin=0;begin<columns;begin+=256) {
            for(unsigned warp=0;warp<16;++warp) {
                std::set<unsigned> sectors;
                for(unsigned lane=0;lane<warp_size;++lane) {
                    const unsigned tid=warp*warp_size+lane;
                    const unsigned out=address(begin+4*(tid/8)+tid%4,first+(tid/4)%2);
                    assert(out<N && !written[out]);
                    written[out]=true;
                    sectors.insert(out/sector_words);
                }
                assert(sectors.size()==4);
            }
        }
    }
    for(bool seen:written)assert(seen);
    for(unsigned first=0;first<columns;first+=4) {
        for(unsigned part=0;part<4;++part) {
            std::set<unsigned> sectors;
            for(unsigned lane=0;lane<warp_size;++lane) {
                const unsigned out=address(first+lane%4,lane/4+8*part);
                assert(out==first*rows+part*warp_size+lane);
                sectors.insert(out/sector_words);
            }
            assert(sectors.size()==4);
        }
    }
    std::cout<<"PASS "<<N<<" partial words with full sectors for each middle store and inverse load\n";
}

int main() {
    check<32768>();
    check<65536>();
}
