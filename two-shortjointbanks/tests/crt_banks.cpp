// SPDX-License-Identifier: Apache-2.0
#include "../crt-shared-layout.hpp"

#include <array>
#include <cassert>
#include <iostream>

using bytecrt::crt_layout::word;

template<unsigned RowWords,unsigned Count>
void check() {
    constexpr unsigned stride=RowWords+4;
    const auto bank=[](unsigned row,unsigned j) {
        return (row*stride+word(row,j))%32;
    };
    for(unsigned row=0;row<32;++row) {
        std::array<bool,RowWords> written{};
        for(unsigned j=0;j<RowWords;++j) {
            const unsigned destination=word(row,j);
            assert(destination<RowWords && !written[destination]);
            assert(word(row,destination)==j);
            written[destination]=true;
        }
        for(unsigned begin:{0u,Count}) {
            for(unsigned first=0;first<Count;first+=32) {
                std::array<bool,32> used{};
                for(unsigned j=first;j<Count && j<first+32;++j) {
                    const unsigned index=bank(row,begin+j);
                    assert(!used[index]);
                    used[index]=true;
                }
            }
        }
    }
    for(unsigned j=0;j<RowWords;++j) {
        std::array<unsigned,32> before{},after{};
        for(unsigned row=0;row<32;++row) {
            ++before[(row*stride+j)%32];
            ++after[bank(row,j)];
        }
        unsigned four_way=0;
        for(unsigned n:before)four_way+=n==4;
        assert(four_way==8);
        for(unsigned n:after)assert(n==1);
    }
    for(unsigned row_group=0;row_group<32;row_group+=8) {
        for(unsigned first=0;first<RowWords;first+=4) {
            std::array<bool,32> used{};
            for(unsigned lane=0;lane<32;++lane) {
                const unsigned index=bank(row_group+lane/4,first+lane%4);
                assert(!used[index]);
                used[index]=true;
            }
        }
    }
    std::cout<<"PASS "<<32*RowWords
             <<" coefficient words, all CRT tensor operands and both alpha reductions\n";
}
int main() {
    check<120,58>();
    check<64,30>();
}
