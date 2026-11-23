// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace sparse_crt {
inline std::vector<uint8_t> pack(const std::vector<uint8_t>& dense,
                                 unsigned limbs,unsigned primes) {
    const unsigned planes=4*primes;
    const unsigned stride=32*((planes+4+7)/8);
    const unsigned words=(4*planes+4+3)/4;
    const unsigned rows=7*((limbs+13)/14);
    const unsigned header=128*((4*(rows+1)+127)/128);
    const unsigned weight_start=128*((header+8*rows*((words+7)/8)+127)/128);
    if(words>256 || primes>=256 || dense.size()!=size_t(4*limbs)*stride)
        throw std::invalid_argument("invalid dense CRT matrix");
    std::vector<uint8_t> result(weight_start);
    unsigned cursor=0;
    auto offset=[&](unsigned row){
        for(unsigned b=0;b<4;++b)result[4*row+b]=cursor>>(8*b);
    };
    auto byte=[&](unsigned col,unsigned k)->uint8_t {
        if(col>=4*limbs || k>=4*planes+4)return 0;
        unsigned original=k<4*planes?k:4*planes+4*(k-4*planes);
        return dense[col*stride+original];
    };
    for(unsigned row=0;row<rows;++row){
        offset(row);
        std::vector<unsigned> live;
        for(unsigned word=0;word<words;++word){
            bool nonzero=false;
            for(unsigned col=8*row;col<8*row+8;++col)
                for(unsigned b=0;b<4;++b)nonzero|=byte(col,4*word+b)!=0;
            if(nonzero)live.push_back(word);
        }
        for(unsigned pos=0;pos<live.size();pos+=8){
            std::array<uint8_t,256> tile{};
            for(unsigned slot=0;slot<8 && pos+slot<live.size();++slot){
                unsigned word=live[pos+slot];
                result[header+8*cursor+slot]=word;
                for(unsigned col=0;col<8;++col)
                    for(unsigned b=0;b<4;++b){
                        unsigned lane=4*col+slot%4;
                        tile[128*(slot/4)+4*lane+b]=byte(8*row+col,4*word+b);
                    }
            }
            result.insert(result.end(),tile.begin(),tile.end());
            ++cursor;
        }
    }
    offset(rows);
    return result;
}
}
