// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace sparse_crt {
inline unsigned header_words(unsigned tile_rows) { return (2*tile_rows+31)/32*32; }

// Each warp computes eight output bytes. A live tile carries two contiguous
// groups of 32 words, in the exact lane order of the tensor instruction.
inline std::vector<uint8_t> pack(const std::vector<uint8_t>& dense,
                                 unsigned limbs,unsigned primes) {
    const unsigned stride=32*((7*primes+14)/8);
    const unsigned rows=7*((limbs+13)/14);
    if(stride/32>32 || dense.size()!=size_t(4*limbs)*stride)
        throw std::invalid_argument("invalid dense CRT matrix");
    std::vector<uint8_t> result(4*header_words(rows));
    unsigned cursor=0;
    auto put_word=[&](unsigned index,uint32_t value){
        for(unsigned byte=0;byte<4;++byte)result[4*index+byte]=value>>(8*byte);
    };
    for(unsigned row=0;row<rows;++row) {
        uint32_t mask=0;
        put_word(2*row+1,cursor);
        for(unsigned k=0;k<stride;k+=32){
            uint8_t tile[256]{};
            for(unsigned lane=0;lane<32;++lane){
                unsigned col=8*row+lane/4;
                if(col>=4*limbs)continue;
                for(unsigned half=0;half<2;++half)
                    for(unsigned b=0;b<4;++b)
                        tile[128*half+4*lane+b]=dense[col*stride+k+16*half+4*(lane%4)+b];
            }
            if(std::any_of(tile,tile+256,[](uint8_t x){return x!=0;})){
                mask|=uint32_t{1}<<(k/32);
                result.insert(result.end(),tile,tile+256);
                ++cursor;
            }
        }
        put_word(2*row,mask);
    }
    return result;
}
}
