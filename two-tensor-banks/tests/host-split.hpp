#pragma once
#include "../two-digit-word.hpp"
#include <array>
#include <cassert>

template<unsigned W,unsigned L,unsigned Width,unsigned Split>
auto split_row(const uint32_t* raw) {
    std::array<two_digit::WordPair,Width> loaded{},result{};
    unsigned reads[L]{};
    for(unsigned lane=0;lane<Width;++lane)
        loaded[lane]=two_digit::load_words<W,L,Width>(lane,[&](unsigned k){++reads[k];return raw[k];});
    for(auto n:reads)assert(n==1);
    for(unsigned lane=0;lane<Width;++lane)
        result[lane]=two_digit::split_words<Split,Width>(loaded[lane],lane,[&](uint32_t self,unsigned src){
            auto send=[&](unsigned i){return i<Split/4?loaded[i].second:loaded[i].first;};
            assert(self==send(lane));
            return send(src);
        });
    return result;
}
