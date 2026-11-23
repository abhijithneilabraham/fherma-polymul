// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace stages_profile {
using Product = std::array<double,10>;
using Metrics = std::vector<std::pair<std::string,double>>;
inline Metrics summarize(const std::vector<Product>& products,double end) {
    if(products.empty() || !std::isfinite(end) || end<=0)
        throw std::invalid_argument("invalid batch span");
    std::vector<std::pair<double,unsigned>> events;
    std::array<double,5> inclusive{},alone{};
    Product prior{};
    for(const auto& t:products) {
        if(!std::all_of(t.begin(),t.end(),[](double x){return std::isfinite(x);}) ||
           !std::is_sorted(t.begin(),t.end()) || t[9]>end ||
           t[0]<prior[3] || t[2]<prior[5] || t[4]<prior[7] || t[6]<prior[9])
            throw std::invalid_argument("inconsistent stage ordering");
        for(unsigned stage=0;stage<5;++stage) {
            inclusive[stage]+=t[2*stage+1]-t[2*stage];
            events.emplace_back(t[2*stage],1u<<stage);
            events.emplace_back(t[2*stage+1],1u<<stage);
        }
        prior=t;
    }
    // Each stage issues on one stream. Each endpoint toggles its active bit.
    // Coincident endpoints have zero duration between them.
    std::sort(events.begin(),events.end());
    events.emplace_back(end,0);
    std::array<double,4> concurrency{};
    unsigned mask=0;
    double previous=0;
    for(const auto& [time,bit]:events) {
        unsigned active=__builtin_popcount(mask);
        double duration=time-previous;
        if(duration>0) {
            if(active>3)throw std::invalid_argument("more than three active streams");
            concurrency[active]+=duration;
            if(active==1)alone[__builtin_ctz(mask)]+=duration;
        }
        mask^=bit;previous=time;
    }
    constexpr const char* names[]={"encode","forward","middle","inverse","crt"};
    Metrics result;
    for(unsigned stage=0;stage<5;++stage) {
        result.emplace_back(std::string("batch_")+names[stage]+"_ms",inclusive[stage]);
        result.emplace_back(std::string("alone_")+names[stage]+"_ms",alone[stage]);
    }
    result.emplace_back("zero_active_ms",concurrency[0]);
    result.emplace_back("two_active_ms",concurrency[2]);
    result.emplace_back("three_active_ms",concurrency[3]);
    result.emplace_back("batch_compute_ms",end);
    return result;
}
}
