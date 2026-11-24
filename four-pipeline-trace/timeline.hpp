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
using Product = std::array<double,6>;
using Metrics = std::vector<std::pair<std::string,double>>;
inline Metrics summarize(const std::vector<Product>& products,double end) {
    if(products.empty() || !std::isfinite(end) || end<=0)
        throw std::invalid_argument("invalid batch span");
    std::vector<std::pair<double,unsigned>> events;
    std::array<double,3> inclusive{},alone{};
    double forward_slot_wait=0, middle_slot_wait=0, middle_input_wait=0, output_input_wait=0;
    for(std::size_t index=0;index<products.size();++index) {
        const auto& t=products[index];
        const Product prior=index?products[index-1]:Product{};
        if(!std::all_of(t.begin(),t.end(),[](double x){return std::isfinite(x);}) ||
           !std::is_sorted(t.begin(),t.end()) || t[5]>end ||
           t[0]<prior[1] || t[2]<prior[3] || t[4]<prior[5] ||
           (index>=2 && (t[0]<products[index-2][3] || t[2]<products[index-2][5])))
            throw std::invalid_argument("inconsistent stage ordering");
        if(index) {
            const double forward_release=index>=2?products[index-2][3]:0;
            const double partial_release=index>=2?products[index-2][5]:0;
            // Attribute a wait only where one dependency is the last
            // condition keeping an otherwise free stage from being ready.
            forward_slot_wait+=std::max(0.0,forward_release-prior[1]);
            middle_slot_wait+=std::max(0.0,partial_release-std::max(prior[3],t[1]));
            middle_input_wait+=std::max(0.0,t[1]-std::max(prior[3],partial_release));
            output_input_wait+=std::max(0.0,t[3]-prior[5]);
        }
        for(unsigned stage=0;stage<3;++stage) {
            inclusive[stage]+=t[2*stage+1]-t[2*stage];
            events.emplace_back(t[2*stage],1u<<stage);
            events.emplace_back(t[2*stage+1],1u<<stage);
        }
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
    constexpr const char* names[]={"encode_forward","middle","inverse_crt"};
    Metrics result;
    for(unsigned stage=0;stage<3;++stage) {
        result.emplace_back(std::string("batch_")+names[stage]+"_ms",inclusive[stage]);
        result.emplace_back(std::string("alone_")+names[stage]+"_ms",alone[stage]);
    }
    result.emplace_back("zero_active_ms",concurrency[0]);
    result.emplace_back("two_active_ms",concurrency[2]);
    result.emplace_back("three_active_ms",concurrency[3]);
    result.emplace_back("forward_slot_wait_ms",forward_slot_wait);
    result.emplace_back("middle_slot_wait_ms",middle_slot_wait);
    result.emplace_back("middle_input_wait_ms",middle_input_wait);
    result.emplace_back("output_input_wait_ms",output_input_wait);
    result.emplace_back("batch_compute_ms",end);
    return result;
}
}
