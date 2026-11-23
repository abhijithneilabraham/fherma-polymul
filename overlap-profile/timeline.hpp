// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace overlap_profile {
using Metrics = std::vector<std::pair<std::string, double>>;
inline Metrics summarize(const std::vector<std::array<double, 8>>& products, double end) {
    if (products.empty() || !std::isfinite(end) || end < 0)
        throw std::invalid_argument("invalid batch span");
    constexpr const char* stages[] = {"encode", "forward", "middle", "inverse", "crt"};
    constexpr unsigned starts[] = {0, 1, 2, 4, 6};
    std::array<double, 5> sums{};
    std::array<double, 4> intersections{};
    double last_inverse = 0, last_crt = 0;
    for (const auto& t : products) {
        // Stream order and the release of the previous weighted buffer.
        if (!std::all_of(t.begin(), t.end(), [](double x){ return std::isfinite(x); }) ||
            !std::is_sorted(t.begin(), t.end()) || t[0] < last_inverse ||
            t[4] < last_crt || t[7] > end)
            throw std::invalid_argument("inconsistent product timestamps");
        last_inverse = t[5];
        last_crt = t[7];
        for (unsigned k = 0; k < sums.size(); ++k)
            sums[k] += t[starts[k]+1] - t[starts[k]];
        for (const auto& other : products) {
            for (unsigned k = 0; k < intersections.size(); ++k) {
                unsigned i = starts[k];
                intersections[k] += std::max(0.0, std::min(t[i+1],other[7]) -
                                                   std::max(t[i],other[6]));
            }
        }
    }
    Metrics result;
    double covered = 0;
    for (unsigned k = 0; k < sums.size(); ++k) {
        result.emplace_back(std::string("batch_")+stages[k]+"_ms", sums[k]);
        covered += sums[k];
    }
    for (unsigned k = 0; k < intersections.size(); ++k) {
        result.emplace_back(std::string("crt_with_")+stages[k]+"_ms", intersections[k]);
        covered -= intersections[k];
    }
    result.emplace_back("batch_uncovered_ms", end-covered);
    result.emplace_back("batch_compute_ms", end);
    return result;
}
}
