#include "../timeline.hpp"
#include <cassert>
#include <map>

int main() {
    using Times = std::vector<std::array<double,8>>;
    Times times{{0,2,3,4,4,5,6,13}, {5,8,10,11,13,14,14,16}};
    auto results = overlap_profile::summarize(times, 17);
    std::map<std::string,double> m(results.begin(),results.end());
    assert(m.at("crt_with_encode_ms")==2 && m.at("crt_with_forward_ms")==2);
    assert(m.at("crt_with_middle_ms")==1 && m.at("crt_with_inverse_ms")==0);
    assert(m.at("batch_crt_ms")==9 && m.at("batch_uncovered_ms")==1);
    times[1][4]=12;
    try { overlap_profile::summarize(times,17); }
    catch(const std::invalid_argument&) { return 0; }
    return 1;
}
