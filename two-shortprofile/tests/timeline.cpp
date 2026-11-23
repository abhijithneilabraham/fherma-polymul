#include "../timeline.hpp"

#include <cassert>
#include <map>

int main() {
    using stages_profile::Product;
    const std::vector<Product> stamps{
        {0,2,2,4,4,5,5,7},
        {2,4,5,6,7,8,8,10},
        {4,9,9,10,10,11,11,13},
    };
    auto measured=stages_profile::summarize(stamps,13);
    const std::map<std::string,double> values(measured.begin(),measured.end());
    assert(values.at("batch_encode_forward_ms")==9);
    assert(values.at("batch_middle_ms")==4);
    assert(values.at("batch_inverse_ms")==3);
    assert(values.at("batch_crt_ms")==6);
    assert(values.at("alone_encode_forward_ms")==2);
    assert(values.at("alone_middle_ms")==0);
    assert(values.at("alone_inverse_ms")==1);
    assert(values.at("alone_crt_ms")==2);
    assert(values.at("zero_active_ms")==0);
    assert(values.at("two_active_ms")==7);
    assert(values.at("three_active_ms")==1);
    auto wrong=stamps;
    wrong[2][0]=3.75;
    try {
        stages_profile::summarize(wrong,13);
        assert(false);
    } catch(const std::invalid_argument&) {}
}
