#include "../timeline.hpp"

#include <cassert>
#include <map>

int main() {
    using stages_profile::Product;
    const std::vector<Product> stamps{
        {0,2,2,4,4,7},
        {2,5,5,6,7,10},
        {5,9,9,11,11,13},
    };
    const auto measured=stages_profile::summarize(stamps,13);
    const std::map<std::string,double> values(measured.begin(),measured.end());
    assert(values.at("batch_encode_forward_ms")==9);
    assert(values.at("batch_middle_ms")==5);
    assert(values.at("batch_inverse_crt_ms")==8);
    assert(values.at("alone_encode_forward_ms")==2);
    assert(values.at("alone_middle_ms")==1);
    assert(values.at("alone_inverse_crt_ms")==2);
    assert(values.at("zero_active_ms")==0);
    assert(values.at("two_active_ms")==7);
    assert(values.at("three_active_ms")==1);

    // The first product still reads its partials through time 7. The
    // third product must wait even when its forward transform ends first.
    auto overwrite=stamps;
    overwrite[1][1]=4.75;
    overwrite[2]={4.75,6.25,6.25,11,11,13};
    try {
        stages_profile::summarize(overwrite,13);
        assert(false);
    } catch(const std::invalid_argument&) {}
}
