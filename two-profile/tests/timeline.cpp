#include "../timeline.hpp"
#include <cassert>
#include <map>
int main(){
    using namespace stages_profile;
    std::vector<Product> t{{0,1,1,2,2,3,3,4,4,10},
                           {2,5,5,6,6,9,10,11,11,13},
                           {6,8,9,10,11,12,13,14,14,15}};
    auto result=summarize(t,16);
    std::map<std::string,double> m(result.begin(),result.end());
    assert(m.at("zero_active_ms")==1 && m.at("three_active_ms")==2);
    assert(m.at("two_active_ms")==7 && m.at("alone_crt_ms")==2);
    assert(m.at("alone_encode_ms")==1 && m.at("alone_forward_ms")==1);
    assert(m.at("alone_inverse_ms")==2 && m.at("alone_middle_ms")==0);
    assert(m.at("batch_crt_ms")==9);
    t[2][2]=8.5;
    try { summarize(t,16); } catch(const std::invalid_argument&) { return 0; }
    return 1;
}
