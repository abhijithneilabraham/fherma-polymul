#include "../two-plan.hpp"
#include "host-split.hpp"
#include <fstream>
#include <iostream>
template<class V>
void write(const char* name,const V& v){
    std::ofstream f(name,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(v[0]));
    if(!f) throw std::runtime_error("write failed");
}
int main(int argc,char** argv) {
    unsigned n=std::stoul(argv[1]),w=n==65536?1747:868,L=(w+31)/32,gap=n==65536?9699327:43646975;
    std::vector<uint32_t> q(L,UINT32_MAX);q.front()=0u-gap;q.back()=(1u<<(w%32))-1;
    auto plan=two_digit::make(n,w,q);
    write("primes",plan.primes);write("theta",plan.theta);write("inverse_theta",plan.inverse_theta);
    write("crt_inverse",plan.transform.crt_inverse);write("encode",plan.encode);write("decode",plan.decode);write("crt_bias",plan.crt_bias);
    unsigned cases;std::cin>>cases;
    for(unsigned row=0;row<cases;++row) {
        std::vector<uint32_t> raw(L);for(auto& x:raw)std::cin>>x;
        auto write_digits=[](const auto& values) {
            for(const auto& v:values)std::cout<<v.first<<' ';
            for(const auto& v:values)std::cout<<v.second<<' ';
        };
        if(n==65536)write_digits(split_row<1747,55,32,110>(raw.data()));
        else write_digits(split_row<868,28,16,55>(raw.data()));
        std::cout<<'\n';
    }
}
