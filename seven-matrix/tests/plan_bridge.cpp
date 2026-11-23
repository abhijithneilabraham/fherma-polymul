#include "../seven-plan.hpp"
#include <fstream>
#include "../sparse-crt.hpp"
template<class T> void save(const char* path,const std::vector<T>& v) {
    std::ofstream f(path,std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));
    if(!f)throw std::runtime_error("plan output");
}
int main(int argc,char** argv){
    unsigned w=std::stoul(argv[1]),n=w==1747?65536:32768,gap=w==1747?9699327:43646975;
    std::vector<uint32_t> q((w+31)/32,UINT32_MAX);
    q.front()=0u-gap;q.back()=(1u<<(w%32))-1;
    auto plan=seven_plan::make(n,w,q);
    save("packed",sparse_crt::pack(plan.crt,q.size(),plan.transform.pcount));
    save("crt",plan.crt);save("encode",plan.matrix_words);save("crt_inverse",plan.transform.crt_inverse);
    save("reciprocal48",plan.reciprocal48);save("primes",plan.transform.primes);
    std::vector<uint32_t> twists,inverse_tiles;
    for(auto& f:plan.encode_fields)for(auto t:f.digit_twist)twists.push_back(t.value);
    for(auto& f:plan.inverse_fields)for(auto& tile:f.tiles)
        inverse_tiles.insert(inverse_tiles.end(),std::begin(tile),std::end(tile));
    save("twists",twists);save("inverse_tiles",inverse_tiles);
}
