#include "../recovery.hpp"
using namespace fermat_native;
extern "C" void recover(const Stored<48>* a,const Stored<40>* b,const Stored<32>* c,uint32_t* out) {
    recovery::output(recovery::reconstruct(*a,*b,*c),out);
}
