#include "fherma.h"
#include <gmp.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <thread>

namespace {
struct Integer {
    mpz_t value;
    Integer() { mpz_init(value); }
    ~Integer() { mpz_clear(value); }
    Integer(const Integer&) = delete;
};
struct Plan {
    unsigned batch, n, limbs;
    size_t slot;
    Integer modulus;
    explicit Plan(const fherma::Point& p) : batch(p.B), n(p.N), limbs(p.L) {
        if (!n || !limbs || p.q.data.size()!=limbs) throw std::invalid_argument("invalid point");
        mpz_import(modulus.value,limbs,-1,4,0,0,p.q.data.data());
        if (mpz_cmp_ui(modulus.value,2)<0) throw std::invalid_argument("invalid modulus");
        Integer limit;
        mpz_sub_ui(limit.value,modulus.value,1);
        mpz_mul(limit.value,limit.value,limit.value);
        mpz_mul_ui(limit.value,limit.value,n);
        slot=(mpz_sizeinbase(limit.value,2)+7)/8;
    }
};
struct Workspace {
    Integer a,b,product,low,high;
    std::vector<unsigned char> bytes;
    explicit Workspace(const Plan& p):bytes(2*p.n*p.slot) {}
    void pack(const Plan& p,const uint32_t* in,Integer& out) {
        auto count=p.n*p.slot;
        std::fill_n(bytes.begin(),count,0);
        // The judge's packed limbs and host are little endian.
        for(unsigned i=0;i<p.n;++i)
            std::memcpy(bytes.data()+i*p.slot,in+size_t(i)*p.limbs,
                        std::min(p.slot,size_t(p.limbs)*4));
        mpz_import(out.value,count,-1,1,0,0,bytes.data());
    }
    void multiply(const Plan& p,const uint32_t* x,const uint32_t* y,uint32_t* out) {
        pack(p,x,a); pack(p,y,b);
        mpz_mul(product.value,a.value,b.value);
        std::fill(bytes.begin(),bytes.end(),0);
        mpz_export(bytes.data(),nullptr,-1,1,0,0,product.value);
        size_t half=p.n*p.slot;
        for(unsigned i=0;i<p.n;++i) {
            auto* row=bytes.data()+i*p.slot;
            mpz_import(low.value,p.slot,-1,1,0,0,row);
            mpz_import(high.value,p.slot,-1,1,0,0,row+half);
            mpz_sub(low.value,low.value,high.value);
            mpz_mod(low.value,low.value,p.modulus.value);
            mpz_export(out+size_t(i)*p.limbs,nullptr,-1,4,0,0,low.value);
        }
    }
};
}
void* fherma_init(const fherma::Point& p){ return new Plan(p); }
void fherma_free(void* state){ delete static_cast<Plan*>(state); }
fherma::Outputs fherma_run(void* state,const fherma::Inputs& in) {
    const auto& p=*static_cast<Plan*>(state);
    size_t stride=size_t(p.n)*p.limbs,count=stride*p.batch;
    if(in.a.data.size()!=count || in.b.data.size()!=count) throw std::invalid_argument("input size");
    fherma::Outputs out;
    out.c.shape={p.batch,p.n,p.limbs}; out.c.data.resize(count);
    std::atomic<unsigned> next{0};
    auto work=[&] {
        Workspace w(p);
        for(unsigned i=next++; i<p.batch; i=next++) {
            size_t offset=i*stride;
            w.multiply(p,in.a.data.data()+offset,in.b.data.data()+offset,out.c.data.data()+offset);
        }
    };
    unsigned threads=std::min({p.batch,8u,std::max(1u,std::thread::hardware_concurrency())});
    std::vector<std::thread> workers;
    for(unsigned i=1;i<threads;++i) workers.emplace_back(work);
    work();
    for(auto& t:workers)t.join();
    return out;
}
