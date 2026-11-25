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
        slot=(mpz_sizeinbase(limit.value,2)+GMP_NUMB_BITS-1)/GMP_NUMB_BITS;
    }
};
struct Workspace {
    Integer a,b,product,low,high;
    explicit Workspace(const Plan&) {}
    // Populate GMP's native destination directly. A slot is a whole number of
    // limbs, so multiplying cannot carry between polynomial coefficients.
    void pack(const Plan& p,const uint32_t* in,Integer& out) {
        static_assert(GMP_NAIL_BITS==0 && sizeof(mp_limb_t)%4==0);
        const size_t count=p.n*p.slot;
        auto* dest=mpz_limbs_write(out.value,count);
        std::fill_n(dest,count,mp_limb_t(0));
        for(unsigned i=0;i<p.n;++i)
            std::memcpy(dest+i*p.slot,in+size_t(i)*p.limbs,
                        std::min(p.slot*sizeof(mp_limb_t),size_t(p.limbs)*4));
        mpz_limbs_finish(out.value,count);
    }
    void multiply(const Plan& p,const uint32_t* x,const uint32_t* y,uint32_t* out) {
        pack(p,x,a); pack(p,y,b);
        mpz_mul(product.value,a.value,b.value);
        const auto* data=mpz_limbs_read(product.value);
        const size_t used=mpz_size(product.value);
        auto read=[&](Integer& dest,size_t offset) {
            if(offset>=used) {mpz_set_ui(dest.value,0);return;}
            mpz_import(dest.value,std::min(p.slot,used-offset),-1,sizeof(mp_limb_t),0,0,data+offset);
        };
        for(unsigned i=0;i<p.n;++i) {
            read(low,i*p.slot);
            read(high,(p.n+i)*p.slot);
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
