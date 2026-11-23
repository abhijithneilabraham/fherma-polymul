// Execute the actual warp butterfly code with a synchronous CPU shuffle model.
#include <array>
#include <cassert>
#include <iostream>
#include <random>
#include "../warp-tables.hpp"

static std::array<uint32_t,256> sent;
static unsigned lane, slot, active_distance;
static uint32_t __shfl_xor_sync(unsigned mask, uint32_t value, unsigned distance) {
    assert(mask == 0xffffffff && distance == active_distance);
    assert(sent[32*slot+lane] == value);
    return sent[32*slot++ + (lane^distance)];
}
#define __device__
#define __forceinline__ inline
#include "../warp-fft.cuh"
#include "../odd-fft.cuh"
using namespace bytecrt::warp_ntt;
using bytecrt::rns_detail::pow_mod;
using U = uint32_t;
static U mul(U a,U b,U p) { return uint64_t(a)*b%p; }

template<unsigned Len, bool Odd, bool Inverse, unsigned D=Inverse?1:Len/2>
void transform(U* v, const Twiddle* roots, Mod m) {
    if constexpr (D<32) for (unsigned i=0;i<Len;++i) {
        sent[i]=v[i];
        if (i&D) {
            if constexpr (Odd && !Inverse)
                sent[i]=mul(v[i],roots[Len/(2*D)+i/(2*D)].value,m.p);
            if constexpr (!Odd && Inverse)
                sent[i]=mul(v[i],roots[(Len-(i&(D-1))*(Len/(2*D)))&(Len-1)].value,m.p);
        }
    }
    for (lane=0;lane<32;++lane) {
        U regs[Len/32];
        for (unsigned r=0;r<Len/32;++r) regs[r]=v[lane+32*r];
        slot=0;
        active_distance=D;
        if constexpr (Odd) odd_step<Len,D,Inverse>(regs,roots,m,lane);
        else step<Len,D,Inverse>(regs,roots,m,lane);
        assert(slot==(D<32?Len/32:0));
        for (unsigned r=0;r<Len/32;++r) v[lane+32*r]=regs[r];
    }
    if constexpr (Inverse && D<Len/2) transform<Len,Odd,Inverse,2*D>(v,roots,m);
    if constexpr (!Inverse && D>1) transform<Len,Odd,Inverse,D/2>(v,roots,m);
}


#include "../folded-math.hpp"
#include <fstream>
constexpr unsigned n=65536,k=256;
using Coeff=std::array<U,55>;
using Spectrum=std::array<U,8>;
constexpr unsigned bit_reverse3(unsigned i){return ((i&1)<<2)|(i&2)|((i&4)>>2);}

Spectrum digit_forward(const Coeff& x,const folded::Prime& prime,const Mod& m) {
    Spectrum result{},digits;
    for(unsigned d=0;d<8;++d) {
        U chunks[14];
        for(unsigned i=0;i<14;++i) chunks[i]=folded::chunk(x.data(),d,i);
        digits[d]=folded::encode_digit(chunks,d,prime);
    }
    for(unsigned f=0;f<8;++f) for(unsigned d=0;d<8;++d)
        result[bit_reverse3(f)]=add(result[bit_reverse3(f)],multiply(digits[d],prime.roots[(f*d)%8],m),m.p);
    return result;
}

void product(std::vector<U>& a,std::vector<U>& b,const Tables& t,unsigned p) {
    const Mod m=t.mods[p];
    const Twiddle* roots=t.roots.data()+p*(2*k+256);
    auto forward=[&](std::vector<U>& x) {
        std::array<U,k> v;
        for(unsigned id=0;id<256;++id) {
            for(unsigned i=0;i<k;++i) v[i]=x[id+256*i];
            transform<k,true,false>(v.data(),roots,m);
            for(unsigned i=0;i<k;++i) x[id+256*i]=multiply(v[i],t.forward_phase[p*n+id*k+i],m);
        }
    };
    forward(a);forward(b);
    for(unsigned row=0;row<k;++row) {
        U *x=a.data()+256*row,*y=b.data()+256*row;
        transform<256,false,false>(x,roots+2*k,m);
        transform<256,false,false>(y,roots+2*k,m);
        for(unsigned i=0;i<256;++i)x[i]=multiply(x[i],y[i],m);
        transform<256,false,true>(x,roots+2*k,m);
        for(unsigned i=0;i<256;++i)x[i]=multiply(x[i],t.inverse_phase[p*n+row*256+i],m);
    }
    for(unsigned id=0;id<256;++id) {
        std::array<U,k> v;
        for(unsigned i=0;i<k;++i)v[i]=a[id+256*i];
        transform<k,true,true>(v.data(),roots+k,m);
        for(unsigned i=0;i<k;++i)a[id+256*i]=v[i];
    }
}
int main(int argc,char** argv) {
    assert(argc==3);
    std::vector<Coeff> left(n);
    std::array<Coeff,3> taps;
    std::ifstream input(argv[1],std::ios::binary);
    input.read(reinterpret_cast<char*>(left.data()),sizeof(Coeff)*n);
    input.read(reinterpret_cast<char*>(taps.data()),sizeof(taps));
    assert(input);
    auto plan=polynomial_plan();auto tables=make_tables(plan);
    struct Digits {U values[8][16];};
    std::vector<Digits> all(n);
    for(unsigned p=0;p<16;++p) {
        auto m=tables.mods[p];const auto& prime=folded::host_plan.primes[p];
        std::vector<Spectrum> grid(n);
        for(unsigned i=0;i<n;++i)grid[i]=digit_forward(left[i],prime,m);
        Spectrum b0=digit_forward(taps[0],prime,m),b1=digit_forward(taps[1],prime,m),bl=digit_forward(taps[2],prime,m);
        for(unsigned f=0;f<8;++f) {
            std::vector<U> a(n),b(n);
            for(unsigned i=0;i<n;++i)a[i]=grid[i][f];
            b[0]=b0[f];b[1]=b1[f];b[n-1]=bl[f];
            product(a,b,tables,p);
            for(unsigned i=0;i<n;++i)grid[i][f]=a[i];
        }
        for(unsigned i=0;i<n;++i)for(unsigned d=0;d<8;++d) {
            U x=0;
            for(unsigned f=0;f<8;++f)
                x=add(x,multiply(grid[i][bit_reverse3(f)],prime.roots[(8-f*d%8)%8],m),m.p);
            all[i].values[d][p]=multiply(x,prime.recover[d],m);
        }
    }
    std::ofstream output(argv[2],std::ios::binary);
    for(const auto& c:all) {
        U digits[8][16];Coeff out;
        for(unsigned d=0;d<8;++d)folded::recover_digit(c.values[d],digits[d],folded::host_plan);
        folded::assemble(digits,out.data(),folded::host_plan);
        output.write(reinterpret_cast<const char*>(out.data()),sizeof(out));
    }
    assert(output);
}
