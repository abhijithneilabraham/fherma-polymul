// CRT conversion for the ranked three-Fermat basis, using sparse multiples.
#pragma once
#include "diminished.hpp"
#ifdef __CUDACC__
#define REC_HD __host__ __device__ inline
#else
#define REC_HD inline
#endif
namespace fermat_native {
namespace recovery {
struct Integer {
    uint32_t word[122];
    unsigned used=0;
};
REC_HD void trim(Integer& x) {while(x.used && x.word[x.used-1]==0)--x.used;}
REC_HD int compare(const Integer& a,const Integer& b) {
    if(a.used!=b.used)return a.used>b.used?1:-1;
    for(unsigned i=a.used;i>0;--i)if(a.word[i-1]!=b.word[i-1])return a.word[i-1]>b.word[i-1]?1:-1;
    return 0;
}
REC_HD void add(Integer& a,const Integer& b,unsigned shift=0) {
    if(!b.used)return;
    while(a.used<shift)a.word[a.used++]=0;
    uint64_t carry=0;
    for(unsigned i=0;i<b.used || carry;++i) {
        unsigned k=i+shift;
        if(k==a.used)a.word[a.used++]=0;
        uint64_t sum=uint64_t(a.word[k])+(i<b.used?b.word[i]:0u)+carry;
        a.word[k]=uint32_t(sum);carry=sum>>32;
    }
}
REC_HD void subtract(Integer& a,const Integer& b) {
    uint64_t borrow=0;
    for(unsigned i=0;i<a.used;++i) {
        uint64_t sub=uint64_t(i<b.used?b.word[i]:0)+borrow;
        borrow=a.word[i]<sub;
        a.word[i]-=uint32_t(sub);
    }
    trim(a);
}
REC_HD Integer one() {Integer result;result.used=1;result.word[0]=1;return result;}
REC_HD Integer modulus(unsigned words) {
    Integer r=one(),unit=one();add(r,unit,words);return r;
}
REC_HD void shift_right_one(Integer& a) {
    uint32_t carry=0;
    for(unsigned i=a.used;i>0;--i) {
        uint32_t old=a.word[i-1];a.word[i-1]=(old>>1)|carry;carry=old<<31;
    }
    trim(a);
}
REC_HD Integer difference(Integer a,const Integer& b,unsigned words) {
    if(compare(a,b)<0) {auto m=modulus(words);add(a,m);}
    subtract(a,b);return a;
}
REC_HD Integer fold(const Integer& x,unsigned words) {
    Integer r;
    for(unsigned chunks=(x.used+words-1)/words;chunks>0;--chunks) {
        Integer chunk;
        unsigned begin=(chunks-1)*words;
        chunk.used=x.used-begin<words?x.used-begin:words;
        for(unsigned i=0;i<chunk.used;++i)chunk.word[i]=x.word[begin+i];
        trim(chunk);r=difference(chunk,r,words);
    }
    return r;
}
REC_HD void halve(Integer& x,unsigned words) {
    if(x.used && (x.word[0]&1)) {auto m=modulus(words);add(x,m);}
    shift_right_one(x);
}
template<unsigned W>
REC_HD Integer decode(const Stored<W>& stored) {
    Integer x;
    if(stored.zero)return x;
    for(unsigned i=0;i<W;++i)x.word[i]=stored.limb[i];
    x.used=W;trim(x);auto unit=one();add(x,unit);return x;
}

REC_HD Integer reconstruct(const Stored<48>& a,const Stored<40>& b,const Stored<32>& c) {
    Integer x=decode(a);
    Integer d=difference(decode(b),fold(x,40),40);
    Integer poly=d;
    for(unsigned k=1;k<5;++k)add(poly,d,8*k);
    Integer t=fold(poly,40);halve(t,40);
    add(x,t);add(x,t,48);
    d=difference(decode(c),fold(x,32),32);
    poly.used=0;add(poly,d,16);add(poly,d,24);
    t=fold(poly,32);halve(t,32);
    add(x,t);add(x,t,40);add(x,t,48);add(x,t,88);
    return x;
}

REC_HD void output(Integer x,uint32_t* destination) {
    Integer product=one(),unit=one();
    for(unsigned offset=4;offset<=6;++offset)add(product,unit,8*offset);
    for(unsigned offset=9;offset<=11;++offset)add(product,unit,8*offset);
    add(product,unit,120);
    Integer midpoint=product;shift_right_one(midpoint);
    bool negative=compare(x,midpoint)>0;
    if(negative) {subtract(product,x);x=product;}
    constexpr uint32_t gap=9699327,mask=(1u<<19)-1;
    while(x.used>55 || (x.used==55 && x.word[54]>mask)) {
        Integer high;
        high.used=x.used-54;
        for(unsigned i=0;i<high.used;++i) {
            unsigned j=i+54;
            high.word[i]=(x.word[j]>>19)|(j+1<x.used?x.word[j+1]<<13:0u);
        }
        trim(high);
        x.used=55;x.word[54]&=mask;trim(x);
        uint64_t carry=0;
        for(unsigned i=0;i<high.used;++i) {
            uint64_t v=uint64_t(high.word[i])*gap+carry;
            high.word[i]=uint32_t(v);carry=v>>32;
        }
        if(carry)high.word[high.used++]=uint32_t(carry);
        add(x,high);
    }
    Integer q;
    q.used=55;q.word[0]=0u-gap;
    for(unsigned j=1;j<54;++j)q.word[j]=UINT32_MAX;
    q.word[54]=mask;
    if(compare(x,q)>=0)subtract(x,q);
    if(negative && x.used) {subtract(q,x);x=q;}
    for(unsigned j=0;j<55;++j)destination[j]=j<x.used?x.word[j]:0;
}
} // namespace recovery
} // namespace fermat_native
#undef REC_HD
