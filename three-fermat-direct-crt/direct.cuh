// Exact centered CRT for N(q-1)^2 < 2^3510. One group owns one coefficient.
#pragma once
#include "transform.cuh"
namespace fermat_native::direct {
constexpr unsigned count=128, local=count/lanes;
constexpr uint32_t gap=9699327, top_mask=(1u<<19)-1;
struct Scratch {int64_t raw[count];uint32_t word[count];};

template<unsigned W>
__device__ inline int64_t term(const Stored<W>& t,unsigned j,unsigned offset) {
    unsigned index=j-offset;
    if(t.zero || index>=W)return 0;
    return int64_t(t.limb[index])+(index==0);
}

template<unsigned W,unsigned D,unsigned E>
__device__ inline int64_t cofactor(const Stored<W>& t,unsigned j) {
    return term(t,j,0)+term(t,j,D)+term(t,j,E)+term(t,j,D+E);
}

// Split each signed column into adjacent words before propagating. The
// residual carry into a lane then fits in {-1,0,1} for every use below.
__device__ inline void normalize(Scratch& s) {
    unsigned lane=threadIdx.x%lanes, mask=group_mask(), begin=local*lane;
    __syncwarp(mask);
    int carry=0;
    bool zeros=true, ones=true;
    for(unsigned j=begin;j<begin+local;++j) {
        int64_t v=int64_t(uint32_t(s.raw[j]))+(j?s.raw[j-1]>>32:0)+carry;
        uint32_t word=uint32_t(v);
        s.word[j]=word;zeros&=word==0;ones&=word==UINT32_MAX;
        carry=int(v>>32);
    }
    int prefix=0,initial=0;
    for(unsigned owner=0;owner<lanes;++owner) {
        if(lane==owner) {
            initial=prefix;
            prefix=carry+(initial<0&&zeros?-1:initial>0&&ones?1:0);
        }
        prefix=__shfl_sync(mask,prefix,owner,lanes);
    }
    for(unsigned j=begin;j<begin+local;++j) {
        int64_t v=int64_t(s.word[j])+initial;
        s.word[j]=uint32_t(v);initial=int(v>>32);
    }
    __syncwarp(mask);
}

__device__ inline uint32_t at_bit(const uint32_t* words,unsigned bit) {
    unsigned j=bit/32,shift=bit%32;
    return shift?(words[j]>>shift)|(words[j+1]<<(32-shift)):words[j];
}
__device__ inline uint32_t chunk(const uint32_t* words,unsigned start,unsigned j) {
    if(j>=55)return 0;
    uint32_t word=at_bit(words,start+32*j);
    return j==54?word&top_mask:word;
}

__device__ inline void recover(const Stored<48>& a,const Stored<40>& b,const Stored<32>& c,
                              uint32_t* dest,Scratch& s) {
    unsigned lane=threadIdx.x%lanes, mask=group_mask();
    unsigned top=(a.zero?0:a.limb[47]>>29)+(b.zero?0:b.limb[39]>>29)+(c.zero?0:c.limb[31]>>29);
    const unsigned quotient=(top+4)/8;
    for(unsigned j=lane;j<count;j+=lanes) {
        int64_t value=cofactor<48,40,32>(a,j)+cofactor<40,48,32>(b,j)+cofactor<32,48,40>(c,j);
        if(j==0||j==32||j==40||j==48||j==72||j==80||j==88||j==120)value-=quotient;
        s.raw[j]=value;
    }
    normalize(s);
    bool negative=(s.word[127]>>31)!=0;
    if(negative) {
        for(unsigned j=lane;j<count;j+=lanes)s.raw[j]=int64_t(~s.word[j])+(j==0);
        normalize(s);
    }
    uint64_t upper=uint64_t(at_bit(s.word,3494))*gap*gap;
    for(unsigned j=lane;j<count;j+=lanes) {
        uint64_t extra=j==0?uint32_t(upper):j==1?upper>>32:0;
        s.raw[j]=chunk(s.word,0,j)+uint64_t(gap)*chunk(s.word,1747,j)+extra;
    }
    normalize(s);
    // Bias by gap so a carry across bit 1747 detects a quotient of one.
    uint64_t correction=(uint64_t(at_bit(s.word,1747))+1)*gap;
    for(unsigned j=lane;j<count;j+=lanes) {
        uint64_t extra=j==0?uint32_t(correction):j==1?correction>>32:0;
        s.raw[j]=uint64_t(chunk(s.word,0,j))+extra;
    }
    normalize(s);
    bool crossed=(s.word[54]>>19)!=0;
    for(unsigned j=lane;j<count;j+=lanes)
        s.raw[j]=int64_t(chunk(s.word,0,j))-(j==0&&!crossed?gap:0);
    normalize(s);
    if(negative) {
        unsigned nonzero=0;
        for(unsigned j=lane;j<55;j+=lanes)nonzero|=s.word[j];
        bool subtract=__ballot_sync(mask,nonzero)!=0;
        if(subtract) {
            for(unsigned j=lane;j<count;j+=lanes) {
                uint32_t q=j==0?0u-gap:j<54?UINT32_MAX:j==54?top_mask:0;
                s.raw[j]=int64_t(q)-s.word[j];
            }
            normalize(s);
        }
    }
    for(unsigned j=lane;j<55;j+=lanes)dest[j]=s.word[j];
}
} // namespace fermat_native::direct
