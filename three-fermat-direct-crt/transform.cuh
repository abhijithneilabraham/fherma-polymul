// One GPU block handles one complete transform of large Fermat residues.
#pragma once
#include "diminished.hpp"
#include "product.hpp"
#include <cuda_runtime.h>

namespace fermat_native {
constexpr unsigned lanes=8, threads=256;
__device__ __forceinline__ unsigned group_mask() {return 255u<<((threadIdx.x%32)&~7u);}
template<unsigned W>
__device__ __forceinline__ void copy_number(const Stored<W>& source,Stored<W>& dest,unsigned exponent=0) {
    const unsigned lane=threadIdx.x%lanes;
    for(unsigned i=lane;i<W;i+=lanes) dest.limb[i]=rotated_word(source,i,exponent);
    if(!lane)dest.zero=source.zero;
}

template<unsigned W,bool Both>
__device__ __forceinline__ void butterfly(const Stored<W>& a,const Stored<W>& b,unsigned ea,unsigned eb,
                                          Stored<W>& sum,Stored<W>* diff) {
    constexpr unsigned Local=W/lanes;
    const unsigned lane=threadIdx.x%lanes,mask=group_mask(),base=(threadIdx.x%32)&~7u;
    uint32_t x[Local],y[Local],s[Local],d[Local];
    const bool az=a.zero,bz=b.zero;
    #pragma unroll
    for(unsigned i=0;i<Local;++i) {
        x[i]=rotated_word(a,lane*Local+i,ea);y[i]=rotated_word(b,lane*Local+i,eb);
    }
    __syncwarp(mask);
    bool sz=az&&bz,dz=sz;
    if(az || bz) {
        #pragma unroll
        for(unsigned i=0;i<Local;++i) {
            s[i]=az?y[i]:x[i];
            if constexpr(Both)d[i]=az?~y[i]:x[i];
        }
    } else {
        unsigned g,p;
        local_sum(x,y,false,s,g,p);
        unsigned gs=__ballot_sync(mask,g)>>base, ps=__ballot_sync(mask,p)>>base;
        sz=(ps==255);
        apply_carry(s,carry_into(gs,ps,(1u<<lane)-1,1-carry_into(gs,ps,255,0)));
        if constexpr(Both) {
            local_sum(x,y,true,d,g,p);
            gs=__ballot_sync(mask,g)>>base;ps=__ballot_sync(mask,p)>>base;
            dz=(ps==255);
            apply_carry(d,carry_into(gs,ps,(1u<<lane)-1,1-carry_into(gs,ps,255,0)));
        }
    }
    #pragma unroll
    for(unsigned i=0;i<Local;++i) {
        sum.limb[lane*Local+i]=s[i];
        if constexpr(Both)diff->limb[lane*Local+i]=d[i];
    }
    if(!lane) {
        sum.zero=sz;
        if constexpr(Both)diff->zero=dz;
    }
}

template<unsigned W,unsigned N,bool Inverse,bool Negacyclic>
__device__ void transform(Stored<W>* v) {
    constexpr unsigned bits=32*W;
    for(unsigned half=Inverse?1:N/2;half&&half<N;half=Inverse?2*half:half/2) {
        for(unsigned pair=threadIdx.x/lanes;pair<N/2;pair+=threads/lanes) {
            unsigned j=pair&(half-1), lo=2*(pair-j)+j,hi=lo+half;
            unsigned ea=0,eb=0;
            if constexpr(Inverse) {
                eb=2*bits-j*(bits/half);
            } else if(half==N/2) {
                if constexpr(Negacyclic) {ea=lo*(bits/N);eb=hi*(bits/N);}
            } else {
                // Consume the deferred high-branch rotations from the preceding level.
                const unsigned stride=bits/(2*half);
                if(lo&(2*half))ea=(lo%(2*half))*stride;
                if(hi&(2*half))eb=(hi%(2*half))*stride;
            }
            butterfly<W,true>(v[lo],v[hi],ea,eb,v[lo],&v[hi]);
        }
        __syncthreads();
    }
}

template<unsigned W>
__device__ void multiply_group(const Stored<W>& a,const Stored<W>& b,Stored<W>& output) {
    constexpr unsigned local=W/lanes;
    const unsigned lane=threadIdx.x%lanes,mask=group_mask(),base=(threadIdx.x%32)&~7u;
    if(a.zero || b.zero) {
        for(unsigned i=lane;i<W;i+=lanes)output.limb[i]=0;
        if(!lane)output.zero=1;
        return;
    }
    int64_t co[local];uint32_t words[local];
    #pragma unroll
    for(unsigned j=0;j<local;++j)co[j]=raw_column(a,b,lane*local+j);
    int32_t preceding=__shfl_sync(mask,int32_t(co[local-1]>>32),(lane+7)%8,lanes);
    if(!lane) preceding=-preceding;
    int carry=0;
    #pragma unroll
    for(unsigned owner=0;owner<lanes;++owner) {
        if(lane==owner) {
            #pragma unroll
            for(unsigned j=0;j<local;++j) {
                int64_t high=j?co[j-1]>>32:preceding;
                int64_t total=int64_t(uint32_t(co[j]))+high+carry;
                words[j]=uint32_t(total);carry=int(total>>32);
            }
        }
        carry=__shfl_sync(mask,carry,owner,lanes);
    }
    uint32_t propagate=1;
    for(unsigned j=0;j<local;++j)propagate&=(words[j]==(carry<0?UINT32_MAX:0));
    unsigned all_propagate=__ballot_sync(mask,propagate)>>base;
    const unsigned prior=(1u<<lane)-1;
    int correction=(all_propagate&prior)==prior?-carry:0;
    for(unsigned j=0;j<local;++j) {
        int64_t value=int64_t(words[j])+correction;
        output.limb[lane*local+j]=uint32_t(value);correction=int(value>>32);
    }
    if(!lane)output.zero=(carry!=0 && all_propagate==255);
}

template<unsigned W>
__global__ void outer_forward(const uint32_t* a,const uint32_t* b,Stored<W>* out) {
    extern __shared__ uint32_t scratch[];
    Stored<W>* tile=reinterpret_cast<Stored<W>*>(scratch);
    unsigned channel=blockIdx.x,operand=blockIdx.y,i=threadIdx.x;
    const uint32_t* input=(operand?b:a)+(i*256+channel)*55;
    // The input magnitude is below the square of every chosen Fermat radix.
    int64_t borrow=0;uint32_t any=0;
    for(unsigned j=0;j<W;++j) {
        int64_t value=int64_t(input[j])-(j+W<55?input[j+W]:0u)+borrow;
        tile[i].limb[j]=uint32_t(value);any|=uint32_t(value);borrow=value>>32;
    }
    tile[i].zero=!any&&!borrow;
    if(!borrow) {
        uint32_t decrement=1;
        for(unsigned j=0;j<W;++j) {
            uint32_t w=tile[i].limb[j];tile[i].limb[j]=w-decrement;decrement=(w<decrement);
        }
    }
    __syncthreads();
    transform<W,256,false,true>(tile);
    for(unsigned slot=threadIdx.x/lanes;slot<256;slot+=threads/lanes)
        copy_number(tile[slot],out[operand*65536+slot*256+channel]);
}

template<unsigned W>
__global__ void inner_forward(const Stored<W>* input,Stored<W>* out) {
    extern __shared__ uint32_t scratch[];
    Stored<W>* tile=reinterpret_cast<Stored<W>*>(scratch);
    unsigned slot=blockIdx.x,operand=blockIdx.y;
    for(unsigned i=threadIdx.x/lanes;i<256;i+=threads/lanes) {
        copy_number(input[operand*65536+slot*256+i],tile[i]);
        for(unsigned j=threadIdx.x%lanes;j<W;j+=lanes)tile[256+i].limb[j]=0;
        if(!(threadIdx.x%lanes))tile[256+i].zero=1;
    }
    __syncthreads();
    transform<W,512,false,false>(tile);
    for(unsigned i=threadIdx.x/lanes;i<512;i+=threads/lanes)
        copy_number(tile[i],out[(operand*256+slot)*512+i]);
}

template<unsigned W>
__global__ void inner_inverse(const Stored<W>* input,Stored<W>* out) {
    extern __shared__ uint32_t scratch[];
    Stored<W>* tile=reinterpret_cast<Stored<W>*>(scratch);
    constexpr unsigned bits=32*W;
    unsigned slot=blockIdx.x;
    for(unsigned i=threadIdx.x/lanes;i<512;i+=threads/lanes)
        multiply_group(input[slot*512+i],input[(256+slot)*512+i],tile[i]);
    __syncthreads();
    transform<W,512,true,false>(tile);
    unsigned twist=(2*(__brev(slot)>>24)+1)*(bits/256);
    for(unsigned i=threadIdx.x/lanes;i<256;i+=threads/lanes)
        butterfly<W,false>(tile[i],tile[i+256],2*bits-9,2*bits+twist-9,out[i*256+slot],nullptr);
}

template<unsigned W>
__global__ void outer_inverse(const Stored<W>* input,Stored<W>* out) {
    extern __shared__ uint32_t scratch[];
    Stored<W>* tile=reinterpret_cast<Stored<W>*>(scratch);
    constexpr unsigned bits=32*W;
    unsigned channel=blockIdx.x;
    for(unsigned i=threadIdx.x/lanes;i<256;i+=threads/lanes)
        copy_number(input[channel*256+i],tile[i]);
    __syncthreads();
    transform<W,256,true,true>(tile);
    // Scale each residue by the inverse of the other two factors.
    // A term 2^e multiplies by rotating the diminished-one representation.
    auto& sum=tile[256+threadIdx.x/lanes];
    for(unsigned i=threadIdx.x/lanes;i<256;i+=threads/lanes) {
        const unsigned shift=4*bits-8-i*(bits/256)-(W==32?1:2);
        auto first=[&](unsigned e) {
            copy_number(tile[i],sum,shift+e);
            __syncwarp(group_mask());
        };
        auto add=[&](unsigned e) {
            butterfly<W,false>(sum,tile[i],0,shift+e,sum,nullptr);
            __syncwarp(group_mask());
        };
        if constexpr(W==48) {
            first(5*256);add(bits+4*256);add(bits+3*256);
            add(bits+2*256);add(bits+2*256+1);
            add(bits+256);add(bits+257);add(bits);
        } else if constexpr(W==40) {
            first(0);add(2);add(256);add(257);add(2*256);
            add(bits+3*256);add(bits+4*256);add(bits+4*256+1);
        } else {
            first(2*256);add(3*256);
        }
        copy_number(sum,out[i*256+channel]);
        __syncwarp(group_mask());
    }
}
} // namespace fermat_native
