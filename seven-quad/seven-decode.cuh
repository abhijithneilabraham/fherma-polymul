// SPDX-License-Identifier: Apache-2.0
#pragma once


#include "warp-carry.cuh"
#include "odd-fft.cuh"
#include "inverse-field.hpp"
namespace seven_decode {
struct CarryWarp {
    __device__ __forceinline__ uint32_t ballot(bool value) const {
        return __ballot_sync(0xffffffffu,value);
    }
    __device__ __forceinline__ uint32_t get(uint32_t value,unsigned lane) const {
        return __shfl_sync(0xffffffffu,value,lane);
    }
};

template<unsigned W,unsigned Count>
struct QuadStorage {
    static constexpr unsigned digits=((W+31)/32+2)|1;
    static constexpr unsigned bytes=32*((4*7*Count+7+31)/32);
    static constexpr unsigned stride=bytes/4+4;
    alignas(16) uint8_t columns[4][32][4*stride];
    uint32_t lows[2][32][digits],highs[2][32][digits];
};

template<unsigned N,unsigned W,unsigned Count>
__device__ __forceinline__ void reconstruct(unsigned id,unsigned column,QuadStorage<W,Count>& tile,
                       uint32_t* out,const uint8_t* weights,const uint32_t* reciprocal48,
                       uint32_t gap,const seven_inverse::Field* fields) {
    constexpr unsigned MT=32,L=(W+31)/32,groups=(L+13)/14,K=32,J=N/K,planes=7*Count;
    constexpr unsigned Kpad=QuadStorage<W,Count>::bytes,word_stride=QuadStorage<W,Count>::stride;
    static_assert(Kpad*255u*255u<0x80000000u);
    unsigned local=threadIdx.x%256,warp=local/32,lane=local%32,g=lane/4,t=lane%4;
    auto& shared_a=tile.columns[column];
    auto* shared_words=reinterpret_cast<uint32_t*>(shared_a);
    auto& lows=tile.lows[threadIdx.x/256];
    auto& highs=tile.highs[threadIdx.x/256];
    for(unsigned work=local;work<MT*Count;work+=256){
        unsigned row=work/Count,prime=work%Count;
        uint32_t* terms=shared_words+row*word_stride;
        uint32_t x[7],y[7];
        #pragma unroll
        for(unsigned d=0;d<7;++d)x[d]=terms[d*Count+prime];
        const auto& f=fields[prime];
        rader7::transform(x,f.inverse,f.mod,y);
        #pragma unroll
        for(unsigned d=0;d<7;++d)terms[d*Count+prime]=bytecrt::warp_ntt::multiply_twiddle(y[d],f.recovery[d],f.mod);
    }
    __syncthreads();
    // Four lanes own each digit. All lanes participate in the shuffles.
    for(unsigned row=warp;row<MT;row+=8) {
        uint32_t* terms=shared_words+row*word_stride;
        unsigned digit=lane/4;
        uint64_t sum=0;
        if(digit<7) {
            for(unsigned j=lane%4;j<Count;j+=4)
                sum+=uint64_t(terms[digit*Count+j])*reciprocal48[j];
        }
        sum+=__shfl_down_sync(0xffffffffu,sum,2,4);
        sum+=__shfl_down_sync(0xffffffffu,sum,1,4);
        if(digit<7 && lane%4==0)
            shared_a[row][4*planes+digit]=static_cast<uint8_t>((sum+(uint64_t{1}<<47))>>48);
    }
    __syncthreads();
    if(warp<7) {
        for (unsigned group = 0; group < groups; ++group) {
            constexpr unsigned rows=groups*7,header=128*((4*(rows+1)+127)/128);
            constexpr unsigned weights_start=128*((header+8*rows*(Kpad/32)+127)/128);
            const auto* offsets=reinterpret_cast<const uint32_t*>(weights);
            const unsigned slot=7*group+warp, stop=offsets[slot+1];
            int32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            int32_t c4 = 0, c5 = 0, c6 = 0, c7 = 0;
            for (unsigned cursor=offsets[slot];cursor<stop;++cursor) {
                const uint8_t* indices=weights+header+8*cursor;
                unsigned first=4*indices[t],second=4*indices[4+t];
                const auto* tile=reinterpret_cast<const uint32_t*>(weights+weights_start)+64*cursor;
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(shared_a[g] + first);
                const uint32_t a1 = *reinterpret_cast<const uint32_t*>(shared_a[g + 8] + first);
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(shared_a[g] + second);
                const uint32_t a3 = *reinterpret_cast<const uint32_t*>(shared_a[g + 8] + second);
                uint32_t b0=tile[lane],b1=tile[32+lane];
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                const uint32_t a4 = *reinterpret_cast<const uint32_t*>(shared_a[g + 16] + first);
                const uint32_t a5 = *reinterpret_cast<const uint32_t*>(shared_a[g + 24] + first);
                const uint32_t a6 = *reinterpret_cast<const uint32_t*>(shared_a[g + 16] + second);
                const uint32_t a7 = *reinterpret_cast<const uint32_t*>(shared_a[g + 24] + second);
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+r"(c4), "+r"(c5), "+r"(c6), "+r"(c7)
                    : "r"(a4), "r"(a5), "r"(a6), "r"(a7), "r"(b0), "r"(b1));
            }
            const int32_t pairs[4][2]={{c0,c1},{c2,c3},{c4,c5},{c6,c7}};
            #pragma unroll
            for(unsigned part=0;part<4;++part) {
                uint32_t x=pairs[part][0],y=pairs[part][1];
                uint32_t z=__shfl_xor_sync(0xffffffffu,x,1),w=__shfl_xor_sync(0xffffffffu,y,1);
                bytecrt::warp_carry::store_dot<L>(lows,highs,lane,warp,group,part,x,y,z,w);
            }
        }

    }
    __syncthreads();
    for(unsigned r=warp;r<MT;r+=8) {
        unsigned coefficient=id+r*J;
        bytecrt::warp_carry::finish<W,L>(lows[r],highs[r],gap,out+size_t(coefficient)*L,lane,CarryWarp{});
    }
}

template<unsigned N,unsigned W,unsigned Count>
__global__ void inverse_decode(const uint32_t* input,uint32_t* out,const uint8_t* weights,
                       const uint32_t* reciprocal48,uint32_t gap,const bytecrt::warp_ntt::Twiddle* roots,
                       const seven_inverse::Field* fields) {
    constexpr unsigned K=32,J=N/K,planes=7*Count;
    using Storage=QuadStorage<W,Count>;
    extern __shared__ __align__(16) uint32_t quad_storage[];
    auto& tile=*reinterpret_cast<Storage*>(quad_storage);
    unsigned lane=threadIdx.x%32,warp=threadIdx.x/32;
    for(unsigned plane=warp;plane<Storage::bytes/4;plane+=16) {
        uint4 packed{};
        if(plane<planes) {
            const auto* vectors=reinterpret_cast<const uint4*>(input+size_t(plane)*N);
            packed=vectors[blockIdx.x*K+lane];
        }
        uint32_t values[]={packed.x,packed.y,packed.z,packed.w};
        #pragma unroll
        for(unsigned column=0;column<4;++column) {
            uint32_t v[]={values[column]};
            if(plane<planes) {
                unsigned prime=plane%Count;
                bytecrt::warp_ntt::odd_fft<K,true>(v,roots+size_t(prime)*(2*K+J)+K,fields[prime].mod,lane);
            }
            auto* row=reinterpret_cast<uint32_t*>(tile.columns[column][lane]);
            row[plane]=v[0];
        }
    }
    __syncthreads();
    for(unsigned pair=0;pair<2;++pair) {
        unsigned column=2*pair+threadIdx.x/256;
        reconstruct<N,W,Count>(4*blockIdx.x+column,column,tile,out,weights,reciprocal48,gap,fields);
        __syncthreads();
    }
}
}
