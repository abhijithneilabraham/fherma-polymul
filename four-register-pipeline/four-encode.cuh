#pragma once
// SPDX-License-Identifier: Apache-2.0
#include "word-reduce.hpp"
#include "encode.hpp"
#include "odd-fft.cuh"
namespace four_encode {
struct Native {
    __device__ __forceinline__ void mma(int32_t* c,U a0,U a1,U a2,U a3,U b0,U b1) const {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
                     "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                    : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
                    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
};
constexpr unsigned threads=192;

template<unsigned W>
constexpr unsigned shared_bytes() {
    constexpr unsigned words=W==1747?16:8;
    return 4*4*32*words*sizeof(U);
}

// Store aligned groups of four words. The low row bits distribute each
// eight-row matrix load over the 32 shared banks for both widths.
template<unsigned Words>
__device__ __forceinline__ unsigned shared_word(unsigned row,unsigned word) {
    return row*Words+(word^((4*(row/(32/Words)))%Words));
}

template<unsigned Words>
__device__ __forceinline__ void load_fragment(const U* rows,unsigned first,unsigned lane,U (&a)[4]) {
#ifdef __CUDA_ARCH__
    unsigned source_row=lane%16,word=first+4*(lane/16);
    unsigned address=uint32_t(__cvta_generic_to_shared(rows+shared_word<Words>(source_row,word)));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]) : "r"(address) : "memory");
#else
    for(unsigned m=0;m<4;++m) {
        unsigned row=lane/4+8*(m%2),word=first+4*(m/2)+lane%4;
        a[m]=rows[shared_word<Words>(row,word)];
    }
#endif
}

template<unsigned W,unsigned Count>
__device__ __forceinline__ void evaluate_tile(const U* packed,unsigned col,unsigned half,unsigned pair,
        unsigned lane,const U* weights,const Field& field,U (&values)[4]) {
    constexpr unsigned words=W==1747?16:8,chunks=words/8;
    U digits[4];
    const bool odd=(lane&1)!=0;
    #pragma unroll
    for(unsigned d=0;d<4;++d) {
        int32_t acc[4]={};
        const U* rows=packed+((col*4+d)*32+half*16)*words;
        #pragma unroll
        for(unsigned chunk=0;chunk<chunks;++chunk) {
            U a[4];
            load_fragment<words>(rows,8*chunk,lane,a);
            const U* b=weights+((d*chunks+chunk)*(Count/2)+pair)*64;
            Native{}.mma(acc,a[0],a[1],a[2],a[3],b[lane],b[32+lane]);
        }
        int32_t peer0=__shfl_xor_sync(0xffffffffu,odd?acc[0]:acc[2],1);
        int32_t peer1=__shfl_xor_sync(0xffffffffu,odd?acc[1]:acc[3],1);
        int32_t sums[4]={odd?peer0:acc[0],odd?peer1:acc[1],odd?acc[2]:peer0,odd?acc[3]:peer1};
        digits[d]=two_digit::reduce_dot(sums,field.mod.p,field.reciprocal48);
    }
    transform(digits,field.forward,field.mod,values);
}

template<unsigned Distance=8>
__device__ __forceinline__ U outer_lane(U v,unsigned row,const Twiddle* roots,const Mod& m) {
    constexpr unsigned groups=32/(2*Distance);
    bool upper=(row&Distance)!=0;
    if(upper)v=multiply_twiddle(v,roots[groups+row/(2*Distance)],m);
    constexpr unsigned lane_bit=Distance==8?1:4*Distance;
    U peer=__shfl_xor_sync(0xffffffffu,v,lane_bit);
    U out=upper?sub(peer,v,m.p):add(v,peer,m.p);
    if constexpr(Distance>1)return outer_lane<Distance/2>(out,row,roots,m);
    return out;
}

template<unsigned N,unsigned W,unsigned Count>
__global__ __launch_bounds__(threads) void tile_forward(
        const U* a,const U* b,U* output,const U* weights,const Field* fields,
        const U* phases,const Twiddle* roots) {
    constexpr unsigned K=32,J=N/K,L=(W+31)/32,P=4*Count,words=W==1747?16:8;
    extern __shared__ U storage[];
    const unsigned tid=threadIdx.x,lane=tid%32,warp=tid/32,base=4*blockIdx.x,operand=blockIdx.y;
    const U* input=operand?b:a;
    for(unsigned entry=warp;entry<4*K;entry+=threads/32) {
        unsigned col=entry/K,row=entry%K;
        const U* raw=input+size_t(base+col+J*row)*L;
        for(unsigned i=lane;i<4*words;i+=32) {
            unsigned digit=i/words,word=i%words;
            storage[(col*4+digit)*32*words+shared_word<words>(row,word)]=digit_word<W>(raw,digit,word);
        }
    }
    __syncthreads();
    const unsigned short_row=lane/4+8*(lane&1);
    for(unsigned pair=warp;pair<Count/2;pair+=threads/32) {
        unsigned prime=2*pair+(lane%4)/2;
        Field field=fields[prime];
        const Twiddle* table=roots+size_t(prime)*(2*K+J);
        U results[4][2][4];
        #pragma unroll
        for(unsigned col=0;col<4;++col) {
            U lower[4],upper[4];
            evaluate_tile<W,Count>(storage,col,0,pair,lane,weights,field,lower);
            evaluate_tile<W,Count>(storage,col,1,pair,lane,weights,field,upper);
            #pragma unroll
            for(unsigned digit=0;digit<4;++digit) {
                U rotated=multiply_twiddle(upper[digit],table[1],field.mod);
                U lo=add(lower[digit],rotated,field.mod.p);
                U hi=sub(lower[digit],rotated,field.mod.p);
                results[digit][0][col]=outer_lane(lo,short_row,table,field.mod);
                results[digit][1][col]=outer_lane(hi,short_row+16,table,field.mod);
            }
        }
        #pragma unroll
        for(unsigned half=0;half<2;++half) {
            unsigned row=short_row+16*half;
            U phase[4];
            #pragma unroll
            for(unsigned col=0;col<4;++col)phase[col]=phases[size_t(prime)*N+(base+col)*K+row];
            #pragma unroll
            for(unsigned digit=0;digit<4;++digit) {
                U* out=output+size_t(operand*P+digit*Count+prime)*N+(row/2)*(2*J)+2*base+(row%2)*4;
                U v[4];
                #pragma unroll
                for(unsigned col=0;col<4;++col)v[col]=multiply(results[digit][half][col],phase[col],field.mod);
#ifdef __CUDA_ARCH__
                *reinterpret_cast<uint4*>(out)=make_uint4(v[0],v[1],v[2],v[3]);
#else
                for(unsigned col=0;col<4;++col)out[col]=v[col];
#endif
            }
        }
    }
}
}
