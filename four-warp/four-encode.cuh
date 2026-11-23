#pragma once
// SPDX-License-Identifier: Apache-2.0
#include "core/rns-encode-mma.cuh"
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
    __device__ __forceinline__ int32_t neighbor(int32_t v) const {
        return __shfl_xor_sync(0xffffffffu,v,1);
    }
    __device__ __forceinline__ U reduce(const int32_t* sums,const Field& field) const {
        return bytecrt::detail::encode_mma_reduce(sums,field.mod.p,field.reciprocal);
    }
};
// Each lane owns one column and rows row, row+8, row+16, row+24.
template<unsigned Distance=16>
__device__ __forceinline__ void outer_four(U (&v)[4],unsigned row,
                                           const Twiddle* roots,const Mod& mod) {
    constexpr unsigned groups=32/(2*Distance);
    #pragma unroll
    for(unsigned part=0;part<4;++part) {
        const unsigned index=row+8*part;
        if constexpr(Distance>=8) {
            if(part&(Distance/8))continue;
            U a=v[part],b=multiply_twiddle(v[part+Distance/8],roots[groups+index/(2*Distance)],mod);
            v[part]=add(a,b,mod.p);v[part+Distance/8]=sub(a,b,mod.p);
        } else {
            bool upper=(row&Distance)!=0;
            U value=v[part];
            if(upper)value=multiply_twiddle(value,roots[groups+index/(2*Distance)],mod);
            U peer=__shfl_xor_sync(0xffffffffu,value,4*Distance);
            v[part]=upper?sub(peer,value,mod.p):add(value,peer,mod.p);
        }
    }
    if constexpr(Distance>1)outer_four<Distance/2>(v,row,roots,mod);
}

template<unsigned W>
constexpr unsigned shared_bytes() {
    constexpr unsigned L=(W+31)/32,pairs=W==1747?15:8;
    return sizeof(U)*(4*32*(L+1)+pairs*8*4*32);
}

template<unsigned N,unsigned W,unsigned Count>
__global__ __launch_bounds__(32*(Count/2)) void tile_forward(
        const U* a,const U* b,U* output,const U* weights,const Field* fields,
        const U* phases,const Twiddle* roots) {
    constexpr unsigned K=32,J=N/K,L=(W+31)/32,P=4*Count;
    static_assert(Count==(W==1747?30:16));
    extern __shared__ U storage[];
    auto* raw=reinterpret_cast<U (*)[K][L+1]>(storage);
    unsigned base=4*blockIdx.x,operand=blockIdx.y,tid=threadIdx.x,lane=tid%32,pair=tid/32;
    const U* input=operand?b:a;
    for(unsigned i=tid;i<4*K*(L+1);i+=blockDim.x) {
        unsigned limb=i%(L+1),entry=i/(L+1);
        storage[i]=limb==L?0:input[size_t(base+entry/K+J*(entry%K))*L+limb];
    }
    __syncthreads();
    auto* tile=reinterpret_cast<U (*)[4][32]>(storage+4*K*(L+1)+pair*8*4*32);
    const unsigned side=(lane%4)/2;
    // A warp has exclusive use of this tile throughout the prime pair.
    for(unsigned col=0;col<4;++col) {
        for(unsigned half=0;half<2;++half) {
            U result[2][4];
            auto* rows=reinterpret_cast<U (*)[16][L+1]>(&raw[col][16*half]);
            evaluate<W,L,Count>(*rows,lane,fields[2*pair+side],weights+64*pair,result,Native{});
            if(!(lane&1)) {
                #pragma unroll
                for(unsigned h=0;h<2;++h) {
                    unsigned pos=(16*half+lane/4+8*h)^(8*col)^(8*side);
                    #pragma unroll
                    for(unsigned digit=0;digit<4;++digit)tile[2*digit+side][col][pos]=result[h][digit];
                }
            }
        }
    }
    __syncwarp();
    const unsigned col=lane%4,row=lane/4;
    for(unsigned prime_side=0;prime_side<2;++prime_side) {
        unsigned prime=2*pair+prime_side;
        const Mod mod=fields[prime].mod;
        U phase[4];
        #pragma unroll
        for(unsigned part=0;part<4;++part)
            phase[part]=phases[size_t(prime)*N+(base+col)*K+row+8*part];
        for(unsigned digit=0;digit<4;++digit) {
            U v[4];
            #pragma unroll
            for(unsigned part=0;part<4;++part)
                v[part]=tile[2*digit+prime_side][col][(row+8*part)^(8*col)^(8*prime_side)];
            outer_four(v,row,roots+size_t(prime)*(2*K+J),mod);
            #pragma unroll
            for(unsigned part=0;part<4;++part) {
                unsigned frequency=row+8*part;
                unsigned addr=(frequency/2)*(2*J)+2*base+4*(frequency%2)+col;
                output[size_t(operand*P+digit*Count+prime)*N+addr]=multiply(v[part],phase[part],mod);
            }
        }
    }
}
}
