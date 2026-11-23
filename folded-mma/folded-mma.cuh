// SPDX-License-Identifier: MIT
#pragma once
#include "folded-kernels.cuh"
namespace folded {
struct EncodeWeights {
    uint32_t words[8][32][2];
    uint32_t twist_mont[16][8];
};

inline EncodeWeights make_encode_weights(const Plan& plan) {
    EncodeWeights w{};
    for (unsigned p=0; p<16; ++p) {
        uint64_t mod=plan.primes[p].p, power=1, R=(uint64_t(1)<<32)%mod;
        for (unsigned d=0; d<8; ++d) w.twist_mont[p][d]=R*plan.primes[p].digit_weights[d][0]%mod;
        for (unsigned k=0; k<32; ++k) {
            for (unsigned j=0; j<4; ++j) {
                unsigned column=4*p+j, lane=4*(column%8)+(k%16)/4;
                w.words[column/8][lane][(k%32)/16] |= uint32_t((power>>(8*j))&255) << (8*(k%4));
            }
            power=power*256%mod;
        }
    }
    return w;
}

template<unsigned Distance>
__device__ __forceinline__ uint32_t digit_dif_even(uint32_t x,unsigned d,const Prime& prime,const Mod& mod) {
    // Digit index occupies the high three lane bits. Even lanes own residues.
    uint32_t peer=__shfl_xor_sync(0x55555555u,x,4*Distance);
    if (!(d & Distance)) return add(x,peer,mod.p);
    x=sub(peer,x,mod.p);
    if constexpr (Distance>1)
        x=multiply(x,prime.roots[(d%Distance)*(4/Distance)],mod);
    return x;
}

__global__ void encode_mma(const uint32_t* input,uint32_t* planes,const Plan* plan,
                           const Mod* mods,const EncodeWeights* constants) {
    __shared__ union Tile {
        uint32_t raw[32][L];
        uint32_t spectrum[128][33];
    } tile;
    __shared__ uint32_t digits[256][12];
    unsigned tid=threadIdx.x, lane=tid%32, warp=tid/32, g=lane/4, t=lane%4;
    unsigned id=blockIdx.x%256, base=blockIdx.x/256*32;
    for (unsigned i=tid; i<32*L; i+=256)
        tile.raw[i/L][i%L]=input[(id+(base+i/L)*256)*L+i%L];
    __syncthreads();
    #pragma unroll
    for (unsigned i=0;i<7;++i) {
        uint32_t lo=chunk(tile.raw[tid/8],tid%8,2*i);
        uint32_t hi=chunk(tile.raw[tid/8],tid%8,2*i+1);
        digits[tid][i]=lo|(hi<<16);
    }
    digits[tid][7]=0;
    __syncthreads();
    uint32_t b0=constants->words[warp][lane][0], b1=constants->words[warp][lane][1];
    unsigned p=warp*2+t/2;
    Mod mod=mods[p];
    for (unsigned pair=0;pair<16;++pair) {
        unsigned first=16*pair+g;
        uint32_t c[4]={};
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
            : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
            : "r"(digits[first][t]),"r"(digits[first+8][t]),
              "r"(digits[first][t+4]),"r"(digits[first+8][t+4]),"r"(b0),"r"(b1));
        #pragma unroll
        for(unsigned half=0;half<2;++half) {
            uint32_t s0=c[2*half],s1=c[2*half+1];
            uint32_t s2=__shfl_xor_sync(0xffffffffu,s0,1);
            uint32_t s3=__shfl_xor_sync(0xffffffffu,s1,1);
            if (!(lane&1)) {
                const Prime& prime=plan->primes[p];
                uint64_t v=uint64_t(s0)+(uint64_t(s1)<<8)+(uint64_t(s2)<<16)+(uint64_t(s3)<<24);
                uint64_t rem=v-__umul64hi(v,prime.reciprocal)*prime.p;
                uint32_t x=uint32_t(rem>=prime.p ? rem-prime.p : rem);
                x=multiply(x,constants->twist_mont[p][g],mod);
                x=digit_dif_even<4>(x,g,prime,mod);
                x=digit_dif_even<2>(x,g,prime,mod);
                x=digit_dif_even<1>(x,g,prime,mod);
                tile.spectrum[8*p+g][2*pair+half]=x;
            }
        }
    }
    __syncthreads();
    for (unsigned i=tid;i<128*32;i+=256)
        planes[(i/32)*N+id*256+base+i%32]=tile.spectrum[i/32][i%32];
}
}
