// SPDX-License-Identifier: MIT
#pragma once
#include "folded-kernels.cuh"
#include "crt-weights.hpp"
namespace folded {
__global__ void decode_mma(const uint32_t* weighted, uint32_t* output, const Plan* plan,
                           const Mod* mods, const CrtWeights* weights) {
    constexpr unsigned rows = 16, digit_rows = rows*8;
    __shared__ union Workspace {
        uint32_t frequency[128][rows+1];
        uint64_t columns[digit_rows][16];
    } workspace;
    __shared__ uint32_t values[digit_rows][17], digits[rows][8][16];
    __shared__ uint32_t alpha[digit_rows], scratch[rows][64];
    unsigned tid = threadIdx.x, lane = tid%32, warp = tid/32;
    unsigned id = blockIdx.x%256, base = (blockIdx.x/256)*rows;
    for (unsigned index = tid; index < 128*rows; index += blockDim.x) {
        unsigned f = index/rows, row = index%rows;
        workspace.frequency[f][row] = weighted[(id*128+f)*256+base+row];
    }
    __syncthreads();
    if (tid < digit_rows) {
        unsigned row = tid/8, digit = tid%8;
        uint64_t estimate = uint64_t{1}<<47;
        for (unsigned p=0; p<16; ++p) {
            const Prime& prime = plan->primes[p];
            Mod mod = mods[p];
            uint32_t x = workspace.frequency[p*8+digit][row];
            x = digit_step<true,1>(x,digit,prime,mod);
            x = digit_step<true,2>(x,digit,prime,mod);
            x = digit_step<true,4>(x,digit,prime,mod);
            uint32_t v = multiply(x,prime.recover[digit],mod);
            values[tid][p] = v;
            estimate += uint64_t(v)*weights->reciprocal48[p];
        }
        alpha[tid] = estimate >> 48;
    }
    __syncthreads();
    unsigned g=lane/4, t=lane%4, first=16*warp+g;
    for (unsigned group=0; group<8; ++group) {
        uint32_t c[4]={};
        #pragma unroll
        for (unsigned half=0; half<2; ++half) {
            unsigned p=8*half+t;
            const uint32_t* w = weights->words[group][half][lane];
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
                : "r"(values[first][p]),"r"(values[first+8][p]),
                  "r"(values[first][p+4]),"r"(values[first+8][p+4]),"r"(w[0]),"r"(w[1]));
        }
        #pragma unroll
        for (unsigned h=0; h<2; ++h) {
            uint32_t a=c[2*h], b=c[2*h+1];
            uint32_t e=__shfl_xor_sync(0xffffffffu,a,1);
            uint32_t f=__shfl_xor_sync(0xffffffffu,b,1);
            if (!(lane & 1))
                workspace.columns[first+8*h][2*group+t/2] =
                    uint64_t(a)+(uint64_t(b)<<8)+(uint64_t(e)<<16)+(uint64_t(f)<<24);
        }
    }
    __syncthreads();
    if (tid < digit_rows) {
        unsigned row=tid/8, digit=tid%8;
        finish_crt_words(workspace.columns[tid],alpha[tid],*plan,digits[row][digit]);
        // All eight digits of a coefficient have the same warp.
        __syncwarp();
        folded_carry::assemble_group(digits[row],*plan,digit,scratch[row],
            output+(id+(base+row)*256)*L);
    }
}
}
